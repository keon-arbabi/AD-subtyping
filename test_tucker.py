import sys, os
import json
import tensorly as tl
import optuna
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk, set_num_threads
from utils import debug, print_df, savefig
from ryp import r, to_r

tl.set_backend('numpy') 
set_num_threads(-1)
debug(third_party=True)

import warnings
warnings.filterwarnings('ignore')

work_dir = 'projects/def-wainberg/karbabi/AD-subtyping'
data_dir = 'projects/def-wainberg/single-cell'

################################################################################

# create tensor from pseudobulk with optional gene/sample subsetting
# missing genes are padded with zeros and tracked in padding_mask
def prepare_tensor_dict(pb, genes=None, samples=None):
    if genes is None:
        genes = sorted(set.intersection(
            *[set(var['_index']) for var in pb.iter_var()]))
    if samples is None:
        samples = sorted(set.intersection(
            *[set(obs['ID']) for obs in pb.iter_obs()]))
   
    cell_types = list(pb.keys())    
    tensor = tl.zeros((len(samples), len(genes), len(cell_types)))
    gene_map = {g: i for i, g in enumerate(genes)}    
    padding_mask = np.zeros((len(genes), len(cell_types)), dtype=bool)

    for ct_idx, (ct_name, (X, obs, var)) in enumerate(pb.items()):
        source_idx = [i for i, g in enumerate(var['_index']) if g in gene_map]
        target_idx = [gene_map[var['_index'][i]] for i in source_idx]
        sample_idx = [i for i, s in enumerate(obs['ID']) if s in samples]
        n_padded = len(genes) - len(target_idx)
        if n_padded > 0:
            print(f"{ct_name}: {n_padded} genes padded")
        tensor[:, target_idx, ct_idx] = X[np.ix_(sample_idx, source_idx)]        
        padding_mask[target_idx, ct_idx] = True

    return {'tensor': tensor, 'genes': genes, 'samples': samples,
            'cell_types': cell_types, 'padding_mask': padding_mask}

# only keep specified genes per cell type and pad all others 
def pad_tensor(tensor_dict, genes_to_keep, pad_value=0):
    padding_mask = tensor_dict.get(
        'padding_mask',
        np.ones((len(tensor_dict['genes']), len(tensor_dict['cell_types'])),
        dtype=bool))
    
    for ct_idx, ct in enumerate(tensor_dict['cell_types']):
        if ct in genes_to_keep:
            keep_indices = [
                i for i, g in enumerate(tensor_dict['genes']) 
                if g in genes_to_keep[ct]]
            padding_mask[:, ct_idx] = False
            padding_mask[keep_indices, ct_idx] = True            
            pad_indices = [
                i for i, g in enumerate(tensor_dict['genes'])
                if g not in genes_to_keep[ct]]
            if pad_indices:
                print(f'{ct}: {len(pad_indices)} genes padded')
            tensor_dict['tensor'][:, pad_indices, ct_idx] = pad_value

    tensor_dict['padding_mask'] = padding_mask
    return tensor_dict

# normalize tensor by applying inverse normal transform to each gene
def normalize_tensor_dict(tensor_dict, min_shift=False):
    from utils import inverse_normal_transform
    unfolded = tl.unfold(tensor_dict['tensor'], 0)
    normalized = np.apply_along_axis(inverse_normal_transform, 0, unfolded)
    tensor = tl.fold(normalized, 0, tensor_dict['tensor'].shape)
    if min_shift:
        tensor = tensor + abs(np.min(tensor))
    tensor_dict['tensor'] = tensor
    return tensor_dict

def tucker_ica(tensor, ranks, random_state=None):
    from sklearn.decomposition import FastICA, FactorAnalysis
    # tucker decomposition to get initial factors
    _, factors = tl.decomposition.tucker(
        tensor, rank=ranks, random_state=random_state)
    
    # apply ICA to gene factors and normalize to unit vectors
    ica = FastICA(n_components=ranks[1], random_state=random_state)
    gene_factors = ica.fit_transform(factors[1])
    gene_factors /= np.sqrt(np.sum(gene_factors**2, axis=0))
    
    # compute rotated core tensor
    kron_prod = np.kron(np.eye(ranks[2]), gene_factors)
    kron_prod = kron_prod.reshape(-1, ranks[1] * ranks[2])    
    core_new = factors[0].T @ tl.unfold(tensor, 0) @ kron_prod
    core_new = core_new.reshape(ranks[0], ranks[1], ranks[2])
    
    # apply varimax rotation to sample mode
    fa = FactorAnalysis(
        n_components=ranks[0], rotation='varimax', random_state=random_state)
    fa.fit(core_new.reshape(ranks[0], -1).T)    
    donor_mat = factors[0] @ fa.components_.T
    core_rotated = np.tensordot(fa.components_, core_new, axes=(1,0))

    return core_rotated, [donor_mat, gene_factors, np.eye(ranks[2])]

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=3,
              test_size=0.1):
    from sklearn.model_selection import train_test_split

    rank_samples = trial.suggest_int(
        'rank_samples', *search_space['rank_samples'])
    rank_genes = trial.suggest_int(
        'rank_genes', *search_space['rank_genes'])
    tensor = tensor_dict['tensor']
    ranks = [rank_samples, rank_genes, tensor.shape[2]]

    mses = []
    for rep in range(1, n_reps + 1):
        sample_train, sample_test = train_test_split(
            range(tensor.shape[0]), 
            test_size=test_size, 
            random_state=rep*rank_genes*rank_samples)
        train_tensor = tensor[sample_train]
        test_tensor = tensor[sample_test]
        
        core, factors = tucker_ica(
            train_tensor, ranks, random_state=rep*rank_samples*rank_genes)
        
        loadings = tl.tenalg.multi_mode_dot(
            core, [factors[1], factors[2]], modes=[1, 2])
        loadings_mat = tl.unfold(loadings, mode=0)
        test_unfolded = tl.unfold(test_tensor, mode=0)
        test_coords = test_unfolded @ np.linalg.pinv(loadings_mat)
        recon_tensor = tl.tucker_to_tensor((
            core, [test_coords, factors[1], factors[2]]))
        
        mse = tl.metrics.regression.MSE(test_tensor, recon_tensor)
        print(f'{rep}: {rank_samples=}, {rank_genes=}: {mse=}')
        mses.append(mse)

    mse = np.mean(mses); se = np.std(mses)
    MSE_trial[(rank_samples, rank_genes)] = (mse, se)
    return mse

def plot_rank_grid(MSE_trial, best_params, filename):
    from matplotlib.colors import LogNorm
    ranks_s = sorted(set(k[0] for k in MSE_trial.keys()))
    ranks_g = sorted(set(k[1] for k in MSE_trial.keys()))
    results = np.zeros((len(ranks_s), len(ranks_g)))
    
    for (rs, rg), (mse, se) in MSE_trial.items():
        results[ranks_s.index(rs), ranks_g.index(rg)] = mse

    best_mse, best_se = MSE_trial[(
        best_params['rank_samples'], best_params['rank_genes'])]
    simple_ranks = min([(rs, rg) for (rs, rg), (mse, _) in MSE_trial.items() 
                       if mse <= best_mse + best_se], 
                      key=lambda x: x[0] + x[1])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, 
                cmap='viridis', 
                xticklabels=ranks_g, 
                yticklabels=ranks_s, 
                norm=LogNorm())
    g_idx = ranks_g.index(simple_ranks[1])
    s_idx = ranks_s.index(simple_ranks[0])
    plt.plot(g_idx, s_idx, 'r*', markersize=15)
    plt.text(g_idx + 0.2, s_idx - 0.2, 
             f'({simple_ranks[0]}, {simple_ranks[1]})', 
             color='red')
    plt.xlabel('Gene Rank'); plt.ylabel('Sample Rank')
    plt.title('MSE across rank combinations')
    savefig(filename)

################################################################################

level = 'broad'
coefficient = 'pmAD'

pb_dict = {}; de_dict = {}
for study in ['Green', 'Mathys']:
    pb_dict[study] = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
        .qc(group_column=pl.col(coefficient),
            verbose=False)
    de_dict[study] = pb_dict[study]\
        .DE(formula=f'~ {coefficient} + age_death + sex + pmi + apoe4_dosage',
            coefficient=coefficient,
            group=coefficient,
            verbose=False)
    print_df(de_dict[study].get_num_hits(threshold=0.1).sort('cell_type'))

de_genes = {
    cell_type: (
        de_dict['Green'].table
            .filter(pl.col.cell_type.eq(cell_type))
            .join(de_dict['Mathys'].table.filter(pl.col.cell_type.eq(cell_type)),
                  on=['cell_type', 'gene'], 
                  how='inner')
            .filter((pl.col.FDR.lt(0.10) & pl.col.p_right.lt(0.05)) | 
                    (pl.col.FDR_right.lt(0.10) & pl.col.p.lt(0.05)))
            .filter(pl.col.logFC_right * pl.col.logFC > 0)
            .get_column('gene').to_list())
    for cell_type in de_dict['Green'].table['cell_type'].unique()
}
print(json.dumps({ct: len(genes) for ct, genes in de_genes.items()}, indent=2))


study = 'Green'
pb = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
        .qc(group_column=coefficient, 
            verbose=False)\
        .log_CPM()

# union of de genes 
genes = sorted(set().union(*[
    set(gene_list) for gene_list in de_genes.values()
]))
print(len(genes))

# # union of qc passing genes
# genes = sorted(set.union(*[
#     set(pb[ct].var[ct]['_index']) for ct in pb.keys()
# ]))
# print(len(genes))

tensor_dict = prepare_tensor_dict(pb, genes=genes)
tensor_dict = pad_tensor(tensor_dict, de_genes)
tensor_dict = normalize_tensor_dict(tensor_dict, min_shift=False)

print(tensor_dict['tensor'].shape)
print(np.max(tensor_dict['tensor']), np.min(tensor_dict['tensor']))

MSE_trial = {}
search_space = {'rank_samples': (5, 100, 5), 'rank_genes': (5, 100, 5)}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(
        {k: [i for i in range(*v)] for k,v in search_space.items()},
        seed=42), 
    direction='minimize')
study.optimize(
    lambda trial: objective(
        trial, tensor_dict, search_space, MSE_trial),
    n_trials=len(range(*search_space['rank_samples'])) * 
        len(range(*search_space['rank_genes'])))

plot_rank_grid(
    MSE_trial, study.best_params, 
    f'{work_dir}/figures/rank_selection_de_padded.png')




data = tl.unfold(recon_tensor, mode=0).T
data = data[~np.all(data == 0, axis=1), :]

print(data.shape)
import sys
sys.setrecursionlimit(10000)

plt.figure(figsize=(8, 15))
sns.heatmap(data, 
            cmap='RdBu_r',
            center=0,
            robust=True,
            xticklabels=False,
            yticklabels=False)
savefig(f'{work_dir}/figures/tmp1.png')

plt.figure(figsize=(8, 15))
sns.clustermap(data,
               cmap='RdBu_r', 
               center=0,
               robust=True,
               xticklabels=False,
               yticklabels=False)
savefig(f'{work_dir}/figures/tmp1_clust.png')





