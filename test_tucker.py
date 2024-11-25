import sys
import json
import warnings
import numpy as np
import polars as pl
import pandas as pd
import tensorly as tl
from functools import partial
from joblib import Parallel, delayed
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk, set_num_threads
from utils import debug, print_df, savefig
from ryp import r, to_r

tl.set_backend('numpy') 
set_num_threads(-1)
debug(third_party=True)


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
    from sklearn.decomposition import FastICA
    from scipy.linalg import svd
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # initial tucker decomposition
    _, factors = tl.decomposition.tucker(
        tensor, rank=ranks, init='svd', random_state=random_state)
    donor_mat, gene_mat, ct_mat = factors

    # ica on gene factors 
    ica = FastICA(n_components=ranks[1], 
                  random_state=random_state, 
                  tol=1e-3,
                  max_iter=1000)
    gene_factors = ica.fit_transform(gene_mat)
    norms = np.sqrt(np.sum(gene_factors**2, axis=0))
    gene_factors = gene_factors / norms[None, :]

    # compute unrotated loadings 
    kron_prod = np.kron(gene_factors, ct_mat)
    core_unrotated = donor_mat.T @ tl.unfold(tensor, 0) @ kron_prod
    loadings_unrotated = core_unrotated @ kron_prod.T

    # varimax rotation
    def varimax(Phi, gamma=1.0, q=100, tol=1e-15):
        p,k = Phi.shape
        R = np.eye(k)
        d = 0
        for _ in range(q):
            d_old = d
            Lambda = Phi @ R
            u,s,vh = svd(
                Phi.T @ (Lambda**3 - (gamma/p) *
                         Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
            R = u @ vh
            d = np.sum(s)
            if d_old!=0 and d/d_old < 1 + tol: break
        return Phi @ R, R

    core_reshaped = core_unrotated.reshape(ranks[0], -1)
    core_rotated, rot_mat = varimax(core_reshaped.T) 
    core_rotated = core_rotated.T.reshape(ranks[0], ranks[1], ranks[2])
    donor_scores = donor_mat @ np.linalg.inv(rot_mat)

    return core_rotated, [donor_scores, gene_factors, ct_mat], \
        loadings_unrotated, rot_mat

def run_trial(params, tensor_dict, n_folds=5, base_seed=42, verbose=False):
    from sklearn.model_selection import KFold
    rank_samples, rank_genes = params
    try:
        tensor = tensor_dict['tensor']
        ranks = [rank_samples, rank_genes, tensor.shape[2]]

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
        
        mses = []
        for fold, (train_idx, test_idx) in enumerate(
            kf.split(range(tensor.shape[0]))):

            train_tensor = tensor[train_idx]
            test_tensor = tensor[test_idx]
            core_rotated, factors, loadings_unrotated, rot_mat = \
                tucker_ica(train_tensor, ranks, random_state=base_seed+fold)
            
            test_unfolded = tl.unfold(test_tensor, 0)
            test_coords = test_unfolded @ np.linalg.pinv(loadings_unrotated)
            test_coords = test_coords @ rot_mat
            
            recon_tensor = tl.tucker_to_tensor((
                core_rotated, 
                [test_coords, factors[1], factors[2]]))
            mse = tl.metrics.regression.MSE(test_tensor, recon_tensor)
            if verbose:
                print(f'Fold {fold+1}: {rank_samples=}, {rank_genes=}: {mse=}')
            mses.append(mse)

        mse = np.mean(mses)
        se = np.std(mses)
        return (rank_samples, rank_genes), (mse, se)
    except: 
        print(f'Failed: {rank_samples=}, {rank_genes=}')
        return (rank_samples, rank_genes), (float('inf'), 0)

def find_simple_ranks(results, tolerance_se=1.0):

    MSE_trial = dict(results) if not isinstance(results, dict) else results    

    best_params_tuple = min(MSE_trial.items(), key=lambda x: x[1][0])
    best_mse, best_se = best_params_tuple[1]
    best_params = {
        'rank_samples': best_params_tuple[0][0],
        'rank_genes': best_params_tuple[0][1]
    }    
    simple_ranks = min(
        [(rs, rg) for (rs, rg), (mse, _) in MSE_trial.items() 
         if mse <= best_mse + tolerance_se * best_se],
        key=lambda x: x[0] + x[1]  # minimize sum of ranks
    )
    return {
        'best_params': best_params,
        'best_mse': best_mse,
        'best_se': best_se,
        'simple_ranks': simple_ranks
    }

def plot_rank_grid(MSE_trial, simple_ranks, filename, log_norm=True):

    from matplotlib.colors import LogNorm
    ranks_s = sorted(set(k[0] for k in MSE_trial.keys()))
    ranks_g = sorted(set(k[1] for k in MSE_trial.keys()))
    results = np.zeros((len(ranks_s), len(ranks_g)))
    
    for (rs, rg), (mse, _) in MSE_trial.items():
        results[ranks_s.index(rs), ranks_g.index(rg)] = mse
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, 
                cmap='viridis', 
                xticklabels=ranks_g, 
                yticklabels=ranks_s, 
                norm=LogNorm() if log_norm else None)
    
    # Plot selected ranks
    g_idx = ranks_g.index(simple_ranks[1])
    s_idx = ranks_s.index(simple_ranks[0])
    plt.plot(g_idx, s_idx, 'r*', markersize=15)
    plt.text(g_idx + 0.2, s_idx - 0.2, 
             f'({simple_ranks[0]}, {simple_ranks[1]})', 
             color='red')
    plt.xlabel('Gene Rank')
    plt.ylabel('Sample Rank')
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

search_space = {'rank_samples': (10, 400, 10), 
                'rank_genes': (50, 1500, 50)}
rank_samples = range(*search_space['rank_samples'])
rank_genes = range(*search_space['rank_genes'])
param_combinations = [(rs, rg) for rs in rank_samples for rg in rank_genes]

run_trial_partial = partial(
    run_trial, 
    tensor_dict=tensor_dict,
    n_folds=5,
    base_seed=42)

print(f'Starting grid search with {len(param_combinations)} combinations...')
results = Parallel(n_jobs=52)(
    delayed(run_trial_partial)(params) 
    for params in tqdm.tqdm(param_combinations, desc='Grid search'))

rank_analysis = find_simple_ranks(results)
plot_rank_grid(
    dict(results),
    rank_analysis['simple_ranks'],
    f'{work_dir}/figures/rank_selection_pad_mean_small_project_k5.png',
    log_norm=True)







data = tl.unfold(tensor, mode=2).T
data = factors[0]

print(data.shape)

plt.figure(figsize=(8, 15))
sns.heatmap(data, 
            cmap='RdBu_r',
            center=0,
            robust=True,
            xticklabels=False,
            yticklabels=False)
savefig(f'{work_dir}/figures/tmp2.png')

import sys
sys.setrecursionlimit(10000)
plt.figure(figsize=(8, 15))
sns.clustermap(data,
               cmap='RdBu_r', 
               center=0,
               robust=True,
               xticklabels=False,
               yticklabels=False)
savefig(f'{work_dir}/figures/tmp8_clust.png')
