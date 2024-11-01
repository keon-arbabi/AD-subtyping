import sys
import os 
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

set_num_threads(-1)
tl.set_backend('numpy') 
debug(third_party=True)

work_dir = 'projects/def-wainberg/karbabi/AD-subtyping'
data_dir = 'projects/def-wainberg/single-cell'

################################################################################

def prepare_tensor_dict(pb, de_genes):
    genes = sorted(
        set.union(*de_genes.values())
        .intersection(*[set(var['_index']) for var in pb.iter_var()]))
    samples = sorted(set.intersection(
        *[set(obs['ID']) for obs in pb.iter_obs()]))
    cell_types = list(pb.keys())
    
    padding_mask = np.zeros((len(genes), len(cell_types)), dtype=bool)
    for ct_idx, ct in enumerate(cell_types):
        padding_mask[:, ct_idx] = [g in de_genes[ct] for g in genes]
    
    tensor = tl.zeros((len(samples), len(genes), len(cell_types)))
    for ct_idx, (_, (X, obs, var)) in enumerate(pb.items()):
        gene_idx = [i for i, g in enumerate(var['_index']) if g in genes]
        sample_idx = [i for i, s in enumerate(obs['ID']) if s in samples]
        tensor[:, :, ct_idx] = X[np.ix_(sample_idx, gene_idx)]
    
    return {'tensor': tensor, 'genes': genes, 'samples': samples,
            'cell_types': cell_types, 'padding_mask': padding_mask}

def normalize_tensor_dict(tensor_dict):
    from utils import inverse_normal_transform
    unfolded = tl.unfold(tensor_dict['tensor'], 0)
    normalized = np.apply_along_axis(inverse_normal_transform, 0, unfolded)
    tensor = tl.fold(normalized, 0, tensor_dict['tensor'].shape)
    tensor_dict['tensor'] = tensor + abs(np.min(tensor))
    return tensor_dict

def zero_pad_tensor(tensor_dict):
    mask_3d = np.broadcast_to(
        tensor_dict['padding_mask'], tensor_dict['tensor'].shape)
    tensor_dict['tensor'] *= mask_3d
    return tensor_dict

def speckle_tensor(tensor_dict, speckle_fraction=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    
    if 'padding_mask' in tensor_dict:
        mask_3d = np.broadcast_to(
            tensor_dict['padding_mask'], tensor_dict['tensor'].shape)
        non_pad_idx = np.where(mask_3d)
    else:
        non_pad_idx = np.where(np.ones_like(tensor_dict['tensor'], dtype=bool))
    
    n_mask = int(speckle_fraction * len(non_pad_idx[0]))
    mask_indices = rng.choice(len(non_pad_idx[0]), n_mask, replace=False)
    
    speckle_mask = np.zeros_like(tensor_dict['tensor'], dtype=bool)
    speckle_mask[tuple(idx[mask_indices] for idx in non_pad_idx)] = True
    
    masked_tensor = tensor_dict['tensor'].copy()
    masked_tensor[speckle_mask] = 0
    return masked_tensor, speckle_mask

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=3):
    rank_samples = trial.suggest_int(
        'rank_samples', *search_space['rank_samples'])
    rank_genes = trial.suggest_int(
        'rank_genes', *search_space['rank_genes'])

    mses = []
    for rep in range(n_reps):
        masked_tensor, mask = speckle_tensor(
            tensor_dict, 
            random_state=rep*rank_genes*rank_samples)
        ranks = [rank_samples, rank_genes, len(tensor_dict['cell_types'])]
        core, factors = tl.decomposition.tucker(
            masked_tensor, 
            ranks, 
            init='random',
            tol=1e-5,
            n_iter_max=np.iinfo(np.int32).max,
            random_state=rep*rank_genes*rank_samples,
            verbose=2)
        reconstructed = tl.tucker_tensor.tucker_to_tensor((core, factors))
        mses.append(np.mean((tensor_dict['tensor'][mask] - 
                           reconstructed[mask]) ** 2))
    
    mse = np.mean(mses)
    MSE_trial[(rank_samples, rank_genes)] = mse
    return mse

def plot_rank_grid(MSE_trial, best_params, filename):
    from matplotlib.colors import LogNorm
    ranks_s = sorted(set(k[0] for k in MSE_trial.keys()))
    ranks_g = sorted(set(k[1] for k in MSE_trial.keys()))
    results = np.zeros((len(ranks_s), len(ranks_g)))
    
    for (rs, rg), mse in MSE_trial.items():
        i = ranks_s.index(rs)
        j = ranks_g.index(rg)
        results[i, j] = mse
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, cmap='viridis', 
                xticklabels=ranks_g, yticklabels=ranks_s,
                norm=LogNorm())
    plt.plot(ranks_g.index(best_params['rank_genes']), 
            ranks_s.index(best_params['rank_samples']), 'r*', markersize=15)
    plt.xlabel('Gene Rank')
    plt.ylabel('Sample Rank')
    plt.title('MSE across rank combinations')
    savefig(filename)

################################################################################

level = 'broad'
coefficient = 'pmAD'

de_dict = {}
for study in ['Green', 'Mathys']:
    de_dict[study] = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
        .qc(group_column=coefficient,
            verbose=False)\
        .DE_old(label_column=coefficient, 
            covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
            case_control=True,
            verbose=False)
    print_df(de_dict[study].get_num_hits(threshold=0.1)
             .sort('cell_type'))

de_genes = {}
for cell_type in de_dict['Green'].table['cell_type'].unique():
    de_genes[cell_type] = set().union(*[
        de_dict[study1].table\
            .filter(pl.col.cell_type.eq(cell_type) & pl.col.FDR.lt(0.10))\
            .join(de_dict[study2].table\
                  .filter(pl.col.cell_type.eq(cell_type) & pl.col.P.lt(0.05)),
                  on=['cell_type', 'gene'],
                  how='inner')\
            .filter(pl.col.logFC_right * pl.col.logFC > 0)['gene']
        for study1, study2 in [('Green', 'Mathys'), ('Mathys', 'Green')]
    ])
print(json.dumps({ct: len(genes) for ct, genes in de_genes.items()}, indent=2))

study = 'Green'
pb = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
        .qc(group_column='pmAD', verbose=False)\
        .log_CPM()

tensor_dict = prepare_tensor_dict(pb, de_genes)
tensor_dict = normalize_tensor_dict(tensor_dict)
tensor_dict = zero_pad_tensor(tensor_dict)
tensor_dict['tensor'].shape

MSE_trial = {}
search_space = {'rank_samples': (2, 20, 2), 'rank_genes': (2, 50, 2)}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(
        {k: [i for i in range(*v)] for k,v in search_space.items()}), 
    direction='minimize')

study.optimize(
    lambda trial: objective(
        trial, tensor_dict, search_space, MSE_trial),
    n_trials=len(range(*search_space['rank_samples'])) * 
        len(range(*search_space['rank_genes'])))

print(study.best_params)
plot_rank_grid(MSE_trial, study.best_params, 
               f'{work_dir}/figures/rank_selection.png')







unfolded = tl.unfold(tensor_dict['tensor'], mode=0).T   
print(unfolded.shape)

plt.figure(figsize=(8, 15))
sns.heatmap(unfolded, 
            cmap='RdBu_r',
            center=0,
            robust=True,
            xticklabels=False,
            yticklabels=False)
plt.xlabel('Donor x Cell Type')
plt.ylabel('Genes')
savefig(f'{work_dir}/figures/tensor_heatmap_samples.png')


unfolded = tl.unfold(reconstructed, mode=0).T   
print(unfolded.shape)

plt.figure(figsize=(8, 15))
sns.heatmap(unfolded, 
            cmap='RdBu_r',
            center=0,
            robust=True,
            xticklabels=False,
            yticklabels=False)
plt.xlabel('Donor x Cell Type')
plt.ylabel('Genes')
savefig(f'{work_dir}/figures/tensor_heatmap_samples_recon.png')













def prepare_tensor_dict(pb, genes, samples):
    cell_types = list(pb.keys())
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    sample_to_idx = {s: i for i, s in enumerate(samples)}
    
    tensor = tl.zeros((len(samples), len(genes), len(cell_types)))
    
    for ct_idx, (_, (X, obs, var)) in enumerate(pb.items()):
        gene_indices = [
            gene_to_idx[g] for g in var['_index'] if g in gene_to_idx]
        sample_indices = [
            sample_to_idx[s] for s in obs['ID'] if s in sample_to_idx]
        
        for i, s_idx in enumerate(sample_indices):
            for j, g_idx in enumerate(gene_indices):
                tensor[s_idx, g_idx, ct_idx] = X[i, j]

    return {'tensor': tensor, 'genes': genes, 'samples': samples, 
            'cell_types': cell_types}

study = 'Green'
pb = pb_dict[study]

overlapping_genes = sorted(
    set.union(*de_genes.values())
    .intersection(*[set(var['_index']) for var in pb.iter_var()]))
print(len(overlapping_genes))

overlapping_samples = sorted(
    set.intersection(*(
        set(obs['ID']) for obs in pb.iter_obs())))
print(len(overlapping_samples))

genes = overlapping_genes
samples = overlapping_samples

tensor_dict = prepare_tensor_dict(pb, genes, samples)
len(tensor_dict['samples'])
len(tensor_dict['genes'])








from single_cell import Pseudobulk

pb = Pseudobulk('projects/def-wainberg/single-cell/Green/pseudobulk/broad')\
    .qc(group_column=None,
        verbose=False)

de = pb\
    .DE(formula='~ pmAD + age_death + sex + pmi + apoe4_dosage',
        coefficient='pmAD1',
        categorical_columns='pmAD',
        group='pmAD',
        verbose=True)


