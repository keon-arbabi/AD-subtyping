import numpy as np
import tensorly as tl
import polars as pl

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

def zero_pad_tensor(tensor_dict):
    mask_3d = np.broadcast_to(
        tensor_dict['padding_mask'], tensor_dict['tensor'].shape)
    tensor_dict['tensor'] *= mask_3d
    return tensor_dict

def speckle_tensor(tensor_dict, speckle_fraction=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    tensor = tensor_dict['tensor']
    if 'padding_mask' in tensor_dict:
        mask_3d = np.broadcast_to(
            tensor_dict['padding_mask'], tensor.shape)
        non_pad_idx = np.where(mask_3d)
    else:
        non_pad_idx = np.where(np.ones_like(tensor, dtype=bool))
    
    n_mask = int(speckle_fraction * len(non_pad_idx[0]))
    mask_indices = rng.choice(len(non_pad_idx[0]), n_mask, replace=False)
    
    speckle_mask = np.zeros_like(tensor, dtype=bool)
    speckle_mask[tuple(idx[mask_indices] for idx in non_pad_idx)] = True
    
    masked_tensor = tensor.copy()
    masked_tensor[speckle_mask] = 0
    return masked_tensor, speckle_mask

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=3):
    # from tensorly.contrib.sparse.decomposition import tucker

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
            tol=1e-3,
            n_iter_max=np.iinfo(np.int32).max,
            random_state=rep*rank_genes*rank_samples,
            verbose=2)
        reconstructed = tl.tucker_to_tensor((core, factors))
        mses.append(tl.metrics.regression.MSE(
            tensor_dict['tensor'][mask], reconstructed[mask]))
    
    mse = np.mean(mses)
    MSE_trial[(rank_samples, rank_genes)] = mse
    return mse

level = 'broad'
coefficient = 'pmAD'

de_dict = {}
for study in ['Green', 'Mathys']:
    de_dict[study] = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
        .qc(group_column=pl.col(coefficient),
            verbose=False)\
        .DE(formula=f'~ {coefficient} + age_death + sex + pmi + apoe4_dosage',
            coefficient=coefficient,
            group=coefficient)
    print_df(de_dict[study].get_num_hits(threshold=0.1)
             .sort('cell_type'))

de_genes = {
    cell_type: (
        de_dict['Green'].table
            .filter(pl.col.cell_type.eq(cell_type))
            .join(de_dict['Mathys'].table.filter(pl.col.cell_type.eq(cell_type)),
                  on=['cell_type', 'gene'], 
                  how='inner')
            .filter((pl.col.FDR.lt(0.10) & pl.col.p_right.lt(0.05)) | 
                    (pl.col.FDR_right.lt(0.10) & pl.col.p.lt(0.05)))
            .filter(pl.col.logFC_right * pl.col.logFC > 0)['gene'])
    for cell_type in de_dict['Green'].table['cell_type'].unique()
}
print(json.dumps({ct: len(genes) for ct, genes in de_genes.items()}, indent=2))

study = 'Green'
pb = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
        .qc(group_column=None, verbose=False)\
        .log_CPM()



def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=5, 
              test_size=0.1):
    from sklearn.model_selection import train_test_split
    rank_samples = trial.suggest_int(
        'rank_samples', *search_space['rank_samples'])
    rank_genes = trial.suggest_int(
        'rank_genes', *search_space['rank_genes'])
    tensor = tensor_dict['tensor']
    
    mses = []
    for rep in range(1, n_reps + 1):
        sample_train, sample_test = train_test_split(
            range(tensor.shape[0]), test_size=test_size,
            random_state=rep*rank_genes*rank_samples)
        gene_train, gene_test = train_test_split(
            range(tensor.shape[1]), test_size=test_size,
            random_state=rep*rank_genes*rank_samples)
        
        # Train
        train_tensor = tensor[
            np.ix_(sample_train, gene_train, range(tensor.shape[2]))]
        core, factors = tl.decomposition.tucker(
            train_tensor,
            rank=[rank_samples, rank_genes, tensor.shape[2]],
            init='random', tol=1e-5,
            n_iter_max=np.iinfo(np.int32).max,
            random_state=rep*rank_genes*rank_samples,
            verbose=0)
        
        # Project test samples
        test_tensor = tensor[
            np.ix_(sample_test, gene_train, range(tensor.shape[2]))]
        test_unfolded = tl.unfold(test_tensor, 0)
        kron = tl.tenalg.kronecker([factors[2], factors[1]])
        test_coords = test_unfolded @ kron @ np.linalg.pinv(tl.unfold(core, 0))
        reconstructed = tl.tucker_to_tensor(
            (core, [test_coords, factors[1], factors[2]]))
        mse = tl.metrics.regression.MSE(test_tensor, reconstructed)
        print(f'{rep}: {rank_samples=}, {rank_genes=}: {mse=}')
        mses.append(mse)

        # # Project test genes
        # test_tensor = tensor[
        #     np.ix_(sample_train, gene_test, range(tensor.shape[2]))]
        # test_unfolded = tl.unfold(test_tensor, 1)
        # kron = tl.tenalg.kronecker([factors[2], factors[0]])
        # test_coords = test_unfolded @ kron @ np.linalg.pinv(tl.unfold(core, 1))
        # reconstructed = tl.tucker_to_tensor(
        #    (core, [factors[0], test_coords, factors[2]]))
        # mse = tl.metrics.regression.MSE(test_tensor, reconstructed)
        # mses.append(mse)
    
    mse = np.mean(mses)
    MSE_trial[(rank_samples, rank_genes)] = mse
    return mse

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=5, 
              test_size=0.1):
    from sklearn.model_selection import train_test_split
    rank_samples = trial.suggest_int(
        'rank_samples', *search_space['rank_samples'])
    rank_genes = trial.suggest_int(
        'rank_genes', *search_space['rank_genes'])
    tensor = tensor_dict['tensor']
    
    rank_samples = 2
    rank_genes = 2

    mses = []
    for rep in range(1, n_reps + 1):
            sample_train, sample_test = train_test_split(
                range(tensor.shape[0]), test_size=test_size,
                random_state=rep*rank_genes*rank_samples)
            
            train_tensor = tensor[sample_train]
            core, factors = tl.decomposition.tucker(
                train_tensor,
                rank=[rank_samples, rank_genes, tensor.shape[2]],
                init='random', tol=1e-5,
                n_iter_max=np.iinfo(np.int32).max,
                random_state=rep*rank_genes*rank_samples,
                verbose=0)
            
            test_tensor = tensor[sample_test]
            test_unfolded = tl.unfold(test_tensor, 0)
            kron = tl.tenalg.kronecker([factors[2], factors[1]])
            test_coords = test_unfolded @ kron @ \
                np.linalg.pinv(tl.unfold(core, 0))
            reconstructed = tl.tucker_to_tensor(
                (core, [test_coords, factors[1], factors[2]]))
        
            mse = tl.metrics.regression.MSE(test_tensor, reconstructed)
            print(f'{rep}: {rank_samples=}, {rank_genes=}: {mse=}')
            mses.append(mse)

    mse = np.mean(mses)
    MSE_trial[(rank_samples, rank_genes)] = mse
    return mse


samples = sorted(set.intersection(
    *[set(obs['ID']) for obs in pb.iter_obs()]))

genes = sorted(set.union(*[
   set.intersection(*[
       set(pb_dict[study][ct].var[ct]['_index']) 
       for study in pb_dict]) 
   for ct in pb_dict['Green'].keys()
]))

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=5, 
              test_size=0.1):
    from sklearn.model_selection import train_test_split

    rank_samples = trial.suggest_int(
        'rank_samples', *search_space['rank_samples'])
    rank_genes = trial.suggest_int(
        'rank_genes', *search_space['rank_genes'])
    tensor = tensor_dict['tensor']

    mses = []
    for rep in range(1, n_reps + 1):
        sample_train, sample_test = train_test_split(
            range(tensor.shape[0]), test_size=test_size,
            random_state=rep * rank_genes * rank_samples)
        train_tensor = tensor[sample_train]
        
        core, factors = tl.decomposition.tucker(
            train_tensor, 
            rank=[rank_samples, rank_genes, tensor.shape[2]], 
            init='random', tol=1e-5,
            n_iter_max=np.iinfo(np.int32).max,
            random_state=rep * rank_genes * rank_samples,
            verbose=0)

        loadings = tl.tenalg.multi_mode_dot(
            core, [factors[1], factors[2]], modes=[1, 2])
        loadings_matrix = tl.unfold(loadings, mode=0)

        test_tensor = tensor[sample_test]
        test_unfolded = tl.unfold(test_tensor, mode=0)
        test_coords = test_unfolded @ np.linalg.pinv(loadings_matrix)
        reconstructed = tl.tucker_to_tensor((
            core, [test_coords, factors[1], factors[2]]))

        mse = tl.metrics.regression.MSE(test_tensor, reconstructed)
        print(f'{rep}: {rank_samples=}, {rank_genes=}: {mse=}')
        mses.append(mse)

    mse = np.mean(mses)
    if MSE_trial is not None:
        MSE_trial[(rank_samples, rank_genes)] = mse

    return mse


# randomly mask a fraction of non-padded values in tensor
def speckle_tensor(tensor_dict, speckle_fraction=0.05, random_state=None):
    rng = np.random.default_rng(random_state)
    tensor = tensor_dict['tensor']
    if 'padding_mask' in tensor_dict:
        mask_3d = np.broadcast_to(tensor_dict['padding_mask'], tensor.shape)
        non_pad_idx = np.where(mask_3d)
    else:
        non_pad_idx = np.where(np.ones_like(tensor, dtype=bool))
    n_mask = int(speckle_fraction * len(non_pad_idx[0]))
    mask_indices = rng.choice(len(non_pad_idx[0]), n_mask, replace=False)
    speckle_mask = np.zeros_like(tensor, dtype=bool)
    speckle_mask[tuple(idx[mask_indices] for idx in non_pad_idx)] = True
    masked_tensor = tensor.copy()
    masked_tensor[speckle_mask] = 0

    return masked_tensor, speckle_mask

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=3):
    rank_samples = trial.suggest_int(
        'rank_samples', *search_space['rank_samples'])
    rank_genes = trial.suggest_int(
        'rank_genes', *search_space['rank_genes'])
    ranks = [rank_samples, rank_genes, len(tensor_dict['cell_types'])]
    
    mses = []
    for rep in range(n_reps):
        random_state = rep*rank_genes*rank_samples
        masked_tensor, mask = speckle_tensor(
            tensor_dict, random_state=random_state)
        core, factors = tl.decomposition.tucker(
            masked_tensor,
            ranks,
            tol=1e-5,
            n_iter_max=np.iinfo(np.int32).max,
            init='svd',
            svd='truncated_svd',
            random_state=random_state,
            verbose=0)
        reconstructed = tl.tucker_to_tensor((core, factors))
        mse = tl.metrics.regression.MSE(
            tensor_dict['tensor'][mask], reconstructed[mask])
        print(f'{rep}: {rank_samples=}, {rank_genes=}: {mse=}')
        mses.append(mse)
    
    mse = np.mean(mses); se = np.std(mses)
    MSE_trial[(rank_samples, rank_genes)] = (mse, se)
    return mse





# # randomly mask a fraction of non-padded values in tensor
# def speckle_tensor(tensor_dict, speckle_fraction=0.05, random_state=None):
#     rng = np.random.default_rng(random_state)
#     tensor = tensor_dict['tensor']
#     if 'padding_mask' in tensor_dict:
#         mask_3d = np.broadcast_to(tensor_dict['padding_mask'], tensor.shape)
#         non_pad_idx = np.where(mask_3d)
#     else:
#         non_pad_idx = np.where(np.ones_like(tensor, dtype=bool))
#     n_mask = int(speckle_fraction * len(non_pad_idx[0]))
#     mask_indices = rng.choice(len(non_pad_idx[0]), n_mask, replace=False)
#     speckle_mask = np.zeros_like(tensor, dtype=bool)
#     speckle_mask[tuple(idx[mask_indices] for idx in non_pad_idx)] = True
#     masked_tensor = tensor.copy()
#     masked_tensor[speckle_mask] = 0

#     return masked_tensor, speckle_mask

# # objective, learn speckled data using Tucker
# def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=3):
#     rank_samples = trial.suggest_int(
#         'rank_samples', *search_space['rank_samples'])
#     rank_genes = trial.suggest_int(
#         'rank_genes', *search_space['rank_genes'])
#     ranks = [rank_samples, rank_genes, len(tensor_dict['cell_types'])]
    
#     mses = []
#     for rep in range(n_reps):
#         random_state = rep*rank_genes*rank_samples
#         masked_tensor, mask = speckle_tensor(
#             tensor_dict, random_state=random_state)
#         core, factors = tl.decomposition.tucker(
#             masked_tensor,
#             ranks,
#             tol=1e-5,
#             n_iter_max=np.iinfo(np.int32).max,
#             init='svd',
#             svd='truncated_svd',
#             random_state=random_state,
#             verbose=0)
#         reconstructed = tl.tucker_to_tensor((core, factors))
#         mse = tl.metrics.regression.MSE(
#             tensor_dict['tensor'][mask], reconstructed[mask])
#         print(f'{rep}: {rank_samples=}, {rank_genes=}: {mse=}')
#         mses.append(mse)
    
#     mse = np.mean(mses); se = np.std(mses)
#     MSE_trial[(rank_samples, rank_genes)] = (mse, se)
#     return mse

# objective, learn test samples using Tucker and projection
def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=3, 
              test_size=0.1):
    from sklearn.model_selection import train_test_split

    rank_samples = trial.suggest_int(
        'rank_samples', *search_space['rank_samples'])
    rank_genes = trial.suggest_int(
        'rank_genes', *search_space['rank_genes'])
    tensor = tensor_dict['tensor']

    mses = []
    for rep in range(1, n_reps + 1):
        sample_train, sample_test = train_test_split(
            range(tensor.shape[0]), test_size=test_size,
            random_state=rep * rank_genes * rank_samples)
        train_tensor = tensor[sample_train]
        
        core, factors = tl.decomposition.tucker(
            train_tensor, 
            rank=[rank_samples, rank_genes, tensor.shape[2]],
            init='svd', 
            svd='truncated_svd',
            tol=1e-5,
            n_iter_max=np.iinfo(np.int32).max,
            random_state=rep * rank_genes * rank_samples,
            verbose=0)

        loadings = tl.tenalg.multi_mode_dot(
            core, [factors[1], factors[2]], modes=[1, 2])
        loadings_matrix = tl.unfold(loadings, mode=0)

        test_tensor = tensor[sample_test]
        test_unfolded = tl.unfold(test_tensor, mode=0)
        test_coords = test_unfolded @ np.linalg.pinv(loadings_matrix)
        reconstructed = tl.tucker_to_tensor((
            core, [test_coords, factors[1], factors[2]]))

        mse = tl.metrics.regression.MSE(test_tensor, reconstructed)
        print(f'{rep}: {rank_samples=}, {rank_genes=}: {mse=}')
        mses.append(mse)

    mse = np.mean(mses); se = np.std(mses)
    MSE_trial[(rank_samples, rank_genes)] = (mse, se)
    return mse