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

# objective, learn speckled data using Tucker
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
              test_size=0.25):
    from sklearn.model_selection import train_test_split
    try:
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
    except:
        print(f'Failed: {rank_samples=}, {rank_genes=}')
        return float('inf')
    




def get_reconstruct_errors_svd(tensor, max_ranks_test, shuffle_tensor=False):
    if shuffle_tensor:
        unfolded = tl.unfold(tensor, 0)
        shuffled = unfolded.copy()
        for i in range(unfolded.shape[1]):
            shuffled[:,i] = np.random.permutation(unfolded[:,i])
        tensor = tl.fold(shuffled, 0, tensor.shape)
    
    mode_rank_errors = []
    modes = [0, 1]
    for m, mode in enumerate(modes, 1):
        unfolded = tl.unfold(tensor, mode)
        u, s, vh = np.linalg.svd(unfolded, full_matrices=False)
        
        rank_errors = []
        for rank in range(1, max_ranks_test[m-1] + 1):
            recon = u[:,:rank] @ np.diag(s[:rank]) @ vh[:rank,:]
            error = np.linalg.norm(recon - unfolded, 'fro')**2 / \
                   np.linalg.norm(unfolded, 'fro')**2
            rank_errors.append(error)
        mode_rank_errors.append(rank_errors)
    
    return mode_rank_errors

def plot_rec_errors_line_svd(real, shuffled, mode):
    df = pd.DataFrame(columns=['rec_error', 'num_ranks', 'num_iter', 'run_type'])
    
    for i, shuffle_iter in enumerate(shuffled, 1):
        ranks = range(1, len(shuffle_iter[mode-1]) + 1)
        df_iter = pd.DataFrame({
            'rec_error': shuffle_iter[mode-1],
            'num_ranks': ranks,
            'num_iter': str(i),
            'run_type': 'shuffled'
        })
        df = pd.concat([df, df_iter])
    
    df_real = pd.DataFrame({
        'rec_error': real[mode-1],
        'num_ranks': range(1, len(real[mode-1]) + 1),
        'num_iter': 'real',
        'run_type': 'real'
    })
    df = pd.concat([df, df_real])
    
    by_interval = 2 if mode == 1 else 5
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df[df.run_type == 'shuffled'], 
                x='num_ranks', y='rec_error', color='gray', alpha=0.5)
    sns.lineplot(data=df[df.run_type == 'real'],
                x='num_ranks', y='rec_error', color='red')
    plt.xticks(range(0, max(df.num_ranks) + 1, by_interval))
    plt.yscale('log')
    plt.title(f"{'Sample' if mode == 1 else 'Gene'} Mode")
    return plt.gcf()

def plot_rec_errors_bar_svd(real, shuffled, mode):
    df_null = pd.DataFrame([s[mode-1] for s in shuffled]).T
    mean_null = df_null.mean(axis=1)
    std_null = df_null.std(axis=1)
    ranks = range(1, len(mean_null) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(ranks, mean_null, yerr=std_null, color='gray', 
                label='Shuffled (mean ± std)')
    plt.plot(ranks, real[mode-1], color='red', label='Real')
    plt.yscale('log')
    plt.xticks(ranks[::2 if mode == 1 else 5])
    plt.legend()
    plt.title(f"{'Sample' if mode == 1 else 'Gene'} Mode")
    return plt.gcf()

def determine_ranks_tucker(tensor_dict, max_ranks_test, num_iter=100, 
                           random_state=42):
    np.random.seed(random_state)
    tensor = tensor_dict['tensor']
    
    null_res = []
    for _ in range(num_iter):
        null_errors = get_reconstruct_errors_svd(
            tensor, max_ranks_test, shuffle_tensor=True)
        null_res.append(null_errors)
    
    real_errors = get_reconstruct_errors_svd(
        tensor, max_ranks_test, shuffle_tensor=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for i, mode in enumerate([1, 2], 0):
        line_plot = plot_rec_errors_line_svd(real_errors, null_res, mode)
        bar_plot = plot_rec_errors_bar_svd(real_errors, null_res, mode)
        axes[i,0].clear()
        axes[i,1].clear()
        line_plot.axes[0].figure = fig
        bar_plot.axes[0].figure = fig
        axes[i,0] = line_plot.axes[0]
        axes[i,1] = bar_plot.axes[0]
        plt.close(line_plot)
        plt.close(bar_plot)
    
    plt.tight_layout()
    return {'real': real_errors, 'null': null_res, 'plot': fig}





def tucker_ica(tensor, ranks, random_state=None):
    from sklearn.decomposition import FastICA
    from scipy.linalg import svd

    _, factors = tl.decomposition.tucker(
        tensor, rank=ranks, init='svd', random_state=random_state)
    donor_mat, gene_mat, _ = factors

    ica = FastICA(n_components=ranks[1], random_state=random_state)
    gene_factors = ica.fit_transform(gene_mat)
    norms = np.sqrt(np.sum(gene_factors**2, axis=0))
    gene_factors = gene_factors / norms[None, :]

    ctype_factors = np.eye(ranks[2])
    kron_prod = np.kron(ctype_factors, gene_factors)

    tensor_unfolded = tl.unfold(tensor, 0)
    core_new = donor_mat.T @ tensor_unfolded @ kron_prod
    core_reshaped = core_new.reshape(ranks[0], -1)

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

    core_rotated, rot_mat = varimax(core_reshaped.T)
    core_rotated = core_rotated.T
    donor_scores = donor_mat @ np.linalg.inv(rot_mat)
    core_rotated = core_rotated.reshape(ranks[0], ranks[1], ranks[2])
    return core_rotated, [donor_scores, gene_factors, ctype_factors]

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_reps=3,
              test_size=0.25):
    from sklearn.model_selection import train_test_split
    try:
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
    except:
        print(f'Failed: {rank_samples=}, {rank_genes=}')
        return float('inf')
    



def tucker_ica(tensor, ranks, random_state=None):
    from sklearn.decomposition import FastICA
    from scipy.linalg import svd

    # initial tucker decomposition
    _, factors = tl.decomposition.tucker(
        tensor, rank=ranks, init='svd', random_state=random_state)
    donor_mat, gene_mat, ct_mat = factors

    # ica on gene factors 
    ica = FastICA(n_components=ranks[1], random_state=random_state)
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

def objective(trial, tensor_dict, search_space, MSE_trial=None, n_folds=5, 
              base_seed=42):
    from sklearn.model_selection import KFold
    try:
        rank_samples = trial.suggest_int(
            'rank_samples', *search_space['rank_samples'])
        rank_genes = trial.suggest_int(
            'rank_genes', *search_space['rank_genes'])
        tensor = tensor_dict['tensor']
        ranks = [rank_samples, rank_genes, tensor.shape[2]]

        # Use KFold with fixed random state
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
        
        mses = []
        for fold, (train_idx, test_idx) in enumerate(
            kf.split(range(tensor.shape[0]))):

            train_tensor = tensor[train_idx]
            test_tensor = tensor[test_idx]
            core_rotated, factors, loadings_unrotated, rot_mat = \
                tucker_ica(train_tensor, ranks, random_state=base_seed+fold)
            
            # project using unrotated loadings
            test_unfolded = tl.unfold(test_tensor, 0)
            test_coords = test_unfolded @ np.linalg.pinv(loadings_unrotated)
            # apply rotation to projected coordinates
            test_coords = test_coords @ rot_mat
            
            # reconstruct and compare
            recon_tensor = tl.tucker_to_tensor((
                core_rotated, 
                [test_coords, factors[1], factors[2]]))
            mse = tl.metrics.regression.MSE(test_tensor, recon_tensor)

            print(f'Fold {fold+1}: {rank_samples=}, {rank_genes=}: {mse=}')
            mses.append(mse)

        mse = np.mean(mses); se = np.std(mses)
        MSE_trial[(rank_samples, rank_genes)] = (mse, se)
        return mse
    except: 
        print(f'Failed: {rank_samples=}, {rank_genes=}')
        return float('inf')
    


MSE_trial = {}
search_space = {'rank_samples': (2, 50, 2), 'rank_genes': (2, 50, 2)}
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



# Calculate gene-factor associations using R's lm function
def get_lm_pvals(tensor_dict):
    from utils import fdr
    
    tensor = tensor_dict['tensor']
    scores = tensor_dict['decomposition']['factors']['donors']
    genes = tensor_dict['dims']['genes']
    cell_types = tensor_dict['dims']['cell_types']
    to_r(scores, 'scores')
    
    results = []
    r('n_factors <- ncol(scores)')   

    for j, ct in enumerate(cell_types):
        for i, gene in enumerate(genes):
            fiber = tensor[:, i, j]
            if np.all(fiber == 0): continue
                
            to_r(fiber, 'fiber')
            r('''
            stats <- lapply(1:n_factors, function(k) {
                fit <- lm(scores[,k] ~ fiber)
                summ <- summary(fit)
                f_stat <- summ$fstatistic
                list(
                    pval = pf(
                        f_stat[1], f_stat[2], f_stat[3], 
                        lower.tail=FALSE),
                    r2 = summ$r.squared,
                    beta = coef(fit)[2],
                    se = summ$coefficients[2,2]
                )
            })
            ''')
            for k in range(scores.shape[1]):
                stats = to_py(f'stats[[{k+1}]]')
                if not np.isnan(stats['pval']):
                    results.append({
                        'gene': gene, 'cell_type': ct, 'factor': k,
                        'pvalue': stats['pval'], 'r2': stats['r2'],
                        'beta': stats['beta'], 'beta_se': stats['se']
                    })
        
    tensor_dict['gene_associations'] = pl.DataFrame(results)\
        .with_columns(fdr(pl.col('pvalue')).alias('padj'))
    return tensor_dict


level = 'broad'
coefficient = 'pmAD'

pb_dict = {}; de_dict = {}
for study in ['Green', 'Mathys']:
    pb_dict[study] = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
        .qc(group_column=pl.col(coefficient),
            verbose=False)
    de_dict[study] = pb_dict[study]\
        .filter_var(pl.col('_index').is_in(get_coding_genes()['gene']))\
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
            .filter((pl.col.FDR.lt(0.20) & pl.col.p_right.lt(0.05)) | 
                    (pl.col.FDR_right.lt(0.20) & pl.col.p.lt(0.05)))
            .filter(pl.col.logFC_right * pl.col.logFC > 0)
            .get_column('gene').to_list())
    for cell_type in de_dict['Green'].table['cell_type'].unique()
}
print(json.dumps({ct: len(genes) for ct, genes in de_genes.items()}, indent=2))

# # union of qc passing genes
# genes = sorted(set.union(*[
#     set(pb[ct].var[ct]['_index']) for ct in pb.keys()
# ]))
# print(len(genes))



# Plot gene loadings heatmap for each factor across cell type
def plot_gene_heatmap(tensor_dict, filename, n_genes_per_factor=10):
    loading_tensor = tensor_dict['decomposition']['loadings']['tensor']
    n_factors = loading_tensor.shape[0]
    genes = tensor_dict['dims']['genes']
    cell_types = tensor_dict['dims']['cell_types']
    
    # Get top genes based on loading magnitudes for each factor-cell type combo
    top_genes = {}
    for f in range(n_factors):
        factor_loadings = loading_tensor[f]  
        top_genes[str(f)] = []
        
        for ct_idx in range(factor_loadings.shape[1]):
            ct_loadings = np.abs(factor_loadings[:, ct_idx])
            top_indices = np.argsort(ct_loadings)[-n_genes_per_factor:]
            top_genes[str(f)].extend([genes[i] for i in top_indices])

        top_genes[str(f)] = list(dict.fromkeys(top_genes[str(f)]))
    
    r('''
    suppressPackageStartupMessages({
        library(ComplexHeatmap)
        library(circlize)
        library(grid)
        library(ggpubr)
    })
    plot_list <- list()
    ''')
    
    for f in range(n_factors):
        factor_loadings = loading_tensor[f]
        to_r(factor_loadings, 'loadings_mat', 
             rownames=genes, colnames=cell_types)
        to_r({str(f): top_genes[str(f)]}, 'top_genes')
        
        r(f'''
        color_lim <- quantile(abs(loadings_mat), 0.99)
        col_fun = colorRamp2(c(-color_lim, 0, color_lim), 
                           c('blue', 'white', 'red'))
        
        row_annot = rowAnnotation(
            foo = anno_mark(
                at = which(rownames(loadings_mat) %in% unlist(top_genes)),
                labels = rownames(loadings_mat)[which(
                    rownames(loadings_mat) %in% unlist(top_genes))],
                labels_gp = gpar(fontsize = 8)
            )
        )
        ht = Heatmap(
            loadings_mat,
            name = paste('Loading', {f}),
            col = col_fun,
            show_row_names = FALSE, 
            show_column_names = TRUE,
            column_names_gp = gpar(fontsize = 12),
            column_names_rot = 45,
            right_annotation = row_annot,
            clustering_method_rows = 'ward.D2',
            cluster_columns = FALSE,
            column_title = paste('Factor', {f+1}),
            column_title_gp = gpar(fontsize = 20, fontface = 'bold'),
            row_title = 'Genes',
            row_title_gp = gpar(fontsize = 14),
            border = TRUE)

        plot_list[[{f+1}]] <- grid.grabExpr(draw(ht))
        ''')
    
    r(f'''
    n_factors = length(plot_list)
    n_cols = min(3, n_factors)
    n_rows = ceiling(n_factors / n_cols)
    
    combined_plot <- ggarrange(
        plotlist = plot_list, 
        ncol = n_cols, 
        nrow = n_rows)
    
    ggsave('{filename}', combined_plot,
           width = 6.75 * n_cols + 2,
           height = 30,
           dpi = 600,
           limitsize = FALSE)
    ''')

# Calculate gene-factor associations using R's lm function
def get_lm_pvals(tensor_dict):
    from utils import fdr
    
    tensor = tensor_dict['tensor']
    scores = tensor_dict['decomposition']['factors']['donors']
    genes = tensor_dict['dims']['genes']
    cell_types = tensor_dict['dims']['cell_types']
    to_r(scores, 'scores')
    
    results = []
    r('n_factors = ncol(scores)')   

    for j, ct in enumerate(cell_types):
        for i, gene in enumerate(genes):
            fiber = tensor[:, i, j]
            if np.all(fiber == 0): continue
                
            to_r(fiber, 'fiber')
            r('''
            stats = lapply(1:n_factors, function(k) {
                fit = lm(scores[,k] ~ fiber)
                summ = summary(fit)
                f_stat = summ$fstatistic
                list(
                    pval = pf(
                        f_stat[1], f_stat[2], f_stat[3], 
                        lower.tail=FALSE),
                    r2 = summ$r.squared,
                    beta = coef(fit)[2],
                    se = summ$coefficients[2,2]
                )
            })
            ''')
            for k in range(scores.shape[1]):
                stats = to_py(f'stats[[{k+1}]]')
                if not np.isnan(stats['pval']):
                    results.append({
                        'gene': gene, 'cell_type': ct, 'factor': k,
                        'pvalue': stats['pval'], 'r2': stats['r2'],
                        'beta': stats['beta'], 'beta_se': stats['se']
                    })
        
    tensor_dict['gene_associations'] = pl.DataFrame(results)\
        .with_columns(fdr(pl.col('pvalue')).alias('padj'))
    return tensor_dict




def plot_gene_loadings_heatmap(tensor_dict, filename, n_genes_per_factor=10,
                              use_sig_only=False, nonsig_to_zero=False):
    loading_tensor = tensor_dict['decomposition']['loadings']['tensor']
    n_factors = loading_tensor.shape[0]
    genes = tensor_dict['dims']['genes']
    cell_types = tensor_dict['dims']['cell_types']
    
    # Get genes to highlight based on significance or loading magnitude
    highlight_genes = {}
    for f in range(n_factors):
        if use_sig_only:
            sig_df = tensor_dict['gene_associations']\
                .filter(pl.col('factor') == f)\
                .filter(pl.col('padj') < 0.1)\
                .sort('padj')\
                .head(n_genes_per_factor)\
                .get_column('gene')
            highlight_genes[str(f)] = sig_df.to_list()
        else:
            factor_loadings = loading_tensor[f]
            highlight_genes[str(f)] = []
            for ct_idx in range(factor_loadings.shape[1]):
                ct_loadings = np.abs(factor_loadings[:, ct_idx])
                top_idx = np.argsort(ct_loadings)[-n_genes_per_factor:]
                highlight_genes[str(f)].extend([genes[i] for i in top_idx])
            highlight_genes[str(f)] = list(dict.fromkeys(highlight_genes[str(f)]))
    
    r('''
    suppressPackageStartupMessages({
        library(ComplexHeatmap)
        library(circlize)
        library(grid)
        library(ggpubr)
    })
    plot_list = list()
    ''')
    
    for f in range(n_factors):
        factor_loadings = loading_tensor[f]
        if nonsig_to_zero and use_sig_only:
            mask = ~np.isin(genes, highlight_genes[str(f)])
            factor_loadings = factor_loadings.copy()
            factor_loadings[mask] = 0
            
        to_r(factor_loadings, 'loadings_mat', 
             rownames=genes, colnames=cell_types)
        to_r({str(f): highlight_genes[str(f)]}, 'highlight_genes')
        
        r(f'''
        max_val = max(abs(loadings_mat))
        if (max_val == 0) {{
            color_lim = 1
        }} else {{
            color_lim = quantile(abs(loadings_mat[loadings_mat != 0]), 0.99)
        }}
        col_fun = colorRamp2(c(-color_lim, 0, color_lim), 
                           c('blue', 'white', 'red'))
        
        row_annot = rowAnnotation(
            foo = anno_mark(
                at = which(rownames(loadings_mat) %in% unlist(highlight_genes)),
                labels = rownames(loadings_mat)[which(
                    rownames(loadings_mat) %in% unlist(highlight_genes))],
                labels_gp = gpar(fontsize = 8)
            )
        )
        ht = Heatmap(
            loadings_mat,
            name = paste('Loading', {f}),
            col = col_fun,
            show_row_names = FALSE, 
            show_column_names = TRUE,
            column_names_gp = gpar(fontsize = 12),
            column_names_rot = 45,
            right_annotation = row_annot,
            clustering_method_rows = 'ward.D2',
            cluster_columns = FALSE,
            column_title = paste('Factor', {f+1}),
            column_title_gp = gpar(fontsize = 20, fontface = 'bold'),
            row_title = 'Genes',
            row_title_gp = gpar(fontsize = 14),
            border = TRUE)

        plot_list[[{f+1}]] = grid.grabExpr(draw(ht))
        ''')
    
    r(f'''
    n_factors = length(plot_list)
    n_cols = min(3, n_factors)
    n_rows = ceiling(n_factors / n_cols)
    
    combined_plot = ggarrange(
        plotlist = plot_list, 
        ncol = n_cols, 
        nrow = n_rows)
    
    ggsave('{filename}', combined_plot,
           width = 6.75 * n_cols + 2,
           height = 30,
           dpi = 300,
           limitsize = FALSE)
    ''')

tensor_dict['sig_gene_counts'] = tensor_dict['gene_associations']\
    .filter(pl.col('padj') < 0.10)\
    .group_by(['cell_type', 'factor'])\
    .agg(pl.count().alias('n_sig_genes'))\
    .sort(['factor', 'cell_type'])
print('\nSignificant genes (FDR < 0.10) per cell type and factor:')
print_df(tensor_dict['sig_gene_counts'])



sc_obs = SingleCell.read_obs(
    f'{data_dir}/{study_name}/{study_name}_qced_labelled.h5ad')

base_FDR, alpha, max_FDR = 0.05, 0.6, 0.3

proportions = sc_obs\
    .group_by(['projid', f'cell_type_{level}'])\
    .len()\
    .with_columns(
        (pl.col('len') / pl.col('len').sum().over('projid'))
        .alias('proportion'))\
    .group_by(f'cell_type_{level}')\
    .agg(pl.median('proportion'))\
    .to_dict(as_series=False)

fdr_thresholds = {
    ct: min(base_FDR * (1 / prop) ** alpha, max_FDR)
    for ct, prop in zip(proportions[f'cell_type_{level}'],
                        proportions['proportion'])
}
print(json.dumps({ct: f'{fdr:.2f}' 
                  for ct, fdr in fdr_thresholds.items()}, indent=2))

# (pl.col('FDR') < fdr_thresholds[ct])) 


overlaps = {}
for ct1 in de_genes:
    for ct2 in de_genes:
        if ct1 < ct2:
            overlap = len(set(de_genes[ct1]) & set(de_genes[ct2]))
            overlaps[f'{ct1}-{ct2}'] = overlap

print('Gene overlaps between cell types:')
print(json.dumps(overlaps, indent=2))