import sys
import json
from pathlib import Path
import numpy as np
import polars as pl
import tensorly as tl
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk, set_num_threads
from utils import debug, print_df, savefig, get_coding_genes
from ryp import r, to_r, to_py

tl.set_backend('numpy') 
set_num_threads(-1)
debug(third_party=True)

data_dir = f'{Path.home()}/projects/def-wainberg/single-cell'
work_dir = f'{Path.home()}/projects/def-wainberg/karbabi/AD-subtyping'

################################################################################

# Create initial tensor_dict with basic data structure
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

    return {
        'original_tensor': tensor,
        'tensor': tensor,
        'dims': {
            'samples': samples,
            'genes': genes,
            'cell_types': cell_types
        },
        'masks': {
            'padding': padding_mask
        }
    }

# Update tensor_dict with padding for specified genes
def pad_tensor(tensor_dict, genes_to_keep, pad_value=0):
    padding_mask = tensor_dict['masks'].get(
        'padding',
        np.ones((len(tensor_dict['dims']['genes']), 
                len(tensor_dict['dims']['cell_types'])),
        dtype=bool))
   
    for ct_idx, ct in enumerate(tensor_dict['dims']['cell_types']):
        if ct in genes_to_keep:
            keep_indices = [
                i for i, g in enumerate(tensor_dict['dims']['genes'])
                if g in genes_to_keep[ct]]
            padding_mask[:, ct_idx] = False
            padding_mask[keep_indices, ct_idx] = True            
            pad_indices = [
                i for i, g in enumerate(tensor_dict['dims']['genes'])
                if g not in genes_to_keep[ct]]
            if pad_indices:
                print(f'{ct}: {len(pad_indices)} genes padded')
            tensor_dict['tensor'][:, pad_indices, ct_idx] = pad_value
    
    tensor_dict['masks']['padding'] = padding_mask
    return tensor_dict

# Add normalized tensor to tensor_dict
def normalize_tensor(tensor_dict, min_shift=False):
    from utils import inverse_normal_transform
    unfolded = tl.unfold(tensor_dict['tensor'], 0)
    normalized = np.apply_along_axis(inverse_normal_transform, 0, unfolded)
    tensor = tl.fold(normalized, 0, tensor_dict['tensor'].shape)
    if min_shift:
        tensor = tensor + abs(np.min(tensor))
    tensor_dict['tensor'] = tensor
    return tensor_dict

# Determine optimal ranks using SVD on unfolded tensor modes
def determine_ranks_svd(tensor_dict, filename, max_ranks=(30, 50)):
    from scipy.linalg import svd
    from kneed import KneeLocator
    tensor = tensor_dict['tensor']
    mode_rank_errors = []

    # Only compute for first two modes
    for mode in [0, 1]:  
        rank_errors = []
        unfolded = tl.unfold(tensor, mode)
        U, s, Vt = svd(unfolded, full_matrices=False)
        
        for rank in range(1, max_ranks[mode] + 1):
            # Reconstruct matrix using truncated SVD
            d = np.diag(s[:rank])
            rec = U[:, :rank] @ d @ Vt[:rank, :]
            
            # Calculate relative Frobenius norm error
            fnorm_relative = (
                np.linalg.norm(rec - unfolded, 'fro')**2 / 
                np.linalg.norm(unfolded, 'fro')**2)
            rank_errors.append(fnorm_relative)
            
        mode_rank_errors.append(rank_errors)

    # Plot results
    best_ranks = []
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    mode_names = ['Sample', 'Gene']
    
    for mode_idx, errors in enumerate(mode_rank_errors):
        kn = KneeLocator(
            range(1, len(errors) + 1),
            errors,
            curve='convex',
            direction='decreasing'
        )
        best_rank = kn.elbow
        best_ranks.append(best_rank)
        
        ax = axes[mode_idx]
        x_vals = range(1, max_ranks[mode_idx] + 1)
        ax.plot(x_vals, errors, 'b-')
        ax.scatter(x_vals, errors, color='blue', s=30, zorder=3)
        ax.axvline(x=best_rank, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of Factors')
        ax.set_ylabel('Relative Error')
        ax.set_title(f'{mode_names[mode_idx]} Mode (Best rank: {best_rank})')
    
    plt.tight_layout()
    savefig(filename)
    best_ranks.append(tensor.shape[2])
    return best_ranks

# Decompose tensor and add results to tensor_dict
def run_tucker_ica(tensor_dict, ranks, random_state=42):
    from sklearn.decomposition import FastICA
    from scipy.linalg import svd
    
    # Initial decomposition
    _, factors = tl.decomposition.tucker(
        tensor_dict['tensor'], 
        rank=ranks, 
        init='svd', 
        random_state=random_state)
    donor_mat, gene_mat, ct_mat = factors
    
    # ICA rotation on genes
    ica = FastICA(
        n_components=ranks[1], 
        random_state=random_state, 
        tol=1e-3, 
        max_iter=1000)
    gene_factors = ica.fit_transform(gene_mat)
    norms = np.sqrt(np.sum(gene_factors**2, axis=0))
    gene_factors = gene_factors / norms[None, :]
    
    # Compute loadings
    kron_prod = np.kron(gene_factors, ct_mat)
    core_unrotated = donor_mat.T @ \
        tl.unfold(tensor_dict['tensor'], 0) @ kron_prod
    loadings_unrotated = core_unrotated @ kron_prod.T
    
    # Varimax rotation
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

    # Compute loading tensor
    loading_tensor = core_rotated.copy()
    for mode in [1, 2]:
        loading_tensor = tl.tenalg.mode_dot(
            loading_tensor,
            [gene_factors, ct_mat][mode-1],
            mode=mode)

    tensor_dict.update({
        'decomposition': {
            'factors': {
                'donors': donor_scores,    # Sample scores for downstream analysis
                'genes': gene_factors,     # Cell-type agnostic gene programs
                'cell_types': ct_mat       # Cell type weights
            },
            'core': {
                'unrotated': core_unrotated,  # Original factor structure
                'rotated': core_rotated       # Interpretable rotated structure
            },
            'loadings': {
                'unrotated': loadings_unrotated,  # Pre-rotation loadings
                'tensor': loading_tensor,         # Final gene patterns per cell type
                'ranks': ranks                    # Dimensions of decomposition
            },
            'rotations': {
                'varimax': rot_mat    # For projecting new data
            }
        }
    })
    return tensor_dict

# Get metadata associations with factor donor scores.
def get_meta_associations(tensor_dict, meta, max_na_pct=0.2):
    if 'ID' in meta.columns:
        meta = meta.drop('ID')
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    to_r(donor_scores, 'donor_scores')
    to_r(meta, 'meta')
    to_r(max_na_pct, 'max_na_pct')

    r('''
    meta_rsq = function(donor_scores, meta, max_na_pct) {
        na_pct = colMeans(is.na(meta))
        vars_use = names(which(na_pct <= max_na_pct))
        res = matrix(
            0, 
            nrow=length(vars_use), 
            ncol=ncol(donor_scores),
            dimnames=list(vars_use, 
                          paste0('factor_', 1:ncol(donor_scores))))
        
        for (var in vars_use) {
            complete = !is.na(meta[[var]])
            var_data = if(is.character(meta[[var]])) 
                        factor(meta[[var]][complete]) 
                      else 
                        meta[[var]][complete]
            
            res[var,] = sapply(1:ncol(donor_scores), function(i) {
                max(0, summary(
                    lm(donor_scores[complete,i] ~ var_data)
                )$adj.r.squared)
            })
        }
        res
    }
    result = meta_rsq(donor_scores, meta, max_na_pct)
    ''')
    
    tensor_dict['meta_associations'] = to_py('result')
    return tensor_dict


# Plot heatmap of donor scores with optional metadata annotations.
def plot_donor_loadings_heatmap(tensor_dict, meta, filename, meta_vars=None):
    if 'ID' in meta.columns:
        meta = meta.drop('ID')
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    to_r(donor_scores, 'donor_scores', 
         rownames=tensor_dict['dims']['samples'],
         colnames=[f'Factor {i+1}' for i in range(donor_scores.shape[1])])
    to_r(tensor_dict['meta_associations'].drop('index'), 'rsq', 
         rownames=tensor_dict['meta_associations']['index'])
    if meta_vars is not None:
        to_r(meta.select(meta_vars), 'meta')
    to_r(filename, 'filename')

    r('''
    suppressPackageStartupMessages({
        library(ComplexHeatmap)
        library(circlize)
        library(grid)
    })
    
    score_lim = quantile(abs(donor_scores), 0.99)
    score_colors = colorRamp2(
        c(-score_lim, 0, score_lim),
        c('blue', 'white', 'red')
    )
    rsq_lim = quantile(as.matrix(rsq), 0.99)
    rsq_colors = colorRamp2(
        c(0, rsq_lim),
        c('white', '#1B7837')
    )
      
    ha_list = list()
    if (exists('meta')) {
        ha_list$meta = rowAnnotation(
            df = meta,
            show_legend = FALSE,
            annotation_name_gp = gpar(fontsize = 8),
            simple_anno_size = unit(0.25, 'cm')
        )
    }    
    ha_list$rsq = HeatmapAnnotation(
        rsq = t(rsq),
        col = list(rsq = rsq_colors),
        show_legend = TRUE,
        annotation_name_gp = gpar(fontsize = 8),
        simple_anno_size = unit(0.25, 'cm')
    )
    
    ht = Heatmap(
        donor_scores,
        name = 'loading',
        col = score_colors,
        cluster_columns = FALSE,
        show_row_names = FALSE,
        row_title = 'Donors',
        row_title_gp = gpar(fontsize = 8),
        column_title = NULL,
        column_names_gp = gpar(fontsize = 8),
        top_annotation = ha_list$rsq,
        left_annotation = ha_list$meta,
        clustering_method_rows = 'ward.D2',
        show_row_dend = FALSE,
        width = unit(6, "cm"), height = unit(10, "cm")
    )
    ''')
    
    r(f'''
    png(filename, units='cm', width=20, height=30, res=300)
    draw(ht, 
         heatmap_legend_side = "right",
         annotation_legend_side = "right",
         padding = unit(c(0.5, 0.5, 0.5, 0.5), "cm"),
         auto_adjust = FALSE)
    dev.off()
    ''')

# Plot gene loadings heatmap for each factor across cell type
def plot_gene_loadings_heatmap(tensor_dict, filename, n_genes_per_factor=10):
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
    plot_list = list()
    ''')
    
    for f in range(n_factors):
        factor_loadings = loading_tensor[f]
        to_r(factor_loadings, 'loadings_mat', 
             rownames=genes, colnames=cell_types)
        to_r({str(f): top_genes[str(f)]}, 'top_genes')
        
        r(f'''
        color_lim = quantile(abs(loadings_mat), 0.99)
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

        plot_list[[{f+1}]] = grid.grabExpr(draw(ht))
        ''')
    
    r(f'''
    n_factors = length(plot_list)
    n_cols = min(4, n_factors)
    n_rows = ceiling(n_factors / n_cols)
    
    combined_plot = ggarrange(
        plotlist = plot_list, 
        ncol = n_cols, 
        nrow = n_rows)
    
    ggsave('{filename}', combined_plot,
           width = 6.75 * n_cols + 2,
           height = 10 * n_rows,
           dpi = 300,
           limitsize = FALSE)
    ''')

def plot_donor_sig_genes(tensor_dict, filename, n_genes_per_ct=10, 
                         selection='both'):
        
    loading_tensor = tensor_dict['decomposition']['loadings']['tensor']
    genes = tensor_dict['dims']['genes']
    cell_types = tensor_dict['dims']['cell_types']
    samples = tensor_dict['dims']['samples']
    n_factors = loading_tensor.shape[0]
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    
    plot_data_list = []
    gene_labels_list = []
    ct_list = []
    
    for factor in range(n_factors):
        factor_genes = []
        factor_cts = []
        factor_data = []
        
        for ct_idx, ct in enumerate(cell_types):
            ct_loadings = loading_tensor[factor, :, ct_idx]
            
            if selection == 'both':
                top_idx = np.argsort(np.abs(ct_loadings))[-n_genes_per_ct:][::-1]
            elif selection == 'positive':
                top_idx = np.argsort(ct_loadings)[-n_genes_per_ct:][::-1]
            else:
                top_idx = np.argsort(ct_loadings)[:n_genes_per_ct]
            
            for gene_idx in top_idx:
                gene_data = tensor_dict['tensor'][:, gene_idx, ct_idx]
                factor_data.append(gene_data)
                factor_genes.append(genes[gene_idx])
                factor_cts.append(ct)
        
        plot_data_list.append(np.array(factor_data))
        gene_labels_list.append(factor_genes)
        ct_list.append(factor_cts)

    for i in range(n_factors):
        to_r(plot_data_list[i], f'expr_mat_{i+1}', format='matrix',
             rownames=gene_labels_list[i], colnames=samples)
        to_r(ct_list[i], f'ct_{i+1}')
        to_r(gene_labels_list[i], f'genes_{i+1}')
    
    to_r(donor_scores, 'donor_scores', format='matrix',
         rownames=samples, colnames=[f"Factor_{i+1}" for i in range(n_factors)])
    to_r(filename, 'filename'); to_r(n_factors, 'n_factors')

    r('''
    suppressPackageStartupMessages({
        library(ComplexHeatmap)
        library(circlize)
        library(grid)
        library(RColorBrewer)
    })
    
    ct_colors = c(
        "Astrocyte" = "#E41A1C",
        "Endothelial" = "#FFFF33", 
        "Excitatory" = "#4DAF4A",
        "Inhibitory" = "#377EB8",
        "Microglia-PVM" = "#984EA3",
        "OPC" = "#FF7F00",
        "Oligodendrocyte" = "#A65628"
    )
    
    ht_list = NULL
    for(i in 1:n_factors) {
        expr_mat = get(paste0("expr_mat_", i))
        ct = factor(get(paste0("ct_", i)), levels=names(ct_colors))
        genes = get(paste0("genes_", i))
        
        sample_order = order(donor_scores[,i])
        expr_mat = expr_mat[, sample_order]
        
        max_val = max(abs(expr_mat))
        min_val = -max_val
        
        ha = HeatmapAnnotation(
            score = donor_scores[sample_order, i],
            col = list(score = colorRamp2(
                c(min(donor_scores[,i]), 0, max(donor_scores[,i])), 
                c('purple', 'white', 'green'))),
            show_legend = TRUE,
            height = unit(5, 'mm')
        )
        
        row_ha = rowAnnotation(
            cell_type = ct,
            col = list(cell_type = ct_colors),
            show_annotation_name = FALSE,
            width = unit(2, 'mm')
        )

        row_ha_right = rowAnnotation(
            gene = anno_text(genes, 
                           location = 0,
                           just = 'left',
                           gp = gpar(fontsize = 7)),
            width = unit(30, 'mm')
        )
        
        ht = Heatmap(expr_mat,
            name = paste('Factor', i),
            cluster_rows = FALSE,
            cluster_columns = FALSE,
            show_row_names = FALSE,
            show_column_names = FALSE,
            col = colorRamp2(c(min_val, 0, max_val), c('blue', 'white', 'red')),
            left_annotation = row_ha,
            right_annotation = row_ha_right,
            bottom_annotation = ha,
            row_split = ct,
            row_gap = unit(1, 'mm'),
            column_title = paste('Factor', i),
            column_title_gp = gpar(fontsize = 10, fontface = 'bold'),
            width = unit(60, 'mm')
        )
        
        if(is.null(ht_list)) {
            ht_list = ht
        } else {
            ht_list = ht_list + ht
        }
    }
    
    png(filename, width=n_factors*4, height=10, units='in', res=300)
    draw(ht_list, 
         ht_gap = unit(-12, 'mm'),
         padding = unit(c(10, 10, 10, 10), 'mm'))
    dev.off()
    ''')

def plot_scores_by_meta(tensor_dict, filename, meta, meta_var, factors=None):
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    if factors is not None:
        if isinstance(factors, int):
            factors = [factors]
        factor_idx = [i-1 for i in factors]
        donor_scores = donor_scores[:, factor_idx]
        factor_names = [f'Factor_{i}' for i in factors]
    else:
        factor_names = [f'Factor_{i+1}' for i in range(donor_scores.shape[1])]
    
    to_r(donor_scores, 'donor_scores', 
         rownames=tensor_dict['dims']['samples'],
         colnames=factor_names)
    to_r(meta, 'meta')
    to_r(meta_var, 'meta_var')
    to_r(filename, 'filename')

    r('''
    library(ggplot2)
    library(ggpubr)
    
    complete_idx = !is.na(meta[[meta_var]])
    meta_subset = meta[complete_idx, ]
    scores_subset = as.matrix(donor_scores[complete_idx, , drop=FALSE])
    meta_val = meta_subset[[meta_var]]
    
    n_unique = length(unique(meta_val))
    is_categorical = n_unique < 10
    
    if (is_categorical) {
        meta_val = factor(meta_val)
    } else if (is.character(meta_val) || is.factor(meta_val)) {
        warning('Converting metadata to numeric values')
        meta_val = as.numeric(as.character(meta_val))
    }
    
    if (ncol(scores_subset) == 1) {
        plot_data = data.frame(
            dscore = scores_subset[,1],
            meta_val = meta_val
        )
        
        if (is_categorical) {
            p = ggplot(plot_data, aes(x=meta_val, y=dscore)) +
                geom_boxplot(width=0.5, outlier.shape=NA) +
                geom_jitter(width=0.2, alpha=0.5) +
                stat_compare_means(method='anova') +
                xlab(meta_var) +
                ylab('Score') +
                ggtitle(colnames(scores_subset)[1]) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5))
        } else {
            p = ggplot(plot_data, aes(x=meta_val, y=dscore)) +
                geom_point(alpha=0.5, shape=18) +
                geom_smooth(method='lm', color='blue', se=FALSE) +
                stat_cor(method='pearson') +
                xlab(meta_var) +
                ylab('Score') +
                ggtitle(colnames(scores_subset)[1]) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5))
        }
        ggsave(filename, p, width = 3, 
               height = 5, dpi = 300)
    } else {
        plot_list = list()
        for (i in 1:ncol(scores_subset)) {
            plot_data = data.frame(
                dscore = scores_subset[,i],
                meta_val = meta_val
            )
            if (is_categorical) {
                p = ggplot(plot_data, aes(x=meta_val, y=dscore)) +
                    geom_boxplot(width=0.5, outlier.shape=NA) +
                    geom_jitter(width=0.2, alpha=0.5) +
                    stat_compare_means(method='anova') +
                    xlab(meta_var) +
                    ylab('Score') +
                    ggtitle(colnames(scores_subset)[i]) +
                    theme_classic() +
                    theme(plot.title = element_text(hjust = 0.5))
            } else {
                p = ggplot(plot_data, aes(x=meta_val, y=dscore)) +
                    geom_point(alpha=0.5, shape=18) +
                    geom_smooth(method='lm', color='blue', se=FALSE) +
                    stat_cor(method='pearson') +
                    xlab(meta_var) +
                    ylab('Score') +
                    ggtitle(colnames(scores_subset)[i]) +
                    theme_classic() +
                    theme(plot.title = element_text(hjust = 0.5))
            }
            plot_list[[i]] = p
        }
        p_final = ggarrange(
            plotlist = plot_list,
            nrow = 1,
            ncol = length(plot_list),
            common.legend = TRUE
        )
        ggsave(filename, p_final,
               width = min(3 * ncol(scores_subset), 20),
               height = 5,
               dpi = 300,
               limitsize = FALSE)
    }
    ''')

def get_gene_loading_table(tensor_dict):
    loading_tensor = tensor_dict['decomposition']['loadings']['tensor']
    genes = tensor_dict['dims']['genes']
    cell_types = tensor_dict['dims']['cell_types']
    n_factors = loading_tensor.shape[0]
    
    return pl.DataFrame({
        'gene': [g for g in genes 
                for _ in range(n_factors * len(cell_types))],
        'factor': [f + 1 for _ in range(len(genes)) 
                  for f in range(n_factors) 
                  for _ in range(len(cell_types))],
        'cell_type': [ct for _ in range(len(genes) * n_factors) 
                     for ct in cell_types],
        'loading': loading_tensor.reshape(-1)
    })

def get_sample_loading_table(tensor_dict):
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    n_factors = donor_scores.shape[1]
    return pl.DataFrame({
        'sample': [s for s in tensor_dict['dims']['samples'] 
                  for _ in range(n_factors)],
        'factor': [f + 1 for _ in tensor_dict['dims']['samples']
                  for f in range(n_factors)],
        'loading': donor_scores.reshape(-1)
    })

################################################################################

study_name = 'Green'
level = 'broad'

de = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .qc(group_column='pmAD',
        custom_filter=pl.col('pmAD').is_not_null(),
        verbose=False)\
    .DE(formula='~ pmAD + age_death + sex + pmi + apoe4_dosage',
        coefficient='pmAD',
        group='pmAD',
        verbose=True)

de_genes = {
    ct: de.table.filter(
        (pl.col('cell_type') == ct) & 
        (pl.col('p') < 0.05)
    ).get_column('gene').to_list()
    for ct in de.table['cell_type'].unique()
}
print(json.dumps({ct: len(genes) for ct, genes in de_genes.items()}, indent=2))

lcpm = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .filter_obs((pl.col('dx_cogn').gt(0) & pl.col('dx_cogn').is_not_null()))\
    .qc(group_column=None, 
        verbose=False)\
    .log_CPM()\
    .regress_out('~ pmi',
                 library_size_as_covariate=True,
                 num_cells_as_covariate=True) 

genes = sorted(set().union(*[
    set(gene_list) for gene_list in de_genes.values()
]))
print(len(genes))

tensor_dict = prepare_tensor_dict(lcpm, genes=genes)
tensor_dict = pad_tensor(tensor_dict, de_genes)
tensor_dict = normalize_tensor(tensor_dict, min_shift=False)

print(tensor_dict['tensor'].shape)
print(np.max(tensor_dict['tensor']), np.min(tensor_dict['tensor']))

# best_ranks = determine_ranks_svd(
#     tensor_dict, 
#     f'{work_dir}/figures/svd_ranks.png',
#     max_ranks=(50, 100)) \
    
# print(best_ranks)

tensor_dict = run_tucker_ica(tensor_dict, ranks=[6, 30, 7])

rosmap_codes = pl.read_csv(
    f'{data_dir}/{study_name}/rosmap_codes_edits.csv')\
    .filter(pl.col.priority | pl.col.code.is_in(['pmi']))\
    .with_columns(pl.col.name.str.replace_all(' - OLD', ''))\
    .select(['code', 'name', 'category'])
code_to_name = dict(rosmap_codes.select(['code', 'name']).iter_rows())

meta = lcpm.obs['Astrocyte']\
    .filter(pl.col.ID.is_in(tensor_dict['dims']['samples']))\
    .select(['ID'] + rosmap_codes['code'].to_list())\
    .rename(code_to_name)\
    .with_columns([
        pl.col('CERAD score').reverse().alias('CERAD score'),
        pl.col('NIA-Reagan diagnosis of AD').reverse()
        .alias('NIA-Reagan diagnosis of AD')])
meta = meta.select(['ID'] + sorted(set(meta.columns) - {'ID'}))

tensor_dict = get_meta_associations(tensor_dict, meta, max_na_pct=0.4)

plot_donor_loadings_heatmap(
    tensor_dict, 
    meta,
    f'{work_dir}/figures/donor_loadings.png',
    meta_vars=None)

plot_scores_by_meta(
    tensor_dict,
    f'{work_dir}/figures/scores_by_var.png',
    meta,
    factors=[1, 4],
    meta_var='Pathological AD')

plot_gene_loadings_heatmap(
    tensor_dict, 
    f'{work_dir}/figures/gene_loadings.png',
    n_genes_per_factor=5)

plot_donor_sig_genes(
    tensor_dict, 
    f'{work_dir}/figures/donor_sig_genes.png',
    n_genes_per_ct=10,
    selection='positive')



get_gene_loading_table(tensor_dict)\
    .with_columns(pl.col('loading').sign().alias('sign'))\
    .sort('loading', descending=True)\
    .group_by(['factor', 'cell_type', 'sign'])\
    .head(30)\
    .sort(['factor', 'cell_type', 'loading'], descending=[False, False, True])\
    .write_csv(f'{work_dir}/gene_loadings.csv')

sample_loading_table = get_sample_loading_table(tensor_dict)













data = tl.unfold(tensor_dict['decomposition']['loadings']['loadings_unrotated'], 1).T
print(data.shape)

plt.figure(figsize=(8, 15))
sns.heatmap(data, 
            cmap='RdBu_r',
            center=0,
            robust=True,
            xticklabels=False,
            yticklabels=False)
savefig(f'{work_dir}/figures/tmp1.png')

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


