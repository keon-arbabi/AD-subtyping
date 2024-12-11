import os
import sys
import json
from pathlib import Path
import numpy as np
import polars as pl
import tensorly as tl
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('projects/utils')
from single_cell import Pseudobulk, set_num_threads
from utils import debug, print_df, savefig
from ryp import r, to_r, to_py

tl.set_backend('numpy') 
set_num_threads(-1)
debug(third_party=True)

data_dir = f'{Path.home()}/projects/single-cell'
work_dir = f'{Path.home()}/projects/AD-subtyping'

################################################################################

# Create initial tensor_dict with basic data structure
def prepare_tensor_dict(pb, genes=None, donors=None):
    if genes is None:
        genes = sorted(set.intersection(
            *[set(var['_index']) for var in pb.iter_var()]))
    if donors is None:
        donors = sorted(set.intersection(
            *[set(obs['ID']) for obs in pb.iter_obs()]))
   
    cell_types = list(pb.keys())    
    tensor = tl.zeros((len(donors), len(genes), len(cell_types)))
    gene_map = {g: i for i, g in enumerate(genes)}    
    padding_mask = np.zeros((len(genes), len(cell_types)), dtype=bool)

    for ct_idx, (ct_name, (X, obs, var)) in enumerate(pb.items()):
        source_idx = [i for i, g in enumerate(var['_index']) if g in gene_map]
        target_idx = [gene_map[var['_index'][i]] for i in source_idx]
        sample_idx = [i for i, s in enumerate(obs['ID']) if s in donors]
        n_padded = len(genes) - len(target_idx)
        if n_padded > 0:
            print(f"{ct_name}: {n_padded} genes padded")
        tensor[:, target_idx, ct_idx] = X[np.ix_(sample_idx, source_idx)]        
        padding_mask[target_idx, ct_idx] = True

    return {
        'original_tensor': tensor,
        'tensor': tensor,
        'dims': {
            'donors': donors,
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

def get_meta_associations(tensor_dict, meta, max_na_pct=0.4, adjust_pvals=True):
    donor_ids = tensor_dict['dims']['donors']
    meta = meta.filter(pl.col('ID').is_in(donor_ids))
    meta = meta.sort('ID')
    if 'ID' in meta.columns:
        meta = meta.drop('ID')
        
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    to_r(donor_scores, 'donor_scores')
    to_r(meta, 'meta')
    to_r(max_na_pct, 'max_na_pct')
    to_r(adjust_pvals, 'adjust_pvals')

    r('''
    meta_assoc = function(donor_scores, meta, na_thresh, adjust_pvals) {  
        na_pct = colMeans(is.na(meta))
        vars_use = names(which(na_pct <= na_thresh))
        
        rsq = matrix(0, nrow=length(vars_use), ncol=ncol(donor_scores),
                    dimnames=list(vars_use, colnames(donor_scores)))
        cor = rsq  
        pvals = rsq  
        
        for (var in vars_use) {
            complete = !is.na(meta[[var]])
            var_data = if(is.character(meta[[var]])) 
                        factor(meta[[var]][complete]) 
                      else 
                        meta[[var]][complete]
            
            for(i in 1:ncol(donor_scores)) {
                fit = lm(donor_scores[complete,i] ~ var_data)
                r2 = summary(fit)$adj.r.squared
                coef_matrix = summary(fit)$coefficients
                
                if(nrow(coef_matrix) > 1) {
                    pvals[var,i] = coef_matrix[2,4]
                    cor[var,i] = cor(donor_scores[complete,i], 
                                   as.numeric(var_data), 
                                   use="complete.obs")
                    rsq[var,i] = sign(cor[var,i]) * max(0, r2)
                } else {
                    pvals[var,i] = 1
                    cor[var,i] = 0
                    rsq[var,i] = 0
                }
            }
        }
        if(adjust_pvals) {
            pvals = apply(pvals, 2, p.adjust, method="fdr")
        }
        dimnames(pvals) = dimnames(rsq)
        
        for(i in 1:ncol(rsq)) {
            cat(sprintf("\nTop associations for Factor %d:\n", i))
            idx = order(abs(rsq[,i]), decreasing=TRUE)[1:5]
            for(j in idx) {
                cat(sprintf("%s: R=%.3f, R²=%.3f, p=%.2e\n", 
                    rownames(rsq)[j], cor[j,i], rsq[j,i], pvals[j,i]))
            }
        }
        list(rsq=rsq, cor=cor, pvals=pvals)
    }
    result = meta_assoc(donor_scores, meta, max_na_pct, adjust_pvals)
    ''')

    tensor_dict['meta_associations'] = to_py('result')
    return tensor_dict

def plot_donor_loadings_heatmap(tensor_dict, filename):
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    meta_assoc = tensor_dict['meta_associations']
    
    to_r(donor_scores, 'donor_scores', 
         rownames=tensor_dict['dims']['donors'], 
         colnames=[f'Factor {i+1}' for i in range(donor_scores.shape[1])])
    to_r(meta_assoc['rsq'].drop('index'), 'result_rsq', 
         rownames=meta_assoc['rsq']['index'].to_list())
    to_r(meta_assoc['pvals'].drop('index'), 'result_pvals',
         rownames=meta_assoc['pvals']['index'].to_list())
    to_r(filename, 'filename')
    
    r('''
    suppressPackageStartupMessages({
      library(ComplexHeatmap)
      library(circlize)
      library(grid)
    })

    rsq_lim = quantile(abs(as.matrix(result_rsq)), 0.99)
    rsq_colors = colorRamp2(
      c(-rsq_lim, 0, rsq_lim),
      c('#7B3294', 'white', '#008837')
    )
    
    score_lim = quantile(abs(donor_scores), 0.99)
    score_colors = colorRamp2(
      c(-score_lim, 0, score_lim),
      c('blue', 'white', 'red')
    )
    
    ha_list = list()
    ha_list$rsq = HeatmapAnnotation(
      rsq = t(result_rsq),
      col = list(rsq = rsq_colors),
      show_legend = TRUE,
      annotation_name_gp = gpar(fontsize = 8),
      simple_anno_size = unit(0.3, 'cm'),
      annotation_legend_param = list(
        title_gp = gpar(fontsize = 8, fontface = "bold"),
        labels_gp = gpar(fontsize = 8)
      )
    )
    
    ht = Heatmap(
      donor_scores,
      name = 'loading',
      col = score_colors,
      cluster_columns = FALSE,
      show_row_names = FALSE,
      row_title = sprintf('Donors (N=%d)', nrow(donor_scores)),
      row_title_gp = gpar(fontsize = 8),
      column_title = NULL,
      column_names_gp = gpar(fontsize = 8),
      top_annotation = ha_list$rsq,
      clustering_method_rows = 'ward.D2',
      show_row_dend = FALSE,
      heatmap_legend_param = list(
        title_gp = gpar(fontsize = 8, fontface = "bold"),
        labels_gp = gpar(fontsize = 8)
      ),
      width = unit(6, "cm"),
      height = unit(10, "cm")
    )
    
    png(filename, units='cm', width=20, height=30, res=300)
    draw(ht, 
         heatmap_legend_side = "right",
         annotation_legend_side = "right",
         padding = unit(c(0.5, 0.5, 2, 0.5), "cm"),
         auto_adjust = FALSE)
         
    decorate_annotation("rsq", {
      pvals = result_pvals
      for(i in 1:nrow(pvals)) {
        for(j in 1:ncol(pvals)) {
          v = pvals[i,j]
          if(!is.na(v) && !is.null(v)) { 
            y_pos = 1 - ((i-0.5)/nrow(pvals))
            x_pos = (j-0.5)/ncol(pvals)
            
            if(v < 0.001) {
              grid.text("***", x = unit(x_pos, "npc"), y = unit(y_pos, "npc"),
                       gp = gpar(fontsize = 8, col = "white"))
            } else if(v < 0.01) {
              grid.text("**", x = unit(x_pos, "npc"), y = unit(y_pos, "npc"),
                       gp = gpar(fontsize = 8, col = "white"))
            } else if(v < 0.05) {
              grid.text("*", x = unit(x_pos, "npc"), y = unit(y_pos, "npc"),
                       gp = gpar(fontsize = 8, col = "white"))
            }
          }
        }
      }
    })
    dev.off()
    ''')

def plot_gene_loadings_heatmap(tensor_dict, filename, n_genes_per_factor=10):
    loading_tensor = tensor_dict['decomposition']['loadings']['tensor']
    n_factors = loading_tensor.shape[0]
    genes = tensor_dict['dims']['genes']
    cell_types = tensor_dict['dims']['cell_types']
    
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
    donors = tensor_dict['dims']['donors']
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
             rownames=gene_labels_list[i], colnames=donors)
        to_r(ct_list[i], f'ct_{i+1}')
        to_r(gene_labels_list[i], f'genes_{i+1}')
    
    to_r(donor_scores, 'donor_scores', format='matrix',
         rownames=donors, colnames=[f"Factor_{i+1}" for i in range(n_factors)])
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
                           gp = gpar(fontsize = 5.5)),
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
    donor_ids = tensor_dict['dims']['donors']
    
    meta = meta.filter(pl.col('ID').is_in(donor_ids))
    meta = meta.sort('ID')
    
    if factors is not None:
        donor_scores = donor_scores[:, [f-1 for f in factors]]
        factor_names = [f'Factor {f}' for f in factors]
    else:
        factor_names = [f'Factor {i+1}' for i in range(donor_scores.shape[1])]
    
    to_r(donor_scores, 'donor_scores', 
         rownames=donor_ids,
         colnames=factor_names)
    to_r(meta, 'meta')
    to_r(meta_var, 'meta_var')
    to_r(filename, 'filename')

    r('''
    library(ggplot2)
    library(ggpubr)
    
    complete_idx = !is.na(meta[[meta_var]])
    meta_subset = meta[complete_idx, ]
    scores_subset = donor_scores[complete_idx, , drop=FALSE]
    meta_val = meta_subset[[meta_var]]
    n_unique = length(unique(meta_val))
    is_categorical = n_unique < 6
    
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
                geom_jitter(width=0.2, alpha=0.5, shape=18, color='#7B3294') +
                stat_compare_means(method='anova') +
                xlab(stringr::str_wrap(meta_var, width=20)) +
                ylab('Score') +
                ggtitle(colnames(scores_subset)[1]) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5))
        } else {
            p = ggplot(plot_data, aes(x=meta_val, y=dscore)) +
                geom_point(alpha=0.5, shape=18, color='#7B3294') +
                geom_smooth(method='lm', color='black', se=FALSE) +
                stat_cor(method='pearson') +
                xlab(stringr::str_wrap(meta_var, width=20)) +
                ylab('Score') +
                ggtitle(colnames(scores_subset)[1]) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5))
        }
        ggsave(filename, p, width = 3, height = 5, dpi = 300)
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
                    geom_jitter(width=0.2, alpha=0.5, shape=18, color='#7B3294') +
                    stat_compare_means(method='anova') +
                    xlab(stringr::str_wrap(meta_var, width=20)) +
                    ylab('Score') +
                    ggtitle(colnames(scores_subset)[i]) +
                    theme_classic() +
                    theme(plot.title = element_text(hjust = 0.5))
            } else {
                p = ggplot(plot_data, aes(x=meta_val, y=dscore)) +
                    geom_point(alpha=0.5, shape=18, color='#7B3294') +
                    geom_smooth(method='lm', color='black', se=FALSE) +
                    stat_cor(method='pearson') +
                    xlab(stringr::str_wrap(meta_var, width=20)) +
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

def project_tucker_ica(tensor_dict, new_pb, de_genes):
    ref_cell_types = set(tensor_dict['dims']['cell_types'])
    new_cell_types = set(new_pb.keys())
    if not ref_cell_types.issubset(new_cell_types):
        missing = ref_cell_types - new_cell_types
        raise ValueError(f'Missing cell types in new data: {missing}')
    
    ref_genes = set(tensor_dict['dims']['genes'])
    new_genes = set.intersection(*[
        set(var['_index'])for var in new_pb.iter_var()])
    genes_use = sorted(ref_genes.intersection(new_genes))
    
    new_tensor_dict = prepare_tensor_dict(
        new_pb, 
        genes=genes_use,
        donors=sorted(set.intersection(
            *[set(obs['ID']) for obs in new_pb.iter_obs()])))
    new_tensor_dict = pad_tensor(new_tensor_dict, de_genes)
    new_tensor_dict = normalize_tensor(new_tensor_dict, min_shift=False)
    
    # Get unrotated loadings
    ldngs = tensor_dict['decomposition']['loadings']['unrotated'] 

    # Create gene:celltype labels and get column indices
    ref_ct_g = []
    for ct in tensor_dict['dims']['cell_types']:
        for g in tensor_dict['dims']['genes']:
            ref_ct_g.append(f'{ct}:{g}')

    new_ct_g, col_idx = [], []
    for ct in tensor_dict['dims']['cell_types']:
        for g in genes_use:
            new_ct_g.append(f'{ct}:{g}')
            if f'{ct}:{g}' in ref_ct_g:
                col_idx.append(ref_ct_g.index(f'{ct}:{g}'))
    
    # Subset loadings to matching genes/cell types
    ldngs = ldngs[:, col_idx]
    # Project new data
    unfolded = tl.unfold(new_tensor_dict['tensor'], 0)
    projection = unfolded @ ldngs.T
    # Normalize projections
    norms = np.sqrt(np.sum(projection**2, axis=0))
    projection = projection / norms[None, :]
    # Apply rotation matrix
    rot_mat = tensor_dict['decomposition']['rotations']['varimax']
    projected_scores = projection @ rot_mat

    new_tensor_dict['decomposition'] = {
        'factors': {
            'donors': projected_scores,    # New donor projections
            'genes': tensor_dict['decomposition']['factors']['genes'],     
            'cell_types': tensor_dict['decomposition']['factors']['cell_types']
        },
        'core': tensor_dict['decomposition']['core'].copy(),
        'loadings': tensor_dict['decomposition']['loadings'].copy(),
        'rotations': tensor_dict['decomposition']['rotations'].copy()
    }
    return new_tensor_dict

def plot_gsea_one_factor(tensor_dict, filename, factor_select, thresh=0.05,
                       signed=True, min_gs_size=15, max_gs_size=500):
    from ryp import r, to_r
    import os
    
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    to_r(donor_scores, 'donor_scores_mat', 
         rownames=tensor_dict['dims']['donors'])
    
    cell_types = tensor_dict['dims']['cell_types']
    genes = tensor_dict['dims']['genes']
    
    r('''
    library(scITD)
    library(colorRamp2)
    library(ComplexHeatmap)
    source('projects/scITD/R/run_gsea.R', local=FALSE)
    scMinimal_ctype <- list()
    experiment_params <- list(ncores=as.integer(1))
    
    donor_scores <- matrix(donor_scores_mat, 
                          nrow=nrow(donor_scores_mat),
                          ncol=ncol(donor_scores_mat))
    rownames(donor_scores) <- rownames(donor_scores_mat)
    ''')
    
    for ct in cell_types:
        exp_data = tensor_dict['tensor'][:, :, cell_types.index(ct)]
        to_r(exp_data, 'exp_mat', 
             rownames=tensor_dict['dims']['donors'],
             colnames=genes)
        r(f'scMinimal_ctype$`{ct}` <- list(pseudobulk=exp_mat)')
    
    to_r(factor_select, 'factor_select')
    to_r(thresh, 'thresh')
    to_r(signed, 'signed')
    to_r(min_gs_size, 'min_gs_size')
    to_r(max_gs_size, 'max_gs_size')
    to_r(filename, 'filename')
    to_r(os.cpu_count(), 'ncores')
    
    return r('''
    container <- list()
    container$gsea_results <- list()
    up_sets_all <- list()
    down_sets_all <- list()
    
    for (ct in names(scMinimal_ctype)) {
        fgsea_res <- run_fgsea(list(tucker_results=list(donor_scores),
                                   scMinimal_ctype=scMinimal_ctype,
                                   experiment_params=experiment_params),
                              factor_select, ct, db_use='GO', 
                              signed=signed, min_gs_size=min_gs_size, 
                              max_gs_size=max_gs_size, ncores=ncores)
        
        up_sets_names <- fgsea_res$pathway[fgsea_res$NES > 0]
        up_sets <- fgsea_res$padj[fgsea_res$NES > 0]
        names(up_sets) <- up_sets_names
        
        down_sets_names <- fgsea_res$pathway[fgsea_res$NES < 0]
        down_sets <- fgsea_res$padj[fgsea_res$NES < 0]
        names(down_sets) <- down_sets_names
        
        up_sets_all[[ct]] <- up_sets
        down_sets_all[[ct]] <- down_sets
    }
    
    container$gsea_results[[as.character(factor_select)]] <- 
        list(up=up_sets_all, down=down_sets_all)
    
    hmap_list <- plot_gsea_hmap(container, factor_select, thresh)
    
    png(filename, width=6, height=8, units='in', res=300)
    draw(hmap_list, padding=unit(c(0, 4, 0, 2), "cm"))
    dev.off()
    
    container$gsea_results[[as.character(factor_select)]]
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

def get_donor_loading_table(tensor_dict):
    donor_scores = tensor_dict['decomposition']['factors']['donors']
    n_factors = donor_scores.shape[1]
    return pl.DataFrame({
        'sample': [s for s in tensor_dict['dims']['donors'] 
                  for _ in range(n_factors)],
        'factor': [f + 1 for _ in tensor_dict['dims']['donors']
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
        (pl.col('p') < 0.01)
    ).get_column('gene').to_list()
    for ct in de.table['cell_type'].unique()
}

print(json.dumps({
    ct: len(genes) for ct, genes in de_genes.items()}, indent=2))
print(json.dumps({
    f'{ct1}-{ct2}': round(
        100 * len(set(de_genes[ct1]) & set(de_genes[ct2])) /
        len(set(de_genes[ct1]) | set(de_genes[ct2])), 1)
    for ct1 in de_genes for ct2 in de_genes if ct1 < ct2}, indent=2))

lcpm = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .filter_obs((pl.col('cogdx').is_between(2, 5) & 
                 pl.col('cogdx').is_not_null()))\
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

tensor_dict = run_tucker_ica(tensor_dict, ranks=[6, 30, 7])

rosmap_codes = pl.read_csv(
    f'{data_dir}/{study_name}/rosmap_codes_edits.csv')\
    .filter(pl.col.priority)\
    .with_columns(pl.col.name.str.replace_all(' - OLD', ''))\
    .select(['code', 'name', 'category'])
code_to_name = dict(rosmap_codes.select(['code', 'name']).iter_rows())

meta = lcpm.obs[next(iter(lcpm.keys()))]\
    .filter(pl.col.ID.is_in(tensor_dict['dims']['donors']))\
    .select(['ID'] + rosmap_codes['code'].to_list())\
    .rename(code_to_name)\
    .with_columns([
        pl.col('CERAD score').reverse().alias('CERAD score'),
        pl.col('NIA-Reagan diagnosis of AD').reverse()
        .alias('NIA-Reagan diagnosis of AD')])
meta = meta.select(['ID'] + sorted(set(meta.columns) - {'ID'}))

r('options(device = png)')

tensor_dict = get_meta_associations(
    tensor_dict, meta, max_na_pct=0.4, adjust_pvals=True)

plot_donor_loadings_heatmap(
    tensor_dict, 
    f'{work_dir}/figures/donor_loadings_{study_name}.png')

plot_scores_by_meta(
    tensor_dict,
    f'{work_dir}/figures/scores_by_var.png',
    meta,
    factors=[2],
    meta_var='TDP-43 stage')

plot_gene_loadings_heatmap(
    tensor_dict, 
    f'{work_dir}/figures/gene_loadings.png',
    n_genes_per_factor=5)

plot_donor_sig_genes(
    tensor_dict, 
    f'{work_dir}/figures/donor_sig_genes.png',
    n_genes_per_ct=15,
    selection='positive')

factor_select = 2
plot_gsea_one_factor(
    tensor_dict,  
    f'{work_dir}/figures/gsea_hmap_{study_name}_factor_{factor_select}.png',
    factor_select=factor_select,
    thresh=0.1,
    signed=True, min_gs_size=15, max_gs_size=100)






study_name = 'SEAAD'
level = 'broad'

lcpm_new = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .with_columns_obs(
        pl.col('Highest Lewy Body Disease').cast(pl.String)
        .replace_strict({
            'Not Identified (olfactory bulb not assessed)': 0,
            'Not Identified (olfactory bulb assessed)': 0,
            'Olfactory bulb only': 1,
            'Amygdala-predominant': 2,
            'Brainstem-predominant': 3,
            'Limbic (Transitional)': 4,
            'Neocortical (Diffuse)': 5}, 
            default=None), 
        pl.when(pl.col('Fresh Brain Weight').eq('Unavailable')).then(None)
            .otherwise(pl.col('Fresh Brain Weight'))
            .cast(pl.Float64)
            .fill_null(strategy='mean')
            .alias('Fresh Brain Weight'),
        pl.when(pl.col('Brain pH').eq('Reference')).then(None)
            .otherwise(pl.col('Brain pH'))
            .cast(pl.Float64)
            .fill_null(strategy='mean')
            .alias('Brain pH'))\
    .qc(group_column=None,
        custom_filter=pl.col('Severely Affected Donor').eq(0))\
    .log_CPM()\
    .regress_out('~ PMI + RIN + `Brain pH` + `Fresh Brain Weight`',
                 library_size_as_covariate=True,
                 num_cells_as_covariate=True)

tensor_dict_new = project_tucker_ica(tensor_dict, lcpm_new, de_genes)

codes = pl.read_csv(f'{data_dir}/{study_name}/codes.csv')\
    .filter(pl.col('priority'))
meta = lcpm_new.obs[next(iter(lcpm_new.keys()))]\
    .select(['ID'] + codes['name'].to_list())

tensor_dict_new = get_meta_associations(
    tensor_dict_new, meta, max_na_pct=0.4, adjust_pvals=False)

plot_donor_loadings_heatmap(
    tensor_dict_new, 
    f'{work_dir}/figures/donor_loadings_proj_{study_name}.png')




# Get neuropath correlations
donor_ids = tensor_dict_new['dims']['donors']
neuropath = pl.read_csv(f'{data_dir}/SEAAD/'
    'sea-ad_all_mtg_quant_neuropath_bydonorid_081122.csv')\
    .rename({'Donor ID': 'ID'})\
    .filter(pl.col('ID').is_in(donor_ids))\
    .join(pl.DataFrame({'ID': donor_ids, '__order': range(len(donor_ids))}),
          on='ID')\
    .sort('__order')\
    .drop('__order')
assert neuropath['ID'].to_list() == donor_ids

donor_scores = tensor_dict_new['decomposition']['factors']['donors']
to_r(donor_scores, 'donor_scores', rownames=donor_ids)
to_r(neuropath.drop('ID'), 'meta')
to_r(f'{work_dir}/figures/neuropath_proj_{study_name}.png', 'filename')

r('''
suppressPackageStartupMessages({
    library(ComplexHeatmap)
    library(circlize)
    library(grid)
})

result = meta_assoc(donor_scores, meta, max_na_pct, adjust_pvals=FALSE)
rsq_mat = result$rsq 
pvals_mat = result$pvals  

top_vars = names(sort(abs(rowMeans(rsq_mat)), decreasing=TRUE)[1:90])
result_subset = rsq_mat[top_vars,]
pvals_subset = pvals_mat[top_vars,]

rsq_lim = quantile(abs(as.matrix(result_subset)), 0.99)
rsq_colors = colorRamp2(
    c(-rsq_lim, 0, rsq_lim),
    c('#7B3294', 'white', '#008837')
)

row_fontfaces = ifelse(
    grepl('aSyn|pTDP|pTAU|6e10|AT8', rownames(result_subset), ignore.case=TRUE),
    'bold', 'plain'
)

ht = Heatmap(
    result_subset,
    name = 'rsq',
    col = rsq_colors,
    cluster_rows = TRUE,
    cluster_columns = FALSE,
    show_row_dend = FALSE,  
    row_names_gp = gpar(fontsize = 8, fontface = row_fontfaces),
    column_names_gp = gpar(fontsize = 8),
    row_title = 'Neuropathology',  
    row_title_gp = gpar(fontsize = 10),
    column_labels = paste0('Factor ', 1:ncol(result_subset)), 
    row_names_max_width = unit(15, 'cm'),
    width = unit(8, 'cm'),
    height = unit(20, 'cm'),
    cell_fun = function(j, i, x, y, width, height, fill) {
        v = pvals_subset[i, j]
        if(!is.null(v)) {
            if(v < 0.001) {
                grid.text("***", x, y, gp = gpar(fontsize = 8, col = "white",
                                                fontface='bold'))
            } else if(v < 0.01) {
                grid.text("**", x, y, gp = gpar(fontsize = 8, col = "white",
                                                fontface='bold'))
            } else if(v < 0.05) {
                grid.text("*", x, y, gp = gpar(fontsize = 8, col = "white",
                                               fontface='bold'))
            }
        }
    }
)

png(filename, width=12, height=10, units='in', res=300)
draw(ht, padding = unit(c(2, 2, 2, 2), 'mm'),
     heatmap_legend_side = "left",
     annotation_legend_side = "left")
dev.off()

for (i in 1:ncol(rsq_mat)) {
    cat(sprintf("\nTop correlations for Factor %d:\n", i))
    top5 = sort(rsq_mat[,i], decreasing=TRUE)[1:5]
    print(round(top5, 3))
}
''')


plot_scores_by_meta(
    tensor_dict_new,
    f'{work_dir}/figures/scores_by_var.png',
    meta,
    factors=[2],
    meta_var='LATE')

plot_scores_by_meta(
    tensor_dict_new,
    f'{work_dir}/figures/scores_by_neuropath.png',
    neuropath,
    factors=[2],
    meta_var='average aSyn positive cell area_Grey matter')




from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.optimize import linear_sum_assignment

def find_optimal_ordering(scores_corr, loadings_corr):
   combined_cost = 1 - (np.abs(scores_corr) + np.abs(loadings_corr)) / 2
   row_ind, col_ind = linear_sum_assignment(combined_cost)
   signs = np.sign(np.diag(scores_corr[row_ind][:, col_ind]))
   return row_ind, col_ind, signs

genes = sorted(set().union(*[set(genes) for genes in de_genes.values()]))
scores, loadings, donors = {}, {}, {}

for study in ['Green', 'Mathys']:
   lcpm = Pseudobulk(f'{data_dir}/{study}/pseudobulk/broad')\
       .filter_obs((pl.col('cogdx').is_between(2, 5) & 
                    pl.col('cogdx').is_not_null()))\
       .qc(group_column=None, verbose=False)\
       .log_CPM()\
       .regress_out('~ pmi')
  
   tensor = prepare_tensor_dict(lcpm, genes=genes)
   tensor = pad_tensor(tensor, de_genes)
   tensor = normalize_tensor(tensor, min_shift=False)
   tucker = run_tucker_ica(tensor, ranks=[6, 30, 7])
  
   scores[study] = tucker['decomposition']['factors']['donors']
   loadings[study] = tl.unfold(tucker['decomposition']['loadings']['tensor'], 0)
   donors[study] = tucker['dims']['donors']

common_donors = sorted(list(set(donors['Green']) & set(donors['Mathys'])))
idx_green = [donors['Green'].index(d) for d in common_donors]
idx_mathys = [donors['Mathys'].index(d) for d in common_donors]
scores['Green'] = scores['Green'][idx_green]
scores['Mathys'] = scores['Mathys'][idx_mathys]

scores_corr = np.corrcoef(scores['Green'].T, scores['Mathys'].T)[:6, 6:]
loadings_corr = np.corrcoef(loadings['Green'], loadings['Mathys'])[:6, 6:]

row_ind, col_ind, signs = find_optimal_ordering(scores_corr, loadings_corr)
labels = [f'Factor {i+1}' for i in range(6)]
reordered_mats = [mat[row_ind][:, col_ind] * signs[:, np.newaxis] 
                  for mat in [scores_corr, loadings_corr]]

fig = plt.figure(figsize=(12, 16))
gs = GridSpec(2, 1, figure=fig)
cmap = mcolors.LinearSegmentedColormap.from_list('custom', 
    [(0.0, '#0000ff'), (0.4, '#ffffff'), (0.6, '#ffffff'), (1.0, '#ff0000')])

for idx, (mat, title) in enumerate(zip(reordered_mats, ['Donor Scores', 'Loadings'])):
    ax = fig.add_subplot(gs[idx])
    sns.heatmap(mat, ax=ax, annot=True, fmt='.2f', cmap=cmap, center=0,
                vmin=-1, vmax=1, square=True,
                xticklabels=[labels[i] for i in col_ind],
                yticklabels=[labels[i] for i in row_ind])
    ax.set_title(f'{title} De Novo Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mathys', fontsize=14, fontweight='bold')
    ax.set_ylabel('Green', fontsize=14, fontweight='bold')

savefig(f'{work_dir}/figures/factor_correlations_combined.png')



study_name = 'Mathys'
level = 'broad'

lcpm_new = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .filter_obs((pl.col('cogdx').is_between(2, 5) & 
                 pl.col('cogdx').is_not_null()))\
    .qc(group_column=None, verbose=False)\
    .log_CPM()\
    .regress_out('~ pmi',
                 library_size_as_covariate=True,
                 num_cells_as_covariate=True) 

tensor_dict_new = project_tucker_ica(tensor_dict, lcpm_new, de_genes)

meta = lcpm_new.obs[next(iter(lcpm_new.keys()))]\
    .filter(pl.col.ID.is_in(tensor_dict_new['dims']['donors']))\
    .select(['ID'] + rosmap_codes['code'].to_list())\
    .rename(code_to_name)\
    .with_columns([
        pl.col('CERAD score').reverse().alias('CERAD score'),
        pl.col('NIA-Reagan diagnosis of AD').reverse()
        .alias('NIA-Reagan diagnosis of AD')])
meta = meta.select(['ID'] + sorted(set(meta.columns) - {'ID'}))

tensor_dict_new = get_meta_associations(tensor_dict_new, meta, max_na_pct=0.4)

plot_donor_loadings_heatmap(
    tensor_dict_new, 
    f'{work_dir}/figures/donor_loadings_proj_{study_name}.png')


