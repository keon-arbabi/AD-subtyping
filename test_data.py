import sys
import json
from pathlib import Path
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk, set_num_threads
from utils_keon import debug, print_df, savefig
from ryp import r, to_r, to_py

set_num_threads(-1)
debug(third_party=True)

data_dir = f'{Path.home()}/projects/def-wainberg/single-cell'
work_dir = f'{Path.home()}/projects/def-wainberg/karbabi/AD-subtyping'

################################################################################

study_name = 'Green'
level = 'fine'

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

pb = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .qc(group_column='pmAD',
        custom_filter=pl.col('pmAD').is_not_null(),
        min_cells=10,
        min_samples=2,
        max_standard_deviations=2,
        min_nonzero_fraction=0.8,
        verbose=False)    

de = pb\
    .DE(formula='~ pmAD + age_death + sex + pmi + apoe4_dosage',
        coefficient='pmAD',
        group='pmAD',
        verbose=False)

de_genes = {
    ct: de.table.filter(
        (pl.col('cell_type') == ct) & 
        (pl.col('FDR') < 0.10)) # fdr_thresholds[ct]
        .get_column('gene').to_list()
    for ct in fdr_thresholds.keys()
}
print(json.dumps({ct: len(genes) for ct, genes in de_genes.items()}, indent=2))

pb = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .drop_cell_types('Sst Chodl')

shared_ids = sorted(set.intersection(
    *(set(obs['ID']) for obs in pb.iter_obs())))
print(len(shared_ids))

lcpm = pb\
    .log_CPM()\
    .filter_obs(pl.col.ID.is_in(shared_ids))\
    .regress_out('~ pmi', 
                 library_size_as_covariate=True,
                 num_cells_as_covariate=True) 

matrices, cell_types, genes = [], [], []
for cell_type, (X, obs, var) in lcpm.items():
    gene_mask = var['_index'].is_in(de_genes[cell_type])
    matrices.append(X.T[gene_mask])
    gene_select = var['_index'].filter(gene_mask).to_list()
    genes.extend(gene_select)
    cell_types.extend([cell_type] * len(gene_select))

from utils import inverse_normal_transform
A = np.vstack(matrices)
A = np.apply_along_axis(inverse_normal_transform, 1, A)
print(A.shape)
corr_matrix = np.corrcoef(A.T)

################################################################################

# Plot correlation matrix by data

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist
linkage_matrix = linkage(
    pdist(corr_matrix), method='ward', optimal_ordering=False)   

# Calculate total within-cluster sum of squares for different k and take elbow
max_k = 20
within_ss = []
k_range = range(1, max_k + 1)
for k in k_range:
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')
    wss = 0
    for i in range(1, k + 1):
        mask = clusters == i
        if mask.any():
            cluster_points = corr_matrix[mask]
            centroid = cluster_points.mean(axis=0)
            wss += np.sum((cluster_points - centroid) ** 2)
    within_ss.append(wss)

from kneed import KneeLocator
kneedle = KneeLocator(k_range, within_ss, curve='convex', direction='decreasing')
n_clusters = kneedle.elbow

optimal_order = leaves_list(linkage_matrix)
corr_matrix = corr_matrix[optimal_order][:, optimal_order]

clusters_orig = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
clusters = clusters_orig[optimal_order]

corr_min = np.min(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])
corr_max = np.max(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])

anno = lcpm.obs[next(iter(lcpm.keys()))]['pmAD'].to_numpy()
anno = anno[optimal_order]
anno_colors = ['#800000' if x else '#155f83' for x in anno]
anno_lut = {'AD': '#800000', 'Control': '#155f83'}

plt.figure(figsize=(13, 10))
g = sns.clustermap(
    corr_matrix,
    row_colors=anno_colors,
    col_colors=anno_colors,
    cmap='RdBu_r',
    center=0,
    vmin=corr_min,
    vmax=corr_max,
    xticklabels=False,
    yticklabels=False,
    row_cluster=False,
    col_cluster=False,
    dendrogram_ratio=0.1,
    colors_ratio=0.02,
    cbar_pos=(0.02, 0.8, 0.03, 0.2))

for cluster_id in range(1, n_clusters + 1):
    mask = clusters == cluster_id
    indices = np.where(mask)[0]
    if len(indices) > 0:
        start, end = indices[0], indices[-1]
        g.ax_heatmap.add_patch(plt.Rectangle(
            (start, start), end - start, end - start,
            fill=False, color='black', linewidth=1))
        center = start + (end - start) / 2
        g.ax_heatmap.text(center, center, str(cluster_id), 
                         ha='center', va='center', 
                         color='black', fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.7, 
                                 edgecolor='none', pad=1))
        
for label, color in anno_lut.items():
    g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
g.ax_col_dendrogram.legend(title='Condition', loc='upper center',
                          bbox_to_anchor=(0.5, -0.1))

plt.savefig(f'{work_dir}/figures/test-data/corrmap_all_samples_'
            f'{level}_data_{study_name}.png',
            dpi=300, bbox_inches='tight')
plt.close()


# Plot cluster proportions by category

rosmap_codes = pl.read_csv(
    f'{data_dir}/Green/rosmap_codes_edits.csv')\
    .with_columns(pl.col.name.str.replace_all(' - OLD', ''))\
    .filter(pl.col.code.is_in([
        'pmAD', 'cogdx', 'niareagansc', 'braaksc', 'ceradsc']))\
    .select(['code', 'name', 'category'])
code_to_name = dict(rosmap_codes.select(['code', 'name']).iter_rows())

plot_data = lcpm.obs[next(iter(lcpm.keys()))]\
    .select(['ID'] + rosmap_codes['code'].to_list())\
    .rename(code_to_name)\
    .with_columns([
        pl.col('CERAD score').reverse().alias('CERAD score'),
        pl.col('NIA-Reagan diagnosis of AD').reverse()
        .alias('NIA-Reagan diagnosis of AD')])\
    .with_columns(pl.Series('cluster', clusters_orig))

to_r(plot_data, 'plot_data')
to_r(n_clusters, 'n_clusters')
to_r(work_dir, 'work_dir')
to_r(level, 'level')
to_r(study_name, 'study_name')

r('''
library(ggplot2)
library(dplyr)
library(viridis)

plot_data_long = plot_data %>%
    tidyr::pivot_longer(-c('ID', 'cluster'), 
                names_to='variable', values_to='value') %>%
    mutate(var_val = paste(variable, value, sep='_'))

ggplot(plot_data_long, aes(x=cluster, fill=var_val)) +
    geom_bar(position='fill', width=0.9) +
    facet_wrap(~variable, nrow=1, labeller = label_wrap_gen(width=20)) +
    scale_y_continuous(expand=c(0,0)) +
    scale_x_continuous(breaks=1:n_clusters) +
    labs(x='Cluster', y='Proportion') +
    theme_void() +
    theme(
        legend.position='none',
        strip.text = element_text(size=8)
    ) +
    scale_fill_manual(values = c(
        'Pathological AD_0' = '#155f83',
        'Pathological AD_1' = '#800000',
        setNames(viridis(6), paste0('Final consensus cognitive diagnosis_', 1:6)),
        setNames(viridis(4), paste0('NIA-Reagan diagnosis of AD_', 1:4)),
        setNames(viridis(7), paste0('Braak stage_', 0:6)),
        setNames(viridis(4), paste0('CERAD score_', 1:4))
    ))
ggsave(paste0(work_dir, '/figures/test-data/barplots_all_samples_',
              level, '_', study_name, '.png'),
       width=6.5, height=0.9)
''')


# Plot correlation matrix by category   

for var in ['pmAD', 'cogdx', 'niareagansc', 'braaksc', 'ceradsc']:
    obs = lcpm.obs[next(iter(lcpm.keys()))]
    indices = obs\
        .with_row_count('idx')\
        .sort(var)\
        .get_column('idx')\
        .to_numpy()

    categories = obs[var].unique().sort()
    color_palette = ['#FFA319', '#BC6622', '#8F3931', '#8BA045', 
                    '#5E593F', '#155F83', '#350E20']
    cat_colors = dict(zip(categories, 
                        color_palette[:len(categories)] + 
                        color_palette[:(len(categories)-len(color_palette))]))

    anno = obs.sort(var)[var].to_numpy()
    anno_colors = [cat_colors[x] for x in anno]
    corr_matrix_by_cat = corr_matrix[indices][:, indices]

    boundaries = [0] + obs\
        .group_by(var, maintain_order=True)\
        .agg(pl.count())\
        .get_column('count')\
        .cum_sum()\
        .to_list()

    plt.figure(figsize=(13, 10))
    g = sns.clustermap(
        corr_matrix_by_cat,
        row_colors=anno_colors,
        col_colors=anno_colors,
        cmap='RdBu_r',
        center=0,
        vmin=corr_min,
        vmax=corr_max,
        xticklabels=False,
        yticklabels=False,
        row_cluster=False,
        col_cluster=False,
        dendrogram_ratio=0.1,
        colors_ratio=0.02,
        cbar_pos=(0.02, 0.8, 0.03, 0.2))

    for i in range(len(categories)):
        start, end = boundaries[i], boundaries[i+1]
        g.ax_heatmap.add_patch(plt.Rectangle(
            (start, start), end - start, end - start,
            fill=False, color='black', linewidth=1))

    for cat, color in cat_colors.items():
        g.ax_col_dendrogram.bar(0, 0, color=color, label=cat, linewidth=0)
    g.ax_col_dendrogram.legend(title=var, loc='center left', 
                            bbox_to_anchor=(1, 0.5))

    plt.savefig(f'{work_dir}/figures/test-data/case-control/'
                f'sample_corr_{level}_{gene_set}_genes_cat_{var}_{study_name}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


# Plot cluster proportions by category

plot_data = lcpm.obs[next(iter(lcpm.keys()))]\
    .with_columns(pl.Series('cluster', clusters_orig))

for var in ['pmAD', 'cogdx', 'niareagansc', 'braaksc', 'ceradsc']:
    to_r(plot_data, 'plot_data'); to_r(var, 'var_name')
    to_r(plot_data, 'plot_data'); to_r(n_clusters, 'n_clusters')
    to_r(work_dir, 'work_dir'); to_r(level, 'level')
    to_r(study_name, 'study_name'); to_r(gene_set, 'gene_set')

    r('''
    library(ggplot2)
    library(dplyr)
    library(ggsci)

    plot_data %>%
        group_by(cluster, .data[[var_name]]) %>%
        summarise(count = n(), .groups='drop') %>%
        group_by(cluster) %>%
        mutate(proportion = count/sum(count)) %>%
        ggplot(aes(x=cluster, y=proportion, fill=factor(.data[[var_name]]))) +
        geom_bar(stat='identity', width=0.9) +
        scale_y_continuous(expand=expansion(mult=c(0, 0.05))) +
        scale_x_continuous(breaks=1:n_clusters) +
        labs(x='Cluster', y='Proportion', fill=var_name) +
        theme_classic() +
        scale_fill_uchicago() +
        theme(axis.text.x=element_text(size=10),
            axis.text.y=element_text(size=10),
            legend.position='right')

    ggsave(paste0(work_dir, "/figures/test-data/case-control/cluster_proportions_",
        level, "_", gene_set, "_genes_", var_name, "_", study_name, ".png"), 
        width=4, height=6)
    ''')













study_name = 'SEAAD'
level = 'broad'

pb = Pseudobulk(f'{data_dir}/{study_name}/pseudobulk/{level}')\
    .qc(group_column=None,
        custom_filter=pl.col('CPS').is_not_null(),
        min_cells=10,
        min_samples=2,
        max_standard_deviations=None,
        min_nonzero_fraction=0.8,
        verbose=False)    

de = pb\
    .DE(formula='~ CPS + Age_death + Sex + PMI + APOE4_Dosage',
        coefficient=0,
        group=False,
        verbose=True)

de_genes = {
    ct: de.table.filter(
        (pl.col('cell_type') == ct) & 
        (pl.col('FDR') < 0.01)) # fdr_thresholds[ct]
        .get_column('gene').to_list()
    for ct in pb.keys()
}
print(json.dumps({ct: len(genes) for ct, genes in de_genes.items()}, indent=2))

shared_ids = sorted(set.intersection(
    *(set(obs['ID']) for obs in pb.iter_obs())))
print(len(shared_ids))

lcpm = pb\
    .log_CPM()\
    .filter_obs(pl.col.ID.is_in(shared_ids))\
    .regress_out('~ PMI', 
                 library_size_as_covariate=True,
                 num_cells_as_covariate=True) 

matrices, cell_types, genes = [], [], []
for cell_type, (X, obs, var) in lcpm.items():
    gene_mask = var['_index'].is_in(de_genes[cell_type])
    matrices.append(X.T[gene_mask])
    gene_select = var['_index'].filter(gene_mask).to_list()
    genes.extend(gene_select)
    cell_types.extend([cell_type] * len(gene_select))

from utils import inverse_normal_transform
A = np.vstack(matrices)
A = np.apply_along_axis(inverse_normal_transform, 1, A)
print(A.shape)
corr_matrix = np.corrcoef(A.T)

# Plot correlation matrix by data

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist
linkage_matrix = linkage(
    pdist(corr_matrix), method='ward', optimal_ordering=True)   

# Calculate total within-cluster sum of squares for different k and take elbow
max_k = 20
within_ss = []
k_range = range(1, max_k + 1)
for k in k_range:
    clusters = fcluster(linkage_matrix, k, criterion='maxclust')
    wss = 0
    for i in range(1, k + 1):
        mask = clusters == i
        if mask.any():
            cluster_points = corr_matrix[mask]
            centroid = cluster_points.mean(axis=0)
            wss += np.sum((cluster_points - centroid) ** 2)
    within_ss.append(wss)

from kneed import KneeLocator
kneedle = KneeLocator(k_range, within_ss, curve='convex', direction='decreasing')
n_clusters = kneedle.elbow

optimal_order = leaves_list(linkage_matrix)
corr_matrix = corr_matrix[optimal_order][:, optimal_order]

clusters_orig = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
clusters = clusters_orig[optimal_order]

corr_min = np.min(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])
corr_max = np.max(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])

anno = lcpm.obs[next(iter(lcpm.keys()))]['Overall AD neuropathological Change']
anno = anno[optimal_order]
colors = {3: '#800000', 2: '#BC6622', 1: '#FFA319', 0: '#155f83'}
anno_colors = [colors[x] for x in anno]
anno_lut = {'High': '#800000', 'Int': '#BC6622', 'Low': '#FFA319', 'None': '#155f83'}

plt.figure(figsize=(13, 10))
g = sns.clustermap(
    corr_matrix,
    row_colors=anno_colors,
    col_colors=anno_colors,
    cmap='RdBu_r',
    center=0,
    vmin=corr_min,
    vmax=corr_max,
    xticklabels=False,
    yticklabels=False,
    row_cluster=False,
    col_cluster=False,
    dendrogram_ratio=0.1,
    colors_ratio=0.02,
    cbar_pos=(0.02, 0.8, 0.03, 0.2))

for cluster_id in range(1, n_clusters + 1):
    mask = clusters == cluster_id
    indices = np.where(mask)[0]
    if len(indices) > 0:
        start, end = indices[0], indices[-1]
        g.ax_heatmap.add_patch(plt.Rectangle(
            (start, start), end - start, end - start,
            fill=False, color='black', linewidth=1))
        center = start + (end - start) / 2
        g.ax_heatmap.text(center, center, str(cluster_id), 
                         ha='center', va='center', 
                         color='black', fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.7, 
                                 edgecolor='none', pad=1))
        
for label, color in anno_lut.items():
    g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
g.ax_col_dendrogram.legend(title='Condition', loc='center left')

plt.savefig(f'{work_dir}/figures/test-data/corrmap_all_samples_'
            f'{level}_data_{study_name}.png',
            dpi=300, bbox_inches='tight')
plt.close()








