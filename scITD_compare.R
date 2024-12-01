suppressPackageStartupMessages({
    library(reticulate)
    library(SingleCellExperiment)
    library(scITD)
    library(Matrix)
    library(Rcpp)
    library(Rmisc)
    library(ComplexHeatmap)
    library(circlize)
    library(parallel)
    library(dplyr)
    library(ggplot2)
    library(ggpubr)
})

r_scripts = list.files(
    "projects/def-wainberg/karbabi/scITD/R", full.names = TRUE)
sapply(r_scripts, source, .GlobalEnv)

working_dir = "projects/def-wainberg/karbabi/AD-subtyping/"

pystr = "
import sys, polars as pl
from pathlib import Path
sys.path.append(f'{Path.home()}/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk
study = '{}'
pb = Pseudobulk(f'{Path.home()}/projects/def-wainberg/single-cell/{study}/' \
    'pseudobulk/broad')
pb = pb.qc(
    group_column=None,
    custom_filter=pl.col.dx_cogn.gt(0) & pl.col.dx_cogn.is_not_null(),
    min_samples=2,
    min_cells=10,
    max_standard_deviations=3,
    min_nonzero_fraction=0.4,
    verbose=False)
shared_ids = set.intersection(*(set(obs['ID']) for obs in pb.iter_obs()))
shared_genes = set.intersection(*(set(var['_index']) for var in pb.iter_var()))
pb = pb.filter_obs(pl.col.ID.is_in(shared_ids))
pb = pb.filter_var(pl.col._index.is_in(shared_genes))
meta = pb.obs['Astrocyte'].to_pandas()
genes = pb.var['Astrocyte']['_index'].to_list()
cell_types = list(pb.keys())
print(cell_types)"

study_data = list()
for (study in c('Mathys', 'Green')) {
    py_run_string(gsub("'\\{\\}'", paste0("'", study, "'"), pystr))

    study_data[[study]] = list(
        pb = py$pb,
        meta = py$meta,
        genes = py$genes,
        cell_types = py$cell_types
    )
    print(nrow(py$meta)); print(length(py$genes))
}

shared_donors = Reduce(
    intersect, 
    lapply(study_data, function(x) x$meta$ID)
)

containers = list()
for (study in c('Mathys', 'Green')) {
    cell_types = as.character(study_data[[study]]$cell_types)
    
    container = new.env()
    container$experiment_params = list(
        ctypes_use = cell_types,
        ncores = detectCores(),
        rand_seed = 1
    )    
    donor_idx = study_data[[study]]$meta$ID %in% shared_donors
    
    container$scMinimal_ctype = list()
    for (ct in cell_types) {
        container$scMinimal_ctype[[ct]] = new.env()
        pseudobulk = as.matrix(study_data[[study]]$pb[[ct]]$X[[1]])[donor_idx, ]
        colnames(pseudobulk) = study_data[[study]]$genes
        rownames(pseudobulk) = study_data[[study]]$meta$ID[donor_idx]
        container$scMinimal_ctype[[ct]]$pseudobulk = t(pseudobulk)
    }
    containers[[study]] = container
}

print(paste('Number of shared donors:', length(shared_donors)))

temp_containers = list()
for (study in c('Mathys', 'Green')) {
    container = containers[[study]]
    
    # edgeR trim-mean normalization,
    # counts are divided by library size times a normalization factor
    # then log transformed
    container = normalize_pseudobulk(
        container, method = 'trim', scale_factor = 1000)
    
    # caluclate normalized variance for each gene,
    # taking into account mean-variance trend
    container = get_normalized_variance(container)

    # select highly variable genes 
    # "norm_var" thresh is the number of top overdispersed genes per cell type
    temp_container = list2env(as.list(container), new.env()) 
    temp_container = get_ctype_vargenes(
        temp_container, method = 'norm_var', thresh = 500)
        
    containers[[study]] = container
    temp_containers[[study]] = temp_container
}

# Find intersection of variable genes between studies
shared_var_genes = Reduce(
    intersect,
    lapply(temp_containers, function(x) x$all_vargenes)
)

print(paste('Number of shared variable genes:', length(shared_var_genes)))

# Now reduce each container to shared variable genes
for (study in c('Mathys', 'Green')) {
    container = containers[[study]]
    
    # Reduce data to shared variable genes
    container$all_vargenes = shared_var_genes
    container = reduce_to_vargenes(container)
   
    # scale variance of genes across donors for each cell type:
    # center and scale genes to unit variance,
    # apply additional scaling based on each gene's normalized variance,
    # adjust scaling intensity with var_scale_power
    container = scale_variance(container, var_scale_power=0.5)

    # build tensor 
    container = stack_tensor(container)

    # run tensor decomposition 
    ranks = c(8, 30)
    container = run_tucker_ica(
        container, ranks,
        tucker_type = 'regular', rotation_type = 'hybrid')

    path_vars = read.csv(
        paste0(working_dir, '../../single-cell/Green/rosmap_codes_edits.csv')) %>%
        filter(priority == TRUE | code %in% c('pmi', 'ID')) %>%
        mutate(name = gsub(' - OLD', '', name)) %>%
        select(code, name) %>%
        arrange(name)

    meta = study_data[[study]]$meta %>%
        select(all_of(path_vars$code)) %>%
        rename(all_of(setNames(path_vars$code, path_vars$name))) %>%
        mutate(Sex = as.factor(Sex)) %>%
        tibble::column_to_rownames('Donor') %>%
        select(where(~n_distinct(.) > 1))
        
    stat_use = 'rsq'
    container = get_meta_associations(container, meta, stat_use=stat_use)

    # gene-factor associations using univariate linear models
    container = get_lm_pvals(container)

    # plot donor scores
    container = plot_donor_matrix(
        container, 
        meta,
        show_donor_ids = FALSE,
        add_meta_associations=stat_use)

    png(paste0(working_dir, "figures/scITD/compare_donor_matrix_", 
        stat_use, "_", study, ".png"), 
        height = 18, width = 8, units = "in", res = 300)
    print(draw(
        container$plots$donor_matrix,
        padding = unit(c(0.5, 0.5, 0.5, 8), "cm"),
        auto_adjust = FALSE))
    dev.off()

    # generate loading plots 
    container = get_all_lds_factor_plots(
        container, 
        use_sig_only=TRUE,
        nonsig_to_zero=TRUE,
        sig_thresh=0.05,
        display_genes=FALSE,
        gene_callouts = FALSE,
        callout_n_gene_per_ctype=5,
        show_var_explained = TRUE)

    png(paste0(working_dir, "figures/scITD/compare_gene_loadings_", study, ".png"), 
        height = 20, width = 15, units = "in", res = 300)
    print(render_multi_plots(container, data_type='loadings'))
    dev.off()

    # plot donor scores and metadata
    for (meta_var in names(meta)) {
        plot_scores_by_meta(container, meta, meta_var)

        png(paste0(working_dir, "figures/scITD/compare_scores_by_meta_", 
            study, "/", gsub("/", "_", gsub(" ", "_", meta_var)), ".png"),
            height = 4, width = 20, units = "in", res = 300)
        print(container$plots$indv_meta_scores_associations)
        dev.off()
    }

    # plot significant genes
    plots = list()
    for (factor_select in 1:ranks[1]) {
        container = plot_donor_sig_genes(
            container, factor_select, top_n_per_ctype=15)
        plots[[factor_select]] = grid::grid.grabExpr(
            ComplexHeatmap::draw(container$plots$donor_sig_genes[[factor_select]])
        )
    }
    png(paste0(working_dir, "figures/scITD/compare_donor_sig_genes_", study, ".png"),
        height = 15, width = 8 * ranks[1], units = "in", res = 300)
    print(ggarrange(plotlist = plots, ncol = ranks[1]))
    dev.off()
    
    containers[[study]] = container
}

png(paste0(working_dir, "figures/scITD/compare_decompositions.png"),
    height = 6, width = 12, units = "in", res = 300)
compare_decompositions(
    containers$Mathys$tucker_results, 
    containers$Green$tucker_results,
    decomp_names = c('Mathys', 'Green'))
dev.off()


# run GSEA on one factor
factor_select = 3
container$Mathys = run_gsea_one_factor(
    containers$Mathys, 
    factor_select=factor_select, 
    method="fgsea", 
    thresh=0.01, 
    min_gs_size=15,
    max_gs_size=150,
    db_use=c("GO"), signed=TRUE)

png(paste0(working_dir, "figures/scITD/compare_gsea_", factor_select, "_Mathys.png"), 
    height = 9, width = 6, units = "in", res = 300)
container$Mathys$plots$gsea[[as.character(factor_select)]]
dev.off()

factor_select = 2
container$Green = run_gsea_one_factor(
    containers$Green, 
    factor_select=factor_select, 
    method="fgsea", 
    thresh=0.01, 
    min_gs_size=15,
    max_gs_size=150,
    db_use=c("GO"), signed=TRUE)

png(paste0(working_dir, "figures/scITD/compare_gsea_", factor_select, "_Green.png"), 
    height = 9, width = 6, units = "in", res = 300)
container$Green$plots$gsea[[as.character(factor_select)]]
dev.off()


# # determine the ranks of the tensor
# container = determine_ranks_tucker(
#     container, 
#     max_ranks_test=c(20, 40),
#     shuffle_level="tensor", 
#     num_iter=50, 
#     norm_method="trim",
#     scale_factor=1000,
#     scale_var=TRUE,
#     var_scale_power=0.5)
# # plot
# container$plots$rank_determination_plot
# ggsave(paste0(working_dir, "figures/scITD/cases_rank_determination_plot.png"), 
#     width = 10, height = 8)