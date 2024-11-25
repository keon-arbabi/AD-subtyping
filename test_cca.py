import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from cca_zoo.linear import GCCA

from cca_zoo.visualisation import (
    CovarianceHeatmapDisplay,
    CorrelationHeatmapDisplay,
    RepresentationScatterDisplay,
    WeightHeatmapDisplay,
    ExplainedVarianceDisplay,
    ExplainedCovarianceDisplay,
    SeparateRepresentationScatterDisplay,
    JointRepresentationScatterDisplay,
    SeparateJointRepresentationDisplay,
    PairRepresentationScatterDisplay,
)

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk, set_num_threads
from utils import debug, print_df, savefig
from ryp import r, to_r

set_num_threads(-1)
debug(third_party=True)

work_dir = 'projects/def-wainberg/karbabi/AD-subtyping'
data_dir = 'projects/def-wainberg/single-cell'

################################################################################


################################################################################

study = 'Green'
level = 'broad'
coefficient = 'pmAD'

pb = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
    .qc(group_column=coefficient, 
        min_nonzero_fraction=0.4,
        verbose=False)

cell_types = list(pb.keys())
samples = sorted(set.intersection(
    *[set(obs['ID']) for obs in pb.iter_obs()]))
genes = sorted(set.intersection(
    *[set(var['_index']) for var in pb.iter_var()]))

print(f'{len(samples)=}, {len(genes)=}')

pb = pb\
    .filter_obs(pl.col.ID.is_in(samples))\
    .filter_var(pl.col._index.is_in(genes))\
    .log_CPM()\
    .with_columns_obs(pl.col.num_cells.log10().alias('log10_num_cells'))\
    .regress_out('~ log10_num_cells + pmi') # TODO: log10_library_size 

rosmap_codes = pl.read_csv(
    'projects/def-wainberg/single-cell/Green/rosmap_codes_edits.csv')\
    .filter(pl.col.priority & pl.col.code.ne('race7'))\
    .with_columns(pl.col.name.str.replace_all(' - OLD', ''))\
    .select(['code', 'name', 'category'])

clinical_codes = [
    code for code in rosmap_codes['code'] 
    if pb.obs['Astrocyte'][code].null_count() / len(pb.obs) < 0.7]
clinical_names = rosmap_codes\
    .filter(pl.col.code.is_in(clinical_codes))['name']
clinical_data = pb.obs['Astrocyte']\
    .select(clinical_codes)\
    .with_columns([
        pl.col('ceradsc').reverse().alias('ceradsc'),
        pl.col('niareagansc').reverse().alias('niareagansc')])

for name, code in zip(clinical_names, clinical_codes): print(f'{name} [{code}]')

'''
Final consensus cognitive diagnosis [cogdx]
Age at death [age_death]
Sex [msex]
Cancer at baseline [cancer_bl]
Head injury loss of consciousness [headinjrloc_bl]
History of stroke [stroke_cum]
NIA-Reagan diagnosis of AD (dichotomous) [ad_reagan]
Amyloid [amyloid]
Arteriolosclerosis [arteriol_scler]
Braak stage [braaksc]
CERAD score [ceradsc]
Presence of one or more gross chronic infarcts [ci_num2_gct]
Presence of one or more chronic microinfarcts [ci_num2_mct]
Cerebral atherosclerosis [cvda_4gp2]
Global AD pathology burden [gpath]
Neurofibrillary tangles [nft]
NIA-Reagan diagnosis of AD [niareagansc]
Diffuse plaque burden [plaq_d]
Neuritic plaque burden [plaq_n]
Post-mortem interval [pmi]
PHF Tau Tangle Density [tangles]
ApoE dosage [apoe4_dosage]
Pathological AD [pmAD]
'''

gene_views = [StandardScaler().fit_transform(X) for X in pb.iter_X()]
clinical_view = StandardScaler().fit_transform(
    SimpleImputer(strategy='mean').fit_transform(clinical_data))

pca = PCA(n_components=50)  
reduced_gene_views = [pca.fit_transform(view) for view in gene_views]
reduced_views = [*reduced_gene_views, clinical_view]

for i, (view, cell_type) in enumerate(zip(reduced_gene_views, cell_types)):
    print(f'\n{cell_type}:')
    for pc in range(5):
        for j, name in enumerate(clinical_names):
            corr, _ = pearsonr(view[:,pc], clinical_view[:,j])
            if abs(corr) > 0.2:
                print(f'PC{pc+1} - {name}: {corr:.3f}')

'''
Astrocyte:
PC5 - Cerebral atherosclerosis: 0.225

Excitatory:
PC3 - Age at death: 0.218

Inhibitory:
PC3 - Age at death: 0.222
PC3 - Amyloid: 0.218
'''

view_weights = np.ones(len(reduced_views))
view_weights[-1] = np.sum(view_weights[:-1])
view_weights = [float(w) for w in view_weights]  

model = GCCA(
    latent_dimensions=10,
    c=0.1,
    view_weights=view_weights,
    random_state=42,
    eps=1e-6
)
model.fit(reduced_views)

# Explained variance across views
ExplainedVarianceDisplay.from_estimator(model, reduced_views).plot()
plt.savefig(f'{work_dir}/figures/explained_variance.png')

# Correlation structure between views
CorrelationHeatmapDisplay.from_estimator(model, reduced_views).plot()
plt.savefig(f'{work_dir}/figures/correlation_heatmap.png')

# Sample patterns in latent space
RepresentationScatterDisplay.from_estimator(
    model, reduced_views, labels=clinical_data['pmAD']
).plot()
plt.savefig(f'{work_dir}/figures/representation_scatter.png')



transformed_views = model.transform(reduced_views)
transformed_gene_views = transformed_views[:-1] 
transformed_clinical_view = transformed_views[-1]

n_comp = 10
clinical_comp_corr = []

for cell_type, view in zip(cell_types, transformed_views[:-1]):
    corr_matrix = np.array([
        [pearsonr(clinical_view[:,i], view[:,j])[0] 
         for j in range(n_comp)]
        for i in range(len(clinical_names))
    ])
    clinical_comp_corr.append(corr_matrix)

# Make colorbar symmetric around zero
max_abs_corr = max(abs(np.min(clinical_comp_corr)), abs(np.max(clinical_comp_corr)))
vmin, vmax = -max_abs_corr, max_abs_corr

fig = plt.figure(figsize=(40, 7))
gs = fig.add_gridspec(1, len(cell_types) + 1, 
                      width_ratios=[*[1]*len(cell_types), 0.05])

for i, corr in enumerate(clinical_comp_corr):
    ax = fig.add_subplot(gs[0, i])
    last_hm = sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        ax=ax,
        vmin=vmin, vmax=vmax,
        xticklabels=[f'Comp {j+1}' for j in range(n_comp)],
        yticklabels=[] if i > 0 else clinical_names,
        cbar=False
    )
    ax.set_title(cell_types[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

fig.colorbar(last_hm.get_children()[0], cax=fig.add_subplot(gs[0, -1]), 
             label='Correlation')
plt.suptitle('Clinical Variable Correlations with Cell Type Components', y=1.02)
plt.tight_layout()
plt.savefig(f'{work_dir}/figures/clinical_variable_correlations.png')