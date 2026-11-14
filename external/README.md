# External resources

| File | Used by | Source |
|---|---|---|
| `schaeffer_subcortical_smith_templates.csv` | network analyses (`visualization/base.py`, `networks.ipynb`, `results_article.ipynb`, `create_roi.ipynb`) | Provided. 18 network templates (rows) × 481 ROIs (columns: 400 Schaefer-2018 parcels then 81 AAL3 subcortical regions) |
| `AAL3v1.nii` | brain maps (`utils/get_aal.py`) | Download AAL3 from <https://www.gin.cnrs.fr/en/tools/aal/> (Rolls et al., 2020, *NeuroImage* 206:116189) |
| `AAL3v1_1mm.xml` | region labels (`utils/get_aal.py`) | Same archive |

`AAL3v1.nii` is the 2 mm version (91 × 109 × 91 voxels), on the same grid as the
Schaefer-2018 2 mm atlas that nilearn downloads automatically.
