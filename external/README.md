# External resources

| File | Used by | Content |
|---|---|---|
| `schaeffer_subcortical_smith_templates.csv` | network analyses (`visualization/base.py`, `results_article.ipynb`) | 18 network templates (rows) × 481 ROIs (columns: 400 Schaefer-2018 parcels then 81 AAL3 subcortical regions, in the order of the time-series files) |
| `schaefer400_7networks_centroids.csv` | `phase_protocol.py` (anatomical ROI sets) | label and centroid (MNI, mm) of the 400 Schaefer-2018 parcels (7 networks), in the order of the first 400 ROIs |

`schaefer400_7networks_centroids.csv` is derived from
`Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv` of the CBIG repository
(https://github.com/ThomasYeoLab/CBIG, `stable_projects/brain_parcellation/Schaefer2018_LocalGlobal`),
Copyright (c) 2016 Computational Brain Imaging Group (CBIG), MIT License. Reference: Schaefer A,
et al. Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional
Connectivity MRI. Cereb Cortex 2018;28:3095-3114.
