"""Write the network templates as brain maps for BrainNet Viewer (right part of Figure 4).

For each network of external/schaeffer_subcortical_smith_templates.csv (18 rows x 481 ROIs),
the script paints the template weight of every ROI into the voxels of that ROI:

    <out>/<network>.nii       volume for BrainNet Viewer (surface + volume mapping)
    <out>/<network>.node      one node per ROI at its centroid, size = weight (subcortical networks)
    <out>/<network>.png       nilearn preview, only to check the map before rendering it
    <out>/cerebellum.nv       cerebellum surface (BrainNet's cortical meshes have none), same for all networks

The figure shows the templates, not SHAP values: the maps do not depend on the model and only
the list of networks to draw comes from the analyses (the networks of the curves of Figure 4).

ROI order (the order of the columns of the templates and of the time series):
    1-400   Schaefer-2018, 400 parcels, 7 networks, 2 mm: voxel value i = column i
    401-481 AAL3 regions named in the columns (prefix "r" removed). AAL3 voxel values are NOT
            consecutive (e.g. 35, 36, 81, 82 are unused), so each region is painted with the index
            read from AAL3v1_1mm.xml, never with its position in the list.

Usage:
    python tools/make_network_maps.py --aal-dir /path/to/AAL3 --networks DAN anteriorDMN
    python tools/make_network_maps.py --aal-dir /path/to/AAL3            # the 18 networks

--aal-dir must contain AAL3v1.nii (2 mm, 91 x 109 x 91) and AAL3v1_1mm.xml
(https://www.gin.cnrs.fr/en/tools/aal/). The Schaefer atlas is downloaded by nilearn.
"""
import argparse
import os
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES = os.path.join(REPO, "external", "schaeffer_subcortical_smith_templates.csv")


def load_atlases(aal_dir):
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
    sch_img = nib.load(schaefer.maps)
    aal_img = nib.load(os.path.join(aal_dir, "AAL3v1.nii"))
    if aal_img.shape != sch_img.shape or not np.allclose(aal_img.affine, sch_img.affine):
        aal_img = image.resample_to_img(aal_img, sch_img, interpolation="nearest")
    root = ET.parse(os.path.join(aal_dir, "AAL3v1_1mm.xml")).getroot()
    aal_index = {lab.find("name").text: int(lab.find("index").text) for lab in root.iter("label")}
    return sch_img, np.asarray(sch_img.dataobj).astype(int), np.asarray(aal_img.dataobj).astype(int), aal_index


def roi_masks(roi_names, sch, aal, aal_index):
    """Boolean mask of each of the 481 ROIs, in the column order of the templates."""
    masks = []
    for i, name in enumerate(roi_names):
        if i < 400:
            if name != f"schaefer{i + 1:03d}":
                raise ValueError(f"Column {i + 1} is {name}, expected schaefer{i + 1:03d}")
            mask = sch == i + 1
        else:
            label = name[1:] if name.startswith("r") else name
            if label not in aal_index:
                raise KeyError(f"{label} is not an AAL3 region of AAL3v1_1mm.xml")
            mask = aal == aal_index[label]
        if not mask.any():
            raise ValueError(f"{name} has no voxel in the atlas")
        masks.append(mask)
    return masks


def write_cerebellum_mesh(aal, aal_index, affine, path, smoothing=1.0):
    """Surface of the cerebellum (AAL3 Cerebellum_* and Vermis_* regions) in BrainNet .nv format.

    BrainNet's cortical meshes do not include the cerebellum: render this mesh with the same
    <network>.nii and the same options file, then place it next to the cortical views.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    ids = [i for name, i in aal_index.items() if name.startswith(("Cerebellum_", "Vermis_"))]
    mask = np.isin(aal, ids).astype(float)
    verts, faces, _, _ = marching_cubes(gaussian_filter(mask, smoothing), level=0.5)
    verts = nib.affines.apply_affine(affine, verts)  # voxel -> MNI mm
    with open(path, "w") as f:
        f.write(f"{len(verts)}\n")
        f.writelines(f"{x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in verts)
        f.write(f"{len(faces)}\n")
        f.writelines(f"{a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)  # .nv faces are 1-based
    return verts, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aal-dir", required=True, help="folder with AAL3v1.nii and AAL3v1_1mm.xml")
    parser.add_argument("--networks", nargs="*", help="networks to write (default: the 18)")
    parser.add_argument("--positive", action="store_true",
                        help="keep only the positive weights (negative ones set to 0)")
    parser.add_argument("--out", default=os.path.join(REPO, "results", "network_maps"))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    templates = pd.read_csv(TEMPLATES).set_index("network_names")
    networks = args.networks or list(templates.index)
    unknown = [n for n in networks if n not in templates.index]
    if unknown:
        raise SystemExit(f"Unknown networks {unknown}; available: {list(templates.index)}")

    ref, sch, aal, aal_index = load_atlases(args.aal_dir)
    roi_names = list(templates.columns)
    masks = roi_masks(roi_names, sch, aal, aal_index)
    # AAL3 voxels inside a Schaefer parcel: the parcellations of the time series were extracted
    # separately, so a voxel may belong to both; the subcortical value is written last.
    overlap = (sch > 0) & np.isin(aal, [aal_index[n[1:]] for n in roi_names[400:]])
    vox = np.array(ref.header.get_zooms()[:3])
    centroids = [nib.affines.apply_affine(ref.affine, np.argwhere(m).mean(0)) for m in masks]

    os.makedirs(args.out, exist_ok=True)
    verts, faces = write_cerebellum_mesh(aal, aal_index, ref.affine, os.path.join(args.out, "cerebellum.nv"))
    print(f"Cerebellum mesh: {len(verts)} vertices, {len(faces)} faces -> cerebellum.nv")
    print(f"{len(roi_names)} ROIs mapped; {int(overlap.sum())} voxels in both a Schaefer parcel "
          f"and an AAL3 region (AAL3 value kept); voxel size {vox.tolist()} mm")
    for net in networks:
        w = templates.loc[net].to_numpy(dtype=float)
        if args.positive:
            w = np.clip(w, 0, None)
        vol = np.zeros(ref.shape, dtype=np.float32)
        for m, v in zip(masks, w):
            vol[m] = v
        img = nib.Nifti1Image(vol, ref.affine)
        img.to_filename(os.path.join(args.out, f"{net}.nii"))

        # BrainNet Viewer node file: x y z colour size label (one ROI per line, label without spaces)
        scale = np.abs(w).max() or 1.0
        with open(os.path.join(args.out, f"{net}.node"), "w") as f:
            for (x, y, z), v, name in zip(centroids, w, roi_names):
                if v > 0 or not args.positive:
                    f.write(f"{x:.1f}\t{y:.1f}\t{z:.1f}\t{1 if v > 0 else 2}\t{abs(v) / scale * 5:.3f}\t{name}\n")

        if not args.no_preview:
            from nilearn import plotting
            pos = w[w > 0]
            thr = float(np.quantile(pos, 0.5)) if pos.size else None
            plotting.plot_stat_map(img, threshold=thr, display_mode="ortho", title=net,
                                   output_file=os.path.join(args.out, f"{net}.png"))
        top = np.argsort(w)[::-1][:5]
        print(f"{net}: max {w.max():.2f}, top ROIs {[roi_names[i] for i in top]}")
    print(f"Maps written to {args.out}")


if __name__ == "__main__":
    main()
