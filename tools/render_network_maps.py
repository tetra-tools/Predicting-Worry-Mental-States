"""Brain maps of the network templates: cortex (6 views) and cerebellum (2 views) per network.

Each network of ``external/schaeffer_subcortical_smith_templates.csv`` is painted on the
atlases (Schaefer-2018 400 parcels + AAL3, as ``tools/make_network_maps.py``: AAL3 regions by
their XML index), then drawn

* on the inflated fsaverage cortex (left and right: lateral, medial and ventral, the last one
  for the fusiform gyrus and the orbitofrontal cortex), sampled between the pial and white
  surfaces;
* on a cerebellum surface built from the AAL3 cerebellar and vermis regions (posterior and
  inferior views), each vertex taking the value of the nearest cerebellar voxel.

Only the positive weights are shown, from ``--threshold`` (default 2) to ``--vmax`` (default 8,
the same scale for every network), in a single-hue colour map ending in the network's colour of
``external/network_colors.csv``: the colour of its curves in ``figure_networks_top9.png``. The
subtitle gives the share of the positive weight in the cortex, the cerebellum and the other
subcortical regions (thalamus, basal ganglia, hippocampus, amygdala), which these views do not
show.

The maps show the templates, not SHAP values: they do not depend on the data.

Usage::

    python tools/render_network_maps.py --aal-dir /path/to/AAL3                  # the 18 networks
    python tools/render_network_maps.py --aal-dir /path/to/AAL3 --top9 R/plots/networks_top9.csv
    python tools/render_network_maps.py --aal-dir /path/to/AAL3 --networks SMA DAN

Outputs (``--out``, default results/network_maps/): ``<network>_brain.png`` per network,
``networks_all_brain.png`` (all networks drawn) and, with ``--top9``, ``networks_top9_brain.png``
(the networks of the top-9 figure, in the order of their first appearance, rank by rank).
``--mesh fsaverage`` uses the high-resolution surface (downloaded by nilearn on first use).
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from make_network_maps import TEMPLATES, load_atlases, roi_masks, write_cerebellum_mesh  # noqa: E402

COLORS = os.path.join(REPO, "external", "network_colors.csv")
CORTEX_VIEWS = [("left", "lateral"), ("left", "medial"), ("left", "ventral"),
                ("right", "ventral"), ("right", "medial"), ("right", "lateral")]   # ventral: fusiform, OFC
CEREBELLUM_VIEWS = [("posterior", "Cerebellum, posterior"), ("ventral", "Cerebellum, inferior")]


def single_hue(color):
    """Colour map from a light tint of the network's colour to the colour itself."""
    rgb = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list("network", [0.6 + 0.4 * rgb, rgb])


def weight_shares(w, roi_names):
    """Share of the positive weight in the cortex, the cerebellum and the other subcortical ROIs."""
    pos = np.clip(w, 0, None)
    total = pos.sum() or 1.0
    cortex = pos[:400].sum()
    cereb = sum(p for p, n in zip(pos[400:], roi_names[400:]) if "Cerebell" in n or "Vermis" in n)
    return cortex / total, cereb / total, 1 - (cortex + cereb) / total


def render(net, label, color, img, cereb, fsavg, args, shares, out_path):
    from nilearn import plotting, surface

    cmap = single_hue(color)
    n_cortex = len(CORTEX_VIEWS)
    fig = plt.figure(figsize=(3.3 * n_cortex + 5, 3.9))
    widths = [1] * n_cortex + [0.8, 0.8]
    grid = fig.add_gridspec(1, n_cortex + 2, width_ratios=widths, wspace=0.0, left=0.01, right=0.95, top=0.80, bottom=0.02)
    for k, (hemi, view) in enumerate(CORTEX_VIEWS):
        try:      # parcel values kept as they are (no blending of neighbouring parcels)
            texture = surface.vol_to_surf(img, fsavg[f"pial_{hemi}"], inner_mesh=fsavg[f"white_{hemi}"],
                                          interpolation="nearest_most_frequent")    # nilearn >= 0.11
        except ValueError:
            texture = surface.vol_to_surf(img, fsavg[f"pial_{hemi}"], inner_mesh=fsavg[f"white_{hemi}"],
                                          interpolation="nearest")                  # older nilearn
        ax = fig.add_subplot(grid[0, k], projection="3d")
        plotting.plot_surf(fsavg[f"infl_{hemi}"], texture, hemi=hemi, view=view, bg_map=fsavg[f"sulc_{hemi}"],
                           cmap=cmap, vmin=args.threshold, vmax=args.vmax, threshold=args.threshold,
                           bg_on_data=False, axes=ax, figure=fig, colorbar=False)
        ax.set_title(f"{hemi.capitalize()} {view}", fontsize=10, pad=-4)
    verts, faces, values = cereb
    for k, (view, title) in enumerate(CEREBELLUM_VIEWS):
        ax = fig.add_subplot(grid[0, n_cortex + k], projection="3d")
        plotting.plot_surf((verts, faces), values, view=view, cmap=cmap, vmin=args.threshold, vmax=args.vmax,
                           threshold=args.threshold, axes=ax, figure=fig, colorbar=False)
        ax.set_title(title, fontsize=10, pad=-4)
    cax = fig.add_axes([0.958, 0.12, 0.008, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(args.threshold, args.vmax))
    fig.colorbar(sm, cax=cax).set_label("Template weight", fontsize=9)
    fig.suptitle(label, x=0.02, y=0.97, ha="left", fontsize=16, fontweight="bold", color=color)
    fig.text(0.02, 0.86, f"positive weight: cortex {shares[0]:.0%}, cerebellum {shares[1]:.0%}, "
                         f"other subcortical {shares[2]:.0%} (not shown)", fontsize=10, color="0.3")
    fig.savefig(out_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)


def contact_sheet(paths, out_path, columns=1, scale=0.5):
    """Stack the per-network images (PIL, same size), one network per row by default."""
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    w, h = int(w * scale), int(h * scale)
    images = [im.resize((w, h), Image.LANCZOS) for im in images]
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (w * columns, h * rows), "white")
    for i, im in enumerate(images):
        sheet.paste(im, ((i % columns) * w, (i // columns) * h))
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aal-dir", required=True, help="folder with AAL3v1.nii and AAL3v1_1mm.xml")
    parser.add_argument("--networks", nargs="*", help="networks to draw (default: the 18)")
    parser.add_argument("--top9", help="networks_top9.csv written by R/Article.Rmd: also draws their sheet")
    parser.add_argument("--threshold", type=float, default=2.0, help="smallest weight shown (default 2)")
    parser.add_argument("--vmax", type=float, default=8.0, help="weight of the full colour (default 8)")
    parser.add_argument("--mesh", default="fsaverage5", choices=["fsaverage5", "fsaverage6", "fsaverage"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", default=os.path.join(REPO, "results", "network_maps"))
    args = parser.parse_args()

    from nilearn import datasets
    from scipy.spatial import cKDTree

    templates = pd.read_csv(TEMPLATES).set_index("network_names")
    colors = pd.read_csv(COLORS).set_index("network")
    networks = args.networks or list(templates.index)
    top9 = None
    if args.top9:
        top9 = pd.read_csv(args.top9)
        order = list(dict.fromkeys(top9.sort_values(["rank", "Condition"])["network"]))
        networks = list(dict.fromkeys((args.networks or []) + order)) if args.networks else order
    unknown = [n for n in networks if n not in templates.index or n not in colors.index]
    if unknown:
        sys.exit(f"Unknown networks {unknown}; available: {list(templates.index)}")

    ref, sch, aal, aal_index = load_atlases(args.aal_dir)
    roi_names = list(templates.columns)
    masks = roi_masks(roi_names, sch, aal, aal_index)
    os.makedirs(args.out, exist_ok=True)
    verts, faces = write_cerebellum_mesh(aal, aal_index, ref.affine, os.path.join(args.out, "cerebellum.nv"))
    cereb_ids = [i for name, i in aal_index.items() if name.startswith(("Cerebellum_", "Vermis_"))]
    cereb_ijk = np.argwhere(np.isin(aal, cereb_ids))
    nearest = cKDTree(nib.affines.apply_affine(ref.affine, cereb_ijk)).query(verts)[1]
    fsavg = datasets.fetch_surf_fsaverage(args.mesh)

    paths = {}
    for net in networks:
        w = templates.loc[net].to_numpy(dtype=float)
        vol = np.zeros(ref.shape, dtype=np.float32)
        for m, v in zip(masks, np.clip(w, 0, None)):
            vol[m] = v
        img = nib.Nifti1Image(vol, ref.affine)
        cereb_values = vol[tuple(cereb_ijk[nearest].T)]
        paths[net] = os.path.join(args.out, f"{net}_brain.png")
        render(net, colors.loc[net, "label"], colors.loc[net, "color"], img, (verts, faces, cereb_values),
               fsavg, args, weight_shares(w, roi_names), paths[net])
        print(f"{net}: {paths[net]}")

    contact_sheet([paths[n] for n in networks], os.path.join(args.out, "networks_all_brain.png"))
    if top9 is not None:
        contact_sheet([paths[n] for n in order], os.path.join(args.out, "networks_top9_brain.png"))
    print(f"Maps written to {args.out}")


if __name__ == "__main__":
    main()
