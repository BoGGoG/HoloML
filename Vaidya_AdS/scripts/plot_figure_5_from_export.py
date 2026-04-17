"""
Reproduce Figure 5 from exported geodesic data.

Reads the .npz files written by plot_figure_5(..., export_dir=...) and
produces the same 2x3 polar grid — no JAX or geodesic integration needed.

Usage:
    python scripts/plot_figure_5_from_export.py
    python scripts/plot_figure_5_from_export.py --export-dir path/to/figure_5_data
                                                 --out figure_5_repro.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.style.use(["science", "grid"])
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "axes.linewidth": 1.8,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 3.2,
        "lines.markersize": 6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def get_mass(v, m_0=1.0, v_s=1.0):
    """m(v) = (m_0+1)/2 * tanh(v/v_s) + (m_0-1)/2"""
    return (m_0 + 1.0) / 2.0 * np.tanh(v / v_s) + (m_0 - 1.0) / 2.0


def load_export_dir(export_dir: Path):
    """
    Load all per-v0 .npz files from export_dir.

    Returns a list of dicts sorted by v0, each with keys:
        v0          : float
        r_stars     : (n_valid,) array
        trajectories: list of (n_trunc, 3) arrays — cols: v, r, x
    """
    files = sorted(export_dir.glob("figure_5_geodesics_v0*.npz"))
    if not files:
        print(f"No export files found in {export_dir}", file=sys.stderr)
        sys.exit(1)

    datasets = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        datasets.append(
            {
                "v0": float(d["v0"]),
                "r_stars": d["r_stars"],
                "trajectories": list(d["trajectories"]),  # list of ragged arrays
            }
        )

    datasets.sort(key=lambda x: x["v0"])
    return datasets


def make_figure_5(datasets, out_path: Path):
    # Collect all r_star values across panels to build a common colormap
    all_r_stars = np.concatenate([d["r_stars"] for d in datasets])
    norm = Normalize(vmin=all_r_stars.min(), vmax=all_r_stars.max())
    cmap = plt.cm.cool
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), subplot_kw={"projection": "polar"})
    plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)
    axes_flat = axes.flatten()

    for i, data in enumerate(datasets):
        ax = axes_flat[i]
        v0 = data["v0"]
        r_stars = data["r_stars"]
        trajs = data["trajectories"]  # (n_valid, n_steps, 6)

        # Boundary circle at R = π/2 (r → ∞)
        theta_boundary = np.linspace(-np.pi / 2, np.pi / 2, 100)
        ax.plot(theta_boundary, [np.pi / 2] * 100, "k-", linewidth=2)

        # Apparent horizon
        m_val = get_mass(v0)
        if m_val > 0:
            R_horizon = np.arctan(np.sqrt(m_val))
            theta_h = np.linspace(0, 2 * np.pi, 200)
            ax.plot(theta_h, [R_horizon] * 200, color="brown", linewidth=2)

        # Geodesics — each traj is (n_trunc, 3): columns v, r, x
        for r_star, traj in zip(r_stars, trajs):
            r_t = traj[:, 1]
            x_t = traj[:, 2]
            R_plot = np.arctan(r_t)
            color = cmap(norm(r_star))
            ax.plot(x_t, R_plot, color=color, linewidth=1)
            ax.plot(-x_t, R_plot, color=color, linewidth=1)

        ax.set_title(r"$v_0$" + f"= {v0:g}", fontsize=12)
        ax.set_ylim(0, np.pi / 2 + 0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Turning Point ($r_\ast$)", fontsize=12)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Figure 5 from exported data")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("geodesics_plots/figure_5_data"),
        help="Directory containing the figure_5_geodesics_v0*.npz files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <export-dir>/figure_5_repro.png)",
    )
    args = parser.parse_args()

    out_path = args.out if args.out is not None else args.export_dir / "figure_5_repro.png"

    datasets = load_export_dir(args.export_dir)
    print(f"Loaded {len(datasets)} panels: v0 = {[d['v0'] for d in datasets]}")
    make_figure_5(datasets, out_path)
