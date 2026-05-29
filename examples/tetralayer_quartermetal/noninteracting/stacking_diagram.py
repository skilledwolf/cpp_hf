"""ABCA stacking diagram for rhombohedral tetralayer graphene.

Schematic illustration of Figure 1(a) in Huang & Wolf (2026): four
graphene layers stacked rhombohedrally, each shifted by one sublattice
vector relative to the layer below.  The eight sublattice labels
(A1, B1, A2, B2, A3, B3, A4, B4) are positioned on each layer.

Pure-matplotlib drawing — no model, no SCF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def hex_ring(center, radius, color, ax, *, lw=1.0, alpha=1.0):
    """Draw a hexagon (just the ring, no fill) at center with given radius."""
    angles = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=2)


def draw_layer(ax, *, x_shift: float, y_offset: float, color: str,
               labels: tuple[str, str], a: float = 1.0,
               n_cells: int = 3, alpha: float = 1.0):
    """One graphene layer (A, B sublattice atoms + hex ring grid).

    Layer is drawn at vertical position ``y_offset`` with a horizontal
    in-plane shift of ``x_shift`` (in units of the sublattice vector).
    """
    # Sublattice vectors for graphene with lattice constant a.
    a1 = np.array([1.5 * a, 0.5 * np.sqrt(3) * a])
    a2 = np.array([1.5 * a, -0.5 * np.sqrt(3) * a])
    delta_AB = np.array([a, 0.0])  # A → B sublattice offset

    # Build atom list over an n_cells×n_cells supercell.
    A_atoms, B_atoms = [], []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            R = i * a1 + j * a2
            A_atoms.append(R)
            B_atoms.append(R + delta_AB)
    A_atoms = np.array(A_atoms)
    B_atoms = np.array(B_atoms)

    # Apply the inter-layer shift in x and add the y_offset for the
    # vertical column-of-layers view.
    A_atoms[:, 0] += x_shift
    A_atoms[:, 1] += y_offset
    B_atoms[:, 0] += x_shift
    B_atoms[:, 1] += y_offset

    # Bonds: each A atom connects to three B atoms at distance a.
    bonds = []
    for A in A_atoms:
        for B in B_atoms:
            if 0.99 * a < np.linalg.norm(B - A) < 1.01 * a:
                bonds.append((A, B))
    for A, B in bonds:
        ax.plot([A[0], B[0]], [A[1], B[1]],
                color=color, lw=0.8, alpha=alpha * 0.7, zorder=1)

    # Atoms
    ax.scatter(A_atoms[:, 0], A_atoms[:, 1], s=40, c=color,
               edgecolors="black", linewidth=0.5, alpha=alpha, zorder=3)
    ax.scatter(B_atoms[:, 0], B_atoms[:, 1], s=40, c="white",
               edgecolors=color, linewidth=1.2, alpha=alpha, zorder=3)

    # Sublattice labels — pick atoms near the centre.
    cidx_A = int(np.argmin(np.sum((A_atoms - np.array([x_shift, y_offset]))**2,
                                    axis=1)))
    cidx_B = int(np.argmin(np.sum((B_atoms - np.array([x_shift, y_offset]))**2,
                                    axis=1)))
    ax.annotate(labels[0], A_atoms[cidx_A], xytext=(-12, -2),
                 textcoords="offset points", color="black", fontsize=9,
                 ha="right", va="center", zorder=4)
    ax.annotate(labels[1], B_atoms[cidx_B], xytext=(12, -2),
                 textcoords="offset points", color=color, fontsize=9,
                 ha="left", va="center", zorder=4)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "stacking.png",
    )
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 8.0), constrained_layout=True)

    # ABCA stack: each layer shifted by one sublattice vector relative
    # to the previous.  The shift amount for the schematic is taken as
    # +a in x; in the real lattice it would be a 1/3 lattice vector,
    # but the visual is clearer with a full a-shift.
    layer_colors = ["#c84b4b", "#3f7fbf", "#5fb95f", "#c84b4b"]
    layer_letters = ["A", "B", "C", "A"]
    a = 1.0
    layer_dy = 5.0  # vertical separation between layers
    in_plane_shift = a  # per-layer horizontal slide

    for i in range(4):
        draw_layer(
            ax, x_shift=i * in_plane_shift, y_offset=i * layer_dy,
            color=layer_colors[i], labels=(f"A{i+1}", f"B{i+1}"),
            a=a, n_cells=2,
        )
        # Layer letter on the far left.
        ax.annotate(layer_letters[i],
                     xy=(-2.5 * a, i * layer_dy),
                     ha="right", va="center",
                     fontsize=14, fontweight="bold",
                     color=layer_colors[i])
        ax.annotate(f"layer {i+1}",
                     xy=(-2.5 * a, i * layer_dy - 0.6),
                     ha="right", va="center",
                     fontsize=8, color="0.4")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rhombohedral ABCA stacking — tetralayer graphene")

    # z-axis arrow on the right
    x_axis = ax.get_xlim()[1] + 1.0
    y_top = 3 * layer_dy + 1.0
    y_bot = -1.0
    ax.annotate("", xy=(x_axis, y_top), xytext=(x_axis, y_bot),
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
    ax.text(x_axis + 0.4, (y_top + y_bot) / 2, "z",
             ha="left", va="center", fontsize=11, color="0.3")

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
