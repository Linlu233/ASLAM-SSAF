from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_ablation_bars import configure_publication_style, save_figure


PALETTE = {
    "input": "#F4F5F7",
    "attr": "#D7E4F4",
    "struct": "#DDECE4",
    "pair": "#F5E8CC",
    "enhanced": "#E3EFD8",
    "final": "#F2D8D1",
    "note": "#FAFAFA",
    "edge": "#2F3338",
    "panel": "#FCFCFD",
    "panel_edge": "#AAB2BD",
    "title": "#20242A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a journal-style architecture diagram for ASLAM-SSAF."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("Figure"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--variant",
        choices=("all", "journal", "aslam"),
        default="all",
        help="Which architecture style to render.",
    )
    parser.add_argument(
        "--hide_header_text",
        action="store_true",
        help="Render a version without the top title and subtitle text.",
    )
    return parser.parse_args()


def add_box(ax, x: float, y: float, w: float, h: float, text: str, facecolor: str, fontsize: float = 10.2) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=1.2",
        linewidth=1.15,
        edgecolor=PALETTE["edge"],
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def add_panel(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    fontsize: float = 10.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=1.6",
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        edgecolor=PALETTE["panel_edge"],
        facecolor=PALETTE["panel"],
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 1.6,
        y + h - 3.0,
        title,
        ha="left",
        va="top",
        fontsize=fontsize,
        weight="semibold",
        color=PALETTE["title"],
        zorder=1,
    )


def add_note(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fontsize: float = 8.6,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=1.0",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        edgecolor="#8A8F98",
        facecolor=PALETTE["note"],
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fontsize, zorder=2)


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float, connection: str | None = None) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12.5,
        linewidth=1.25,
        color=PALETTE["edge"],
        shrinkA=0.0,
        shrinkB=0.0,
        connectionstyle=connection or "arc3,rad=0.0",
        zorder=2,
    )
    ax.add_patch(arrow)


def add_poly_arrow(ax, points: list[tuple[float, float]]) -> None:
    if len(points) < 2:
        return
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color=PALETTE["edge"],
        linewidth=1.25,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=2,
    )
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    head_len = 2.2
    if abs(x2 - x1) >= abs(y2 - y1):
        head_x = x2 - head_len if x2 >= x1 else x2 + head_len
        add_arrow(ax, head_x, y2, x2, y2)
    else:
        head_y = y2 - head_len if y2 >= y1 else y2 + head_len
        add_arrow(ax, x2, head_y, x2, y2)


def add_label(
    ax,
    x: float,
    y: float,
    text: str,
    fontsize: float = 8.8,
    ha: str = "left",
) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color="#5E6977",
        style="italic",
        ha=ha,
        va="center",
        zorder=5,
    )


def save_outputs(fig, output_dir: Path, dpi: int, stem: str, include_default_alias: bool = False) -> None:
    outputs = [
        output_dir / f"{stem}.png",
        output_dir / f"{stem}.pdf",
        output_dir / f"figure_{stem}.tif",
        output_dir / f"figure_{stem}.pdf",
    ]
    if include_default_alias:
        outputs.extend(
            [
                output_dir / "model_architecture_aslam_ssaf.png",
                output_dir / "model_architecture_aslam_ssaf.pdf",
                output_dir / "figure_model_architecture_aslam_ssaf.tif",
                output_dir / "figure_model_architecture_aslam_ssaf.pdf",
            ]
        )
    for output_path in outputs:
        save_figure(fig, output_path, dpi)


def draw_journal_architecture(output_dir: Path, dpi: int, hide_header_text: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.4), dpi=240)
    ax.set_xlim(0, 156)
    ax.set_ylim(0, 88)
    ax.axis("off")

    add_panel(ax, 2, 12, 15, 60, "Inputs")
    add_panel(ax, 20, 52, 100, 18, "Dual-Branch Representation Learning")
    add_panel(ax, 20, 32, 100, 18, "Structure-Aware Sequential Encoding")
    add_panel(ax, 20, 12, 100, 18, "Edge-Level Pair Modeling")
    add_panel(ax, 125, 12, 25, 60, "Adaptive Fusion and Prediction", fontsize=9.2)

    add_box(ax, 4.9, 56.0, 10.4, 8.0, "Node Attributes\n$X$", PALETTE["input"])
    add_box(ax, 4.9, 36.0, 10.4, 8.0, "Graph Topology\n$A$", PALETTE["input"])
    add_box(ax, 4.9, 14.2, 10.4, 8.0, "Edge Heuristics\n$e_{uv}$", PALETTE["input"])

    add_box(ax, 23.0, 55.6, 16.5, 8.8, "Attribute Encoder\nMLP", PALETTE["attr"])
    add_box(ax, 43.0, 55.6, 18.5, 8.8, "Attribute Node\nEmbeddings $h^A$", PALETTE["attr"], fontsize=9.6)
    add_box(ax, 66.0, 55.6, 18.0, 8.8, "Attribute / Structure\nPair Scorers", PALETTE["pair"], fontsize=9.45)
    add_box(ax, 88.8, 55.6, 16.0, 8.8, "Base Fusion\nGate", PALETTE["pair"], fontsize=9.8)
    add_box(ax, 108.6, 55.6, 10.8, 8.8, "Base Score\n$\\ell_{base}$", PALETTE["pair"], fontsize=9.0)

    add_box(ax, 23.0, 34.6, 15.5, 8.8, "Residual GCN\nEncoder", PALETTE["struct"], fontsize=9.6)
    add_box(ax, 42.0, 34.6, 19.0, 8.8, "Local-Global Graph\nEncoder\nGATv2 + GCN", PALETTE["struct"], fontsize=8.3)
    add_box(ax, 65.0, 34.6, 20.0, 8.8, "Bi-directional Selective\nMixer\nDepthwise Conv + Gated Scan", PALETTE["struct"], fontsize=7.8)
    add_box(ax, 89.0, 34.6, 14.8, 8.8, "Structure Node\nEmbeddings $h^S$", PALETTE["struct"], fontsize=9.1)
    add_note(ax, 106.2, 35.0, 11.2, 6.8, "Multi-scale states\n$\\{h^{(0)},\\ldots,h^{(L)}\\}$")

    add_box(ax, 37.5, 12.8, 21.0, 9.2, "Pair Composer\n$[u,v,|u-v|,u\\odot v,e_{uv}]$", PALETTE["pair"], fontsize=9.65)
    add_box(ax, 63.0, 12.8, 15.4, 9.2, "Pairwise Token\nAttention", PALETTE["enhanced"], fontsize=9.7)
    add_box(ax, 82.5, 12.8, 16.4, 9.2, "SSM-Attention\nPair Fusion", PALETTE["enhanced"], fontsize=9.25)
    add_box(ax, 103.0, 12.8, 15.2, 9.2, "Expert Mix\n$\\ell_{expert}$", PALETTE["enhanced"], fontsize=9.65)
    add_note(ax, 64.0, 22.0, 38.5, 3.1, "Pair tokens: $[h_u^A,h_v^A,h_u^S,h_v^S,\\phi_A,\\phi_S,\\phi_\\Delta,\\phi_\\mu,\\phi_e]$")

    add_note(ax, 127.0, 58.4, 20.8, 4.1, "$\\ell_{uv}=(1-\\alpha_{uv})\\ell_{base}+\\alpha_{uv}\\ell_{expert}$", fontsize=8.1)
    add_box(ax, 129.2, 41.0, 18.0, 9.2, "Adaptive Edge Gate\n$\\alpha_{uv}$", PALETTE["final"], fontsize=9.8)
    add_box(ax, 129.2, 22.0, 18.0, 10.0, "Final Link Logit\n$\\ell_{uv}$", PALETTE["final"], fontsize=9.65)

    add_arrow(ax, 15.3, 60.0, 23.0, 60.0)
    add_arrow(ax, 39.5, 60.0, 43.0, 60.0)
    add_arrow(ax, 61.5, 60.0, 66.0, 60.0)
    add_arrow(ax, 84.0, 60.0, 88.8, 60.0)
    add_arrow(ax, 104.8, 60.0, 108.6, 60.0)

    add_arrow(ax, 15.3, 40.0, 23.0, 40.0)
    add_arrow(ax, 38.5, 39.0, 42.0, 39.0)
    add_arrow(ax, 61.0, 39.0, 65.0, 39.0)
    add_arrow(ax, 85.0, 39.0, 89.0, 39.0)

    add_arrow(ax, 15.3, 17.4, 37.5, 17.4)
    add_arrow(ax, 58.5, 17.4, 63.0, 17.4)
    add_arrow(ax, 78.4, 17.4, 82.5, 17.4)
    add_arrow(ax, 98.9, 17.4, 103.0, 17.4)

    add_poly_arrow(ax, [(50.2, 55.6), (50.2, 27.0), (43.6, 27.0), (43.6, 22.0)])
    add_poly_arrow(ax, [(96.4, 34.6), (96.4, 27.0), (54.8, 27.0), (54.8, 22.0)])

    add_poly_arrow(ax, [(119.4, 60.0), (123.2, 60.0), (123.2, 47.8), (129.2, 47.8)])
    add_poly_arrow(ax, [(103.8, 39.0), (103.8, 44.8), (129.2, 44.8)])
    add_poly_arrow(ax, [(118.2, 17.4), (124.2, 17.4), (124.2, 42.0), (129.2, 42.0)])
    add_arrow(ax, 138.2, 41.0, 138.2, 32.0)

    if not hide_header_text:
        ax.text(2.5, 81.0, "ASLAM-SSAF Architecture", fontsize=15.6, weight="semibold")
        ax.text(
            2.5,
            76.8,
            "Selective State-Space Attention Fusion for Attribute-Structure Link Prediction",
            fontsize=11.0,
        )
    save_outputs(
        fig,
        output_dir,
        dpi,
        stem=(
            "model_architecture_aslam_ssaf_journalstyle_notext"
            if hide_header_text
            else "model_architecture_aslam_ssaf_journalstyle"
        ),
        include_default_alias=False,
    )
    plt.close(fig)


def draw_aslam_architecture(output_dir: Path, dpi: int, hide_header_text: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.0), dpi=240)
    ax.set_xlim(0, 152)
    ax.set_ylim(0, 84)
    ax.axis("off")

    add_panel(ax, 2, 10, 16.5, 60, "Inputs")
    add_panel(ax, 20, 50, 100, 20, "Attribute Branch")
    add_panel(ax, 20, 30, 100, 18, "Edge Interaction and Fusion")
    add_panel(ax, 20, 10, 100, 18, "Structure Branch")
    add_panel(ax, 125, 10, 23, 60, "Output Head")

    add_box(ax, 4.1, 55.4, 11.2, 8.0, "Node Attributes\n$X$", PALETTE["input"], fontsize=9.55)
    add_box(ax, 4.1, 32.8, 11.2, 8.0, "Edge Heuristics\n$e_{uv}$", PALETTE["input"], fontsize=9.55)
    add_box(ax, 4.1, 12.6, 11.2, 8.0, "Graph Topology\n$A$", PALETTE["input"], fontsize=9.55)

    add_box(ax, 23.0, 55.0, 16.5, 8.8, "Attribute Encoder\nMLP", PALETTE["attr"], fontsize=10.0)
    add_box(ax, 43.0, 55.0, 18.0, 8.8, "Attribute Embeddings\n$h^A$", PALETTE["attr"], fontsize=10.0)
    add_box(ax, 65.5, 55.0, 17.5, 8.8, "Attribute / Structure\nPair Scorers", PALETTE["pair"], fontsize=9.65)
    add_box(ax, 87.5, 55.0, 15.8, 8.8, "Base Fusion\nGate", PALETTE["pair"], fontsize=9.8)
    add_box(ax, 107.0, 55.0, 10.8, 8.8, "Base Score\n$\\ell_{base}$", PALETTE["pair"], fontsize=9.0)

    add_box(ax, 41.8, 32.2, 20.5, 9.2, "Pair Composer\n$[u,v,|u-v|,u\\odot v,e_{uv}]$", PALETTE["pair"], fontsize=9.7)
    add_box(ax, 66.5, 32.2, 15.2, 9.2, "Pairwise Token\nAttention", PALETTE["enhanced"], fontsize=9.8)
    add_box(ax, 85.5, 32.2, 16.2, 9.2, "SSM-Attention\nPair Fusion", PALETTE["enhanced"], fontsize=9.35)
    add_box(ax, 103.5, 32.2, 15.0, 9.2, "Expert Mix\n$\\ell_{expert}$", PALETTE["enhanced"], fontsize=9.7)
    add_note(ax, 69.0, 42.4, 31.8, 2.5, "Pair tokens: $[h_u^A,h_v^A,h_u^S,h_v^S,\\phi_A,\\phi_S,\\phi_\\Delta,\\phi_\\mu,\\phi_e]$", fontsize=8.0)

    add_box(ax, 23.0, 12.2, 15.0, 8.8, "Residual GCN\nEncoder", PALETTE["struct"], fontsize=9.9)
    add_box(ax, 44.5, 12.2, 18.8, 8.8, "Local-Global Graph\nEncoder\nGATv2 + GCN", PALETTE["struct"], fontsize=8.3)
    add_box(ax, 67.0, 12.2, 20.0, 8.8, "Bi-directional Selective\nMixer\nDepthwise Conv + Gated Scan", PALETTE["struct"], fontsize=7.8)
    add_box(ax, 90.5, 12.2, 15.5, 8.8, "Structure Embeddings\n$h^S$", PALETTE["struct"], fontsize=9.0)
    add_note(ax, 108.2, 21.0, 10.2, 4.6, "Multi-scale states\n$\\{h^{(0)},\\ldots,h^{(L)}\\}$", fontsize=8.0)

    add_note(ax, 126.8, 60.4, 19.2, 3.6, "$\\ell_{uv}=(1-\\alpha_{uv})\\ell_{base}+\\alpha_{uv}\\ell_{expert}$", fontsize=7.8)
    add_box(ax, 127.8, 40.5, 18.0, 9.2, "Adaptive Edge Gate\n$\\alpha_{uv}$", PALETTE["final"], fontsize=9.8)
    add_box(ax, 127.8, 22.2, 18.0, 10.0, "Final Link Logit\n$\\ell_{uv}$", PALETTE["final"], fontsize=9.7)

    add_arrow(ax, 15.3, 59.4, 23.0, 59.4)
    add_arrow(ax, 39.5, 59.4, 43.0, 59.4)
    add_arrow(ax, 61.0, 59.4, 65.5, 59.4)
    add_arrow(ax, 83.0, 59.4, 87.5, 59.4)
    add_arrow(ax, 103.3, 59.4, 107.0, 59.4)

    add_arrow(ax, 15.3, 36.8, 41.8, 36.8)
    add_arrow(ax, 62.3, 36.8, 66.5, 36.8)
    add_arrow(ax, 81.7, 36.8, 85.5, 36.8)
    add_arrow(ax, 101.7, 36.8, 103.5, 36.8)

    add_arrow(ax, 15.3, 16.6, 23.0, 16.6)
    add_arrow(ax, 38.0, 16.6, 44.5, 16.6)
    add_arrow(ax, 63.3, 16.6, 67.0, 16.6)
    add_arrow(ax, 87.0, 16.6, 90.5, 16.6)

    add_arrow(ax, 52.0, 55.0, 52.0, 41.4)
    add_poly_arrow(ax, [(98.25, 21.0), (98.25, 26.6), (58.0, 26.6), (58.0, 32.2)])

    add_poly_arrow(ax, [(117.8, 59.4), (119.0, 59.4), (119.0, 48.2), (127.8, 48.2)])
    add_poly_arrow(ax, [(106.0, 16.6), (123.8, 16.6), (123.8, 41.8), (127.8, 41.8)])
    add_poly_arrow(ax, [(118.5, 36.8), (121.6, 36.8), (121.6, 44.8), (127.8, 44.8)])
    add_arrow(ax, 136.8, 40.5, 136.8, 32.2)

    if not hide_header_text:
        ax.text(3.0, 76.0, "ASLAM-SSAF Architecture", fontsize=15.4, weight="semibold")
        ax.text(
            3.0,
            72.1,
            "Selective State-Space Attention Fusion for Attribute-Structure Link Prediction",
            fontsize=10.8,
        )
    save_outputs(
        fig,
        output_dir,
        dpi,
        stem=(
            "model_architecture_aslam_ssaf_aslamstyle_notext"
            if hide_header_text
            else "model_architecture_aslam_ssaf_aslamstyle"
        ),
        include_default_alias=not hide_header_text,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    if args.variant in ("all", "journal"):
        draw_journal_architecture(args.output_dir, args.dpi, hide_header_text=args.hide_header_text)
    if args.variant in ("all", "aslam"):
        draw_aslam_architecture(args.output_dir, args.dpi, hide_header_text=args.hide_header_text)
    print(f"Saved architecture figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
