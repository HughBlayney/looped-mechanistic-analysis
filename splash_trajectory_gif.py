"""Animate the final token's residual trajectory for the README.

Fit PCA over all 64 recurrences of a seeded, randomly initialised model, then
show the first 16. Each recurrence reveals its layers as the previous one fades.
Run with ``python splash_trajectory_gif.py``; no checkpoints or data are needed.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

from looped_internals.models import MinimalRecurrentConfig, ModelSpec, load_model

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "docs" / "splash_trajectory.gif"
SEED = 42
SEQ_LEN = 32
DISPLAY_RECURRENCES = 16
FPS = 16.5
CONFIG = MinimalRecurrentConfig(
    d_model=512,
    n_heads=4,
    n_layers=24,
    num_cycles=64,
    input_injection=None,
    gated_mlp=False,
    activation="relu",
    norm_pre_attn=True,
    norm_post_attn=False,
    norm_pre_mlp=True,
    norm_post_mlp=False,
    norm_post_attn_residual=True,
    norm_post_mlp_residual=True,
    norm_post_cycle=True,
)


@torch.no_grad()
def collect_trajectory(config=CONFIG, device="cpu"):
    """Return PCA coordinates shaped (recurrences, layers, 2)."""
    torch.manual_seed(SEED)
    model, structure = load_model(ModelSpec(config), device=device)
    tokens = torch.randint(0, config.d_vocab, (1, SEQ_LEN), device=device)
    _, cache = model.run_with_cache(
        tokens,
        accumulate=True,
        names_filter=structure.hook_names(),
        pos_slice=slice(-1, None),
        device="cpu",
    )
    residuals = torch.stack(
        [
            cache[layer.residual_key][0, 0].float()
            for layer in structure.virtual_layers(config.num_cycles)
        ]
    ).numpy()
    # Use every recurrence to fit one fixed basis for the selected token.
    points = PCA(n_components=2, svd_solver="full").fit_transform(residuals)
    return points.reshape(config.num_cycles, config.n_layers, 2)


def save_gif(points, output_path):
    """Reveal one layer per frame, reusing the same two line artists."""
    recurrences, layers, _ = points.shape
    lower, upper = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    padding = np.maximum(0.1 * (upper - lower), 1e-6)
    cmap = plt.colormaps["summer"]
    colors = cmap(np.linspace(0, 1, recurrences))

    with plt.style.context(ROOT / "plots" / "matlab.mplstyle"):
        fig, ax = plt.subplots(figsize=(3.25, 1.5), layout="constrained")
        ax.set(
            xlabel="PC 1",
            ylabel="PC 2",
            xlim=(lower[0] - padding[0], upper[0] + padding[0]),
            ylim=(lower[1] - padding[1], upper[1] + padding[1]),
            facecolor="white",
        )
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap),
            ax=ax,
            fraction=0.045,
            pad=0.02,
            aspect=25,
        )
        colorbar.set_ticks([0, 1], labels=["Early", "Late"])
        colorbar.set_label("Recurrence", labelpad=-10)
        title = ax.set_title(" ")
        (previous,) = ax.plot([], [], linewidth=2)
        (current,) = ax.plot(
            [], [], linewidth=2, marker="o", markersize=2, markevery=[-1]
        )

        def update(frame):
            recurrence, layer = divmod(frame, layers)
            if recurrence:
                previous.set_data(points[recurrence - 1].T)
                previous.set_color(colors[recurrence - 1])
            previous.set_alpha(1 - (layer + 1) / layers if recurrence else 0)
            current.set_data(points[recurrence, : layer + 1].T)
            current.set_color(colors[recurrence])
            title.set_text(
                f"Recurrence {recurrence + 1}/{recurrences}, Layer {layer + 1}/{layers}"
            )

        animation = FuncAnimation(
            fig, update, frames=recurrences * layers, interval=1000 / FPS
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            animation.save(
                output_path,
                writer=PillowWriter(fps=FPS),
                dpi=200,
                savefig_kwargs={"facecolor": "white", "transparent": False},
            )
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    # Small matrix products are slower with large CPU thread pools.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Collecting {CONFIG.num_cycles} recurrences on {args.device}…", flush=True)
    points = collect_trajectory(device=args.device)[:DISPLAY_RECURRENCES]
    print(f"Saving {points.shape[0] * points.shape[1]} frames…", flush=True)
    save_gif(points, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
