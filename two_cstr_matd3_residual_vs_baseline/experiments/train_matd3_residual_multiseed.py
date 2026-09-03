"""Three-seed comparison of original and residual MATD3.

Each seed gets a freshly trained baseline and a residual fine-tuning run. The
old ``matd3_distributed_v1`` checkpoint/result directory is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_matd3 import plot_effect


SEEDS = (42, 123, 2024)
KEYS = (
    "mean_return",
    "iae_mean_mol_m3",
    "settling_time_mean_s",
    "peak_abs_error_mol_m3",
    "steady_error_std_mol_m3",
    "action_variation_normalized",
)
LABELS = {
    "mean_return": "平均回报 ↑",
    "iae_mean_mol_m3": "平均浓度绝对误差 ↓",
    "settling_time_mean_s": "平均稳定时间 (s) ↓",
    "peak_abs_error_mol_m3": "峰值绝对误差 ↓",
    "steady_error_std_mol_m3": "稳态误差标准差 ↓",
    "action_variation_normalized": "归一化动作变化率 ↓",
}


def run_seed(seed: int, args: argparse.Namespace) -> list[dict]:
    root = args.output_dir
    baseline_dir = root / "baseline" / f"seed_{seed}"
    residual_dir = root / "residual" / f"seed_{seed}"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--timesteps", str(args.timesteps),
        "--eval-episodes", str(args.eval_episodes),
        "--num-envs", str(args.num_envs),
        "--batch-size", str(args.batch_size),
        "--buffer-size", str(args.buffer_size),
        "--seed", str(seed),
        "--report-every", str(args.report_every),
        "--cpu",
    ]
    baseline_script = Path(__file__).with_name("train_matd3.py")
    residual_script = Path(__file__).with_name("train_matd3_residual.py")
    subprocess.run([sys.executable, str(baseline_script), *common, "--output-dir", str(baseline_dir)], check=True)
    subprocess.run(
        [
            sys.executable,
            str(residual_script),
            *common,
            "--lr-residual", str(args.lr_residual),
            "--lr-critic", str(args.lr_critic),
            "--max-residual", str(args.max_residual),
            "--baseline-checkpoint", str(baseline_dir / "matd3_distributed.pt"),
            "--output-dir", str(residual_dir),
        ],
        check=True,
    )
    rows = []
    for variant, folder in (("Original MATD3", baseline_dir), ("Residual MATD3", residual_dir)):
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"variant": variant, "seed": seed, **metrics})
    return rows


def rewards(path: Path) -> np.ndarray:
    with (path / "training_episodes.csv").open(encoding="utf-8") as stream:
        return np.asarray([float(row["reward"]) for row in csv.DictReader(stream)], dtype=np.float64)


def plot_learning(root: Path, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=170)
    for variant, color, folder_name in (
        ("Original MATD3", "#0072B2", "baseline"),
        ("Residual MATD3", "#D55E00", "residual"),
    ):
        curves = []
        for seed in SEEDS:
            values = rewards(root / folder_name / f"seed_{seed}")
            window = min(50, max(5, len(values) // 20))
            curves.append(np.convolve(values, np.ones(window) / window, mode="valid"))
        n = min(len(values) for values in curves)
        data = np.stack([values[:n] for values in curves])
        x = np.arange(1, n + 1)
        mean, std = np.mean(data, axis=0), np.std(data, axis=0, ddof=1)
        ax.plot(x, mean, color=color, linewidth=2.2, label=variant)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="#555", linestyle="--", linewidth=1, label="Theoretical maximum (0)")
    ax.set(title="Original vs residual MATD3 — 3-seed learning curves", xlabel="Episode", ylabel="Team return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows: list[dict], root: Path) -> None:
    rows.sort(key=lambda item: (item["variant"], item["seed"]))
    (root / "per_seed_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = []
    for variant in ("Original MATD3", "Residual MATD3"):
        selected = [row for row in rows if row["variant"] == variant]
        summary.append({
            "variant": variant,
            **{f"{key}_mean": float(np.mean([row[key] for row in selected])) for key in KEYS},
            **{f"{key}_std": float(np.std([row[key] for row in selected], ddof=1)) for key in KEYS},
        })
    (root / "comparison_metrics_3seeds.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (root / "comparison_metrics_3seeds.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["variant"] + [item for key in KEYS for item in (f"{key}_mean", f"{key}_std")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Original MATD3 vs residual MATD3 — three random seeds",
        "",
        "Each seed uses a freshly trained original MATD3 and a residual MATD3 fine-tuned from that seed's baseline checkpoint. Values are mean ± sample standard deviation across seeds; each run uses 30 evaluation episodes.",
        "",
        "| 版本 | " + " | ".join(LABELS.values()) + " |",
        "|---|" + "---:|" * len(KEYS),
    ]
    for item in summary:
        lines.append("| " + item["variant"] + " | " + " | ".join(f"{item[key + '_mean']:.3f} ± {item[key + '_std']:.3f}" for key in KEYS) + " |")
    lines += ["", "Higher return is better; errors, settling time, and action variation are better when lower."]
    (root / "comparison_metrics_3seeds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr-residual", type=float, default=5e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--max-residual", type=float, default=0.15)
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "matd3_residual_multiseed_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_seed, seed, args) for seed in SEEDS]
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    write_outputs(rows, args.output_dir)
    plot_learning(args.output_dir, args.output_dir / "comparison_learning_curves_3seeds.png")
    for folder_name, label in (("baseline", "original_matd3"), ("residual", "residual_matd3")):
        trajectories = []
        for seed in SEEDS:
            trajectories.extend(np.load(args.output_dir / folder_name / f"seed_{seed}" / "trajectories.npy", allow_pickle=True).tolist())
        plot_effect(trajectories, args.output_dir / f"{label}_3seeds_effect.png")
    print(f"Completed {len(SEEDS) * 2} MATD3 runs; results written to {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
