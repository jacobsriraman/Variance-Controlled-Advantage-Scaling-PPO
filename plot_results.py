from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="runs/vectr", help="Root run directory")
    p.add_argument("--env-id", type=str, default="CartPole-v1", help="Environment folder name")
    p.add_argument("--outdir", type=str, default="analysis", help="Where to save plots/tables")
    p.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of method folders to include",
    )
    p.add_argument(
        "--x-col",
        type=str,
        default="update",
        choices=["update", "global_step"],
        help="Column to use for x-axis",
    )
    p.add_argument(
        "--var-col",
        type=str,
        default="advantage_var",
        help="Variance column in metrics.csv",
    )
    return p.parse_args()


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def discover_methods(env_root: Path, methods_filter: List[str] | None = None) -> Dict[str, List[Path]]:
    if not env_root.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_root}")

    out: Dict[str, List[Path]] = {}
    for method_dir in sorted(p for p in env_root.iterdir() if p.is_dir()):
        if methods_filter is not None and method_dir.name not in methods_filter:
            continue
        seed_dirs = sorted(p for p in method_dir.iterdir() if p.is_dir())
        if seed_dirs:
            out[method_dir.name] = seed_dirs

    if not out:
        raise RuntimeError(f"No method folders found under {env_root}")

    return out


def choose_return_col(df: pd.DataFrame) -> str:
    if "eval_return_mean" in df.columns:
        return "eval_return_mean"
    if "episode_return_mean" in df.columns:
        return "episode_return_mean"
    raise KeyError("Neither 'eval_return_mean' nor 'episode_return_mean' found in metrics.csv")


def align_runs(
    dfs: List[pd.DataFrame],
    x_col: str,
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cleaned = []
    for df in dfs:
        if x_col not in df.columns:
            raise KeyError(f"Missing x column '{x_col}'")
        if y_col not in df.columns:
            raise KeyError(f"Missing y column '{y_col}'")
        tmp = df[[x_col, y_col]].dropna().reset_index(drop=True)
        cleaned.append(tmp)

    min_len = min(len(df) for df in cleaned)
    cleaned = [df.iloc[:min_len].copy() for df in cleaned]

    x = cleaned[0][x_col].to_numpy()
    y = np.stack([df[y_col].to_numpy() for df in cleaned], axis=0)
    return x, y


def summarize_method(
    method_name: str,
    seed_dirs: List[Path],
    x_col: str,
    var_col: str,
) -> Dict:
    metric_dfs = []
    per_run_rows = []

    for run_dir in seed_dirs:
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config.json"

        if not metrics_path.exists():
            print(f"Skipping {run_dir} (no metrics.csv)")
            continue

        df = pd.read_csv(metrics_path)
        cfg = load_config(config_path)
        metric_dfs.append(df)

        ret_col = choose_return_col(df)

        row = {
            "method": method_name,
            "run_dir": str(run_dir),
            "seed": cfg.get("seed", np.nan),
            "return_col_used": ret_col,
            "final_return": float(df[ret_col].dropna().iloc[-1]),
            "best_return": float(df[ret_col].dropna().max()),
            "final_variance": float(df[var_col].dropna().iloc[-1]) if var_col in df.columns else np.nan,
            "reward_transform": cfg.get("reward_transform", ""),
            "reward_scale": cfg.get("reward_scale", np.nan),
            "use_vectr": cfg.get("use_vectr", False),
            "vectr_target_std": cfg.get("vectr_target_std", np.nan),
        }
        per_run_rows.append(row)

    if not metric_dfs:
        raise RuntimeError(f"No valid runs found for method '{method_name}'")

    ret_col = choose_return_col(metric_dfs[0])
    x_ret, y_ret = align_runs(metric_dfs, x_col=x_col, y_col=ret_col)

    if var_col in metric_dfs[0].columns:
        x_var, y_var = align_runs(metric_dfs, x_col=x_col, y_col=var_col)
        var_mean = y_var.mean(axis=0)
        var_std = y_var.std(axis=0)
    else:
        x_var = x_ret
        var_mean = np.full_like(x_ret, fill_value=np.nan, dtype=float)
        var_std = np.full_like(x_ret, fill_value=np.nan, dtype=float)

    return {
        "method": method_name,
        "return_col_used": ret_col,
        "x_return": x_ret,
        "return_mean": y_ret.mean(axis=0),
        "return_std": y_ret.std(axis=0),
        "x_var": x_var,
        "var_mean": var_mean,
        "var_std": var_std,
        "per_run": pd.DataFrame(per_run_rows),
    }


def pretty_method_name(method: str) -> str:
    mapping = {
        "identity": "Identity",
        "minmax": "MinMax",
        "tanh": "Tanh",
        "zscore_std1.0": "Z-score",
        "vectr_std1.0": "VECTR",
    }
    if method in mapping:
        return mapping[method]
    if method.startswith("scale_"):
        return f"Scale ×{method.split('_', 1)[1]}"
    return method


def plot_learning_curves(summaries: Dict[str, Dict], outpath: Path, x_label: str, y_label: str) -> None:
    plt.figure(figsize=(10, 6))
    for method, s in summaries.items():
        x = s["x_return"]
        mean = s["return_mean"]
        std = s["return_std"]
        label = pretty_method_name(method)
        plt.plot(x, mean, linewidth=2, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_variance_curves(summaries: Dict[str, Dict], outpath: Path, x_label: str, y_label: str) -> None:
    plt.figure(figsize=(10, 6))
    for method, s in summaries.items():
        if np.isnan(s["var_mean"]).all():
            continue
        x = s["x_var"]
        mean = s["var_mean"]
        std = s["var_std"]
        label = pretty_method_name(method)
        plt.plot(x, mean, linewidth=2, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_final_returns(summaries: Dict[str, Dict], outpath: Path) -> pd.DataFrame:
    rows = []
    for method, s in summaries.items():
        per_run = s["per_run"]
        rows.append(
            {
                "method": method,
                "pretty_method": pretty_method_name(method),
                "final_return_mean": per_run["final_return"].mean(),
                "final_return_std": per_run["final_return"].std(ddof=0),
                "best_return_mean": per_run["best_return"].mean(),
                "num_runs": len(per_run),
            }
        )

    df = pd.DataFrame(rows).sort_values("final_return_mean", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    plt.bar(x, df["final_return_mean"], yerr=df["final_return_std"], capsize=4)
    plt.xticks(x, df["pretty_method"], rotation=30, ha="right")
    plt.ylabel("Final Return")
    plt.title("Final Return by Method (mean ± std)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    return df


def plot_variance_vs_return(summaries: Dict[str, Dict], outpath: Path) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(8, 6))

    for method, s in summaries.items():
        per_run = s["per_run"]
        if per_run["final_variance"].isna().all():
            continue

        x = per_run["final_variance"].to_numpy()
        y = per_run["final_return"].to_numpy()

        plt.scatter(x, y, s=60, alpha=0.8, label=pretty_method_name(method))

        for _, row in per_run.iterrows():
            rows.append(
                {
                    "method": method,
                    "seed": row["seed"],
                    "final_variance": row["final_variance"],
                    "final_return": row["final_return"],
                }
            )

    scatter_df = pd.DataFrame(rows)

    pearson_r = np.nan
    if len(scatter_df) >= 2:
        pearson_r = scatter_df["final_variance"].corr(scatter_df["final_return"], method="pearson")

    plt.xlabel("Final Variance")
    plt.ylabel("Final Return")
    plt.title(f"Variance vs Final Return (Pearson r = {pearson_r:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    return scatter_df


def main() -> None:
    args = parse_args()

    env_root = Path(args.root) / args.env_id
    outdir = Path(args.outdir) / args.env_id
    outdir.mkdir(parents=True, exist_ok=True)

    runs_by_method = discover_methods(env_root, methods_filter=args.methods)

    summaries: Dict[str, Dict] = {}
    return_col_used = None

    for method, seed_dirs in runs_by_method.items():
        summary = summarize_method(
            method_name=method,
            seed_dirs=seed_dirs,
            x_col=args.x_col,
            var_col=args.var_col,
        )
        summaries[method] = summary
        if return_col_used is None:
            return_col_used = summary["return_col_used"]

    x_label = "Global Step" if args.x_col == "global_step" else "Update"
    y_label = "Evaluation Return" if return_col_used == "eval_return_mean" else "Training Return"

    plot_learning_curves(
        summaries=summaries,
        outpath=outdir / "learning_curves.png",
        x_label=x_label,
        y_label=y_label,
    )

    plot_variance_curves(
        summaries=summaries,
        outpath=outdir / "variance_curves.png",
        x_label=x_label,
        y_label=args.var_col,
    )

    final_df = plot_final_returns(
        summaries=summaries,
        outpath=outdir / "final_returns.png",
    )
    final_df.to_csv(outdir / "final_return_summary.csv", index=False)

    scatter_df = plot_variance_vs_return(
        summaries=summaries,
        outpath=outdir / "variance_vs_return.png",
    )
    scatter_df.to_csv(outdir / "variance_vs_return_points.csv", index=False)

    all_runs = []
    for method, s in summaries.items():
        all_runs.append(s["per_run"])
    pd.concat(all_runs, ignore_index=True).to_csv(outdir / "per_run_summary.csv", index=False)

    print(f"Saved results to: {outdir}")


if __name__ == "__main__":
    main()
