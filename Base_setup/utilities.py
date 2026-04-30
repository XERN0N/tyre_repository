from pathlib import Path
from datetime import datetime
import itertools
import numpy as np
from tqdm import tqdm


def get_plot_path(plot_dir: Path, default_desc: str) -> Path:
    plot_dir.mkdir(exist_ok=True)

    desc = input("Plot name (Enter for auto): ").strip()
    if not desc:
        desc = default_desc

    existing = [d for d in plot_dir.iterdir() if d.is_dir() and desc in d.name]
    if existing:
        print(f"  Warning: {len(existing)} existing run folder(s) contain '{desc}':")
        for d in sorted(existing):
            print(f"    {d}")
        if input("  Use same name anyway? [y/N]: ").strip().lower() != "y":
            raise SystemExit("Aborted.")

    stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{desc}"
    run_dir = plot_dir / stem
    run_dir.mkdir()
    return run_dir / f"{stem}.png"


def multi_start(
    optimizer_cls,
    residual_fn,
    args: tuple,
    param_grid: dict[str, list],
    base_guess: np.ndarray,
    param_names: list[str],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> list:
    """Run optimizer_cls from every combination in param_grid, return all sorted by cost.

    param_grid:  {param_name: [val1, val2, ...], ...}  — unlisted params use base_guess.
    results[0]  is the best (lowest cost).
    """
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    results = []
    with tqdm(total=len(combinations), desc="Multi-start", unit="run") as pbar:
        for i, combo in enumerate(combinations, 1):
            guess = base_guess.copy()
            for name, val in zip(keys, combo):
                guess[param_names.index(name)] = val

            label = ", ".join(f"{n}={v}" for n, v in zip(keys, combo))
            opt = optimizer_cls(f"Start {i} ({label})", guess, lower_bounds, upper_bounds)
            opt.run(residual_fn, args)
            tqdm.write(f"  [{i}/{len(combinations)}] cost={opt.result.cost:.6e}  {label}")
            pbar.set_postfix(cost=f"{opt.result.cost:.3e}")
            pbar.update(1)
            results.append(opt)

    return sorted(results, key=lambda o: o.result.cost)


def bounds_search(
    optimizer_classes,
    residual_fn,
    args: tuple,
    initial_guess: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    param_bounds_grid: dict[str, list[tuple]],
    param_names: list[str],
) -> list:
    """Run each optimizer class for every combination of per-parameter bound ranges, return sorted by cost.

    optimizer_classes:  single class or list of classes.
    param_bounds_grid:  {param_name: [(lo1, hi1), (lo2, hi2), ...], ...}
    results[0]  is the best (lowest cost).
    """
    if not isinstance(optimizer_classes, (list, tuple)):
        optimizer_classes = [optimizer_classes]

    keys = list(param_bounds_grid.keys())
    combinations = list(itertools.product(*[param_bounds_grid[k] for k in keys]))
    n_runs = len(combinations) * len(optimizer_classes)
    print(f"Bounds search: {len(combinations)} combinations × {len(optimizer_classes)} optimizers = {n_runs} runs")

    results = []
    with tqdm(total=len(combinations), desc="Bounds search", unit="combo") as pbar:
        for i, combo in enumerate(combinations, 1):
            lower = base_lower.copy()
            upper = base_upper.copy()
            for name, (lo, hi) in zip(keys, combo):
                idx = param_names.index(name)
                lower[idx] = lo
                upper[idx] = hi

            bounds_label = ", ".join(f"{n}=[{lo},{hi}]" for n, (lo, hi) in zip(keys, combo))
            best_cost = float("inf")
            for cls in optimizer_classes:
                cls_name = getattr(cls, "__name__", None) or cls.func.__name__
                cls_name = cls_name.replace("Optimizer", "")
                opt = cls(f"{cls_name} Bounds {i} ({bounds_label})", initial_guess, lower, upper)
                opt.param_names = param_names
                opt.run(residual_fn, args)
                tqdm.write(f"  [{i}/{len(combinations)}] {cls_name:<12} cost={opt.result.cost:.6e}  {bounds_label}")
                best_cost = min(best_cost, opt.result.cost)
                results.append(opt)
            pbar.set_postfix(best_cost=f"{best_cost:.3e}")
            pbar.update(1)

    return sorted(results, key=lambda o: o.result.cost)


def plot_results(
    sigma_x: np.ndarray,
    Fs_MF: np.ndarray,
    force_curves: dict[str, np.ndarray],
    optimizers: list,
    initial_guess: np.ndarray,
    param_names: list[str],
    *,
    show_table: bool = True,
) -> tuple:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401 — registers science plot styles

    with plt.style.context(["science", "no-latex", "grid", "high-vis"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sigma_x, Fs_MF, label="Magic Formula", linewidth=1)
        for opt in optimizers:
            if opt.ran:
                ax.plot(sigma_x, force_curves[opt.label],
                        label=f"Basic Brush ({opt.label})", linewidth=1)

        ax.set_xlabel(r'Lateral slip $\sigma_y$ [-]')
        ax.set_ylabel(r'Lateral force normalized $F_y$ [-]')
        ax.legend(loc="right")

        if show_table:
            col_labels = [
                r"$L$ [m]", r"$k_0\ \left[\frac{1}{m}\right]$",
                r"$\mu_d$", r"$\mu_s$",
                r"$v_S\ \left[\frac{m}{s}\right]$", r"$\delta_S$",
            ]
            row_labels = ["Initial guess"] + [opt.label for opt in optimizers if opt.ran]
            table_data = [
                [f"{v:.3f}" for v in initial_guess],
                *([f"{v:.3f}" for v in opt.result.params] for opt in optimizers if opt.ran),
            ]
            tbl = ax.table(
                cellText=table_data,
                rowLabels=row_labels,
                colLabels=col_labels,
                bbox=[0.42, 0.02, 0.56, 0.30],
                cellLoc="center",
            )
            tbl.auto_set_font_size(True)
            tbl.set_zorder(10)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_linewidth(0.5)
                cell.set_edgecolor("#bbbbbb")
                cell.set_facecolor("white")
                cell.set_zorder(10)

        fig.tight_layout()

    return fig, ax
