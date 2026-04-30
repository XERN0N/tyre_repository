from pathlib import Path
from datetime import datetime
import itertools
import json
import numpy as np
from tqdm import tqdm


def get_plot_path(plot_dir: Path, default_desc: str) -> Path:
    """Prompt for a descriptive run name, check for duplicates, create a run folder, and return the plot path.

    Interactively asks the user for a name. If left blank, falls back to
    ``default_desc``. Warns if any existing subfolder of ``plot_dir`` already
    contains the chosen name and asks for confirmation before proceeding.

    The run folder and all sibling output files (JSON, NPZ) share the same
    stem: ``{YYYYMMDD_HHMMSS}_{desc}``.

    Args:
        plot_dir:     Root folder under which the timestamped run folder is created.
        default_desc: Fallback name used when the user presses Enter without typing.

    Returns:
        Path to the ``.png`` file inside the newly created run folder,
        e.g. ``plots/20260430_143022_my_run/20260430_143022_my_run.png``.

    Raises:
        SystemExit: If the user declines to reuse an existing name.
    """
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
    """Run an optimizer from every combination of initial-guess values, return all results sorted by cost.

    Builds the cartesian product of values in ``param_grid`` and runs
    ``optimizer_cls`` once per combination, substituting the varied values
    into ``base_guess`` while leaving all other parameters unchanged.
    Useful for checking whether least-squares converges to the same minimum
    from different starting points (local-minima sensitivity).

    Progress is displayed via a tqdm bar; per-run cost is printed above it.

    Args:
        optimizer_cls: Optimizer class (or ``functools.partial`` thereof) to instantiate
                       for each run. Must accept ``(label, guess, lower_bounds, upper_bounds)``.
        residual_fn:   Residual function passed to ``optimizer_cls.run()``.
        args:          Extra arguments forwarded to ``residual_fn``.
        param_grid:    Mapping of parameter name → list of values to try,
                       e.g. ``{"v_S": [1.0, 5.0, 12.5], "delta_S": [0.3, 0.6]}``.
        base_guess:    Full initial-guess array; varied params are substituted in-place.
        param_names:   Ordered list of parameter names matching the guess array indices.
        lower_bounds:  Lower bound array, same length as ``base_guess``.
        upper_bounds:  Upper bound array, same length as ``base_guess``.

    Returns:
        List of completed optimizer instances sorted by ascending cost.
        ``results[0]`` is the best fit found.
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
    """Run optimizers over every combination of per-parameter bound ranges, return all results sorted by cost.

    Builds the cartesian product of bound ranges in ``param_bounds_grid`` and
    runs every optimizer in ``optimizer_classes`` for each combination, keeping
    ``base_lower``/``base_upper`` fixed for parameters not listed. Useful for
    diagnosing whether the bounded region excludes better minima or whether
    different regions produce structurally different solutions.

    The initial guess is automatically clamped to each combination's bounds
    before the run; a warning is printed if clamping occurs.

    Progress is displayed via a tqdm bar (one tick per bound combination);
    per-run cost is printed above it. Pass ``disp=False`` via
    ``functools.partial`` to suppress per-iteration output from
    ``GeneticOptimizer``.

    Args:
        optimizer_classes: Single class or list of classes (or ``functools.partial``
                           thereof). Each is run for every bound combination.
                           Must accept ``(label, guess, lower_bounds, upper_bounds)``.
        residual_fn:       Residual function passed to each optimizer's ``run()``.
        args:              Extra arguments forwarded to ``residual_fn``.
        initial_guess:     Starting guess array used for every run (clamped per combination).
        base_lower:        Default lower bound array; entries are overridden per combination.
        base_upper:        Default upper bound array; entries are overridden per combination.
        param_bounds_grid: Mapping of parameter name → list of ``(lo, hi)`` tuples,
                           e.g. ``{"v_S": [(0.1, 8.0), (5.0, 20.0)]}``.
                           Repeat the same bound to vary only one side:
                           ``{"v_S": [(0.1, 5.0), (0.1, 10.0), (0.1, 20.0)]}``.
        param_names:       Ordered list of parameter names matching the bound array indices.

    Returns:
        List of all completed optimizer instances sorted by ascending cost.
        ``results[0]`` is the best fit found across all combinations and optimizers.
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
    """Plot Magic Formula and fitted brush-model force curves, optionally with a parameter table.

    Renders all curves that have a corresponding entry in ``force_curves`` using
    the ``science`` / ``high-vis`` matplotlib style. The optional inset table
    compares the initial guess against each optimizer's fitted parameters.

    ``matplotlib`` and ``scienceplots`` are imported lazily so this module can
    be imported without a display environment.

    Args:
        sigma_x:       Longitudinal slip array (x-axis), shape ``(n_v,)``.
        Fs_MF:         Magic Formula reference force curve, shape ``(n_v,)``.
        force_curves:  Mapping of optimizer label → fitted brush-model force curve.
                       Only optimizers present in this dict are plotted.
        optimizers:    List of optimizer instances; used for labels and table rows.
                       Only those with ``opt.ran == True`` are included.
        initial_guess: Parameter array used as the first table row.
        param_names:   Ordered parameter names used as table column headers.
        show_table:    If ``True`` (default), render the inset parameter comparison
                       table. Set to ``False`` for a clean curve-only plot.

    Returns:
        ``(fig, ax)`` — the matplotlib Figure and Axes objects.
        The caller is responsible for ``fig.savefig()`` and ``plt.show()``.
    """
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


def replot_from_file(path: Path, top_n: int = 5, show_table: bool = True) -> tuple:
    """Reload a saved run from disk and reproduce the force-curve plot without recomputing.

    Reads the ``.json`` and ``.npz`` files produced by ``save_run`` or
    ``save_search_results`` and calls ``plot_results`` with the stored data.
    Both single-run and search-run saves are supported; the format is detected
    automatically from the JSON structure.

    Pass either the ``.json`` or the ``.npz`` file — the sibling file is located
    automatically.  For search runs the ``top_n`` best results (rank order) are
    plotted; for single runs all stored optimizers are plotted.

    Args:
        path:       Path to the ``.json`` or ``.npz`` output file from a previous run.
        top_n:      Maximum number of results to plot for search-run saves.
                    Ignored for single-run saves.  Default 5.
        show_table: If ``True`` (default), render the inset parameter table.
                    Forwarded directly to ``plot_results``.

    Returns:
        ``(fig, ax)`` — the matplotlib Figure and Axes objects.
        The caller is responsible for ``fig.savefig()`` and ``plt.show()``.

    Raises:
        ValueError: If ``path`` does not have a ``.json`` or ``.npz`` suffix.
    """
    path = Path(path)
    if path.suffix == ".json":
        json_path, npz_path = path, path.with_suffix(".npz")
    elif path.suffix == ".npz":
        npz_path, json_path = path, path.with_suffix(".json")
    else:
        raise ValueError(f"Expected a .json or .npz file, got: {path.suffix!r}")

    metadata = json.loads(json_path.read_text())
    data = np.load(npz_path)

    # backwards-compatible with runs saved before the sigma_x → sigma_y rename
    sigma_y = data["sigma_y"] if "sigma_y" in data else data["sigma_x"]
    Fs_MF = data["Fs_MF"]

    class _Stub:
        """Minimal stand-in for a completed optimizer instance."""
        ran = True
        def __init__(self, label: str, params: np.ndarray):
            self.label = label
            self.result = type("_R", (), {"params": np.asarray(params)})()

    results_raw = metadata["results"]
    is_search = isinstance(results_raw, list)

    if is_search:
        entries = results_raw[:top_n]
        first_params = entries[0]["params"]
        param_names = list(first_params.keys()) if isinstance(first_params, dict) else None
        stubs = [
            _Stub(
                e["label"],
                list(e["params"].values()) if isinstance(e["params"], dict) else e["params"],
            )
            for e in entries
        ]
        force_curves = {e["label"]: data["Fs_BB"][i] for i, e in enumerate(entries)}
        initial_guess = stubs[0].result.params
    else:
        param_names = None
        stubs, force_curves = [], {}
        for label, res in results_raw.items():
            p = res["params"]
            if isinstance(p, dict):
                if param_names is None:
                    param_names = list(p.keys())
                params = np.array(list(p.values()))
            else:
                params = np.array(p)
            stubs.append(_Stub(label, params))
            force_curves[label] = data[f"Fs_BB_{label.lower().replace(' ', '_')}"]
        inp = metadata.get("inputs", {})
        initial_guess = (
            np.array([inp[n] for n in param_names])
            if param_names and all(n in inp for n in param_names)
            else stubs[0].result.params
        )

    return plot_results(sigma_y, Fs_MF, force_curves, stubs, initial_guess, param_names,
                        show_table=show_table)
