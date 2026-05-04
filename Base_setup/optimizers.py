from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares, differential_evolution
from sklearn.metrics import r2_score
from tqdm import tqdm


@dataclass
class OptimizerResult:
    """Immutable record of a completed optimisation run.

    Attributes:
        params:         Fitted parameter vector, same ordering as the bounds arrays.
        cost:           Scalar cost at the solution (sum of squared residuals for
                        genetic; half the sum of squared residuals for least-squares,
                        matching scipy's convention).
        nfev:           Number of residual function evaluations consumed.
        success:        Whether the solver reported convergence.
        message:        Human-readable convergence message from the solver.
        clamped_params: Names (or index strings) of parameters whose initial guess
                        was outside the bounds and was clipped before the run.
                        Empty list when no clamping occurred.
    """
    params: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str
    clamped_params: list[str] = field(default_factory=list)


class TyreOptimizer(ABC):
    """Abstract base class for tyre-model parameter optimizers.

    Subclasses implement ``run()`` for a specific solver strategy. All share
    the same constructor, ``ran`` guard, and ``result`` accessor so they can
    be used interchangeably in loops and search utilities.

    Attributes:
        label:         Human-readable name shown in logs and saved to output files.
        initial_guess: Parameter starting point (copied and clamped inside ``run()``).
        lower_bounds:  Per-parameter lower bound array.
        upper_bounds:  Per-parameter upper bound array.
        param_names:   Optional list of parameter name strings. Set externally by
                       search utilities (e.g. ``bounds_search``) to enable named
                       clamping warnings. Not required for standalone use.

    Args:
        label:         Identifier for this optimizer instance.
        initial_guess: 1-D array of starting parameter values.
        lower_bounds:  1-D array of lower bounds, same length as ``initial_guess``.
        upper_bounds:  1-D array of upper bounds, same length as ``initial_guess``.
    """

    def __init__(
        self,
        label: str,
        initial_guess: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        self.label = label
        self.initial_guess = np.asarray(initial_guess, dtype=float)
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        self._result: OptimizerResult | None = None

    @property
    def ran(self) -> bool:
        """True once ``run()`` has completed successfully."""
        return self._result is not None

    @property
    def result(self) -> OptimizerResult:
        """The completed ``OptimizerResult``. Raises ``RuntimeError`` if called before ``run()``."""
        if not self.ran:
            raise RuntimeError(f"'{self.label}' has not been run yet.")
        return self._result

    @abstractmethod
    def run(self, residual_fn, args: tuple) -> None:
        """Run the optimisation and store the result.

        Args:
            residual_fn: Callable returning a residual vector or scalar.
            args:        Extra positional arguments forwarded to ``residual_fn``.
        """
        ...


class LeastSquaresOptimizer(TyreOptimizer):
    """Gradient-based optimizer using ``scipy.optimize.least_squares``.

    Minimises the vector residual from ``residual_fn`` using the
    Trust Region Reflective algorithm with Jacobian scaling (``x_scale='jac'``)
    and tight tolerances (``xtol=ftol=gtol=1e-12``). Best used when a good
    starting point is available; susceptible to local minima.

    The initial guess is automatically clamped to the bounds before the solve.
    If clamping occurs, the affected parameters are logged and recorded in
    ``result.clamped_params``.
    """

    def run(self, residual_fn, args: tuple) -> None:
        """Run least-squares optimisation and store the result.

        Args:
            residual_fn: Callable ``f(params, *args) -> np.ndarray`` returning
                         the normalised residual vector.
            args:        Extra arguments forwarded to ``residual_fn``.
        """
        tqdm.write(f"Starting {self.label}...")
        guess = np.clip(self.initial_guess, self.lower_bounds, self.upper_bounds)
        clamped_idx = np.where(guess != self.initial_guess)[0]
        param_names = getattr(self, "param_names", None)
        clamped_names = (
            [param_names[i] for i in clamped_idx] if param_names
            else [str(i) for i in clamped_idx]
        )
        if clamped_idx.size:
            pairs = [f"{name} (idx {i})" for name, i in zip(clamped_names, clamped_idx)]
            tqdm.write(f"  Warning: initial guess clamped for {pairs}")
        res = least_squares(
            residual_fn,
            guess,
            bounds=(self.lower_bounds, self.upper_bounds),
            args=args,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            x_scale="jac",
        )
        self._result = OptimizerResult(
            params=res.x,
            cost=float(res.cost),
            nfev=int(res.nfev),
            success=bool(res.success),
            message=res.message,
            clamped_params=clamped_names,
        )


class ScaledMuResidual:
    """Residual wrapper that enforces mu_s > mu_d by reparametrising with a scale factor.

    Instead of fitting mu_s directly, this wrapper fits k_s where mu_s = k_s * mu_d.
    Setting the lower bound of k_s > 1 guarantees mu_s > mu_d for every candidate
    the optimizer evaluates, without needing post-hoc checks.

    The wrapped residual is called with the physical parameter vector
    ``[..., mu_d, mu_s, ...]`` (k_s replaced by mu_s = k_s * mu_d), so the
    underlying brush model receives correct values. Compatible with both
    ``LeastSquaresOptimizer`` (vector residual) and ``GeneticOptimizer``
    (which internally wraps in ``_ScalarResidual``).

    Args:
        residual_fn: Original residual callable ``f(params, *args) -> np.ndarray``.
                     Must accept a physical parameter vector where index ``k_s_idx``
                     is mu_s (not k_s).
        mu_d_idx:    Index of mu_d in the parameter vector. Default 2.
        k_s_idx:     Index of k_s in the parameter vector. Default 3.
    """

    def __init__(self, residual_fn, mu_d_idx: int = 2, k_s_idx: int = 3):
        self.residual_fn = residual_fn
        self.mu_d_idx = mu_d_idx
        self.k_s_idx = k_s_idx

    def __call__(self, params: np.ndarray, *args) -> np.ndarray:
        p = np.array(params, dtype=float)
        p[self.k_s_idx] = p[self.k_s_idx] * p[self.mu_d_idx]  # k_s -> mu_s
        return self.residual_fn(p, *args)


class _ScalarResidual:
    """Top-level picklable callable wrapping a vector residual as a scalar sum-of-squares.

    ``differential_evolution`` with ``workers=-1`` uses ``multiprocessing``,
    which requires all passed callables to be picklable. Closures (``def``
    inside a method) are not picklable; top-level class instances are.

    Args:
        residual_fn: Vector residual callable ``f(params, *args) -> np.ndarray``.
        args:        Extra arguments forwarded to ``residual_fn``.
    """

    def __init__(self, residual_fn, args: tuple):
        self.residual_fn = residual_fn
        self.args = args

    def __call__(self, params) -> float:
        return float(np.sum(self.residual_fn(params, *self.args) ** 2))


class GeneticOptimizer(TyreOptimizer):
    """Global optimizer using ``scipy.optimize.differential_evolution``.

    Population-based evolutionary search; does not require a good starting
    point and avoids local minima. Uses all available CPU cores (``workers=-1``)
    via multiprocessing with deferred updating. The residual is wrapped in
    ``_ScalarResidual`` to satisfy pickling requirements.

    When used inside ``bounds_search`` or ``multi_start``, pass reduced
    ``maxiter`` and ``disp=False`` via ``functools.partial`` to limit output
    and runtime:
    ``partial(GeneticOptimizer, maxiter=50, disp=False)``.

    Args:
        label:         Identifier for this optimizer instance.
        initial_guess: Starting point (not used directly by the solver, but
                       stored on the base class for consistency).
        lower_bounds:  Per-parameter lower bound array.
        upper_bounds:  Per-parameter upper bound array.
        maxiter:       Maximum number of generations. Default 200.
        popsize:       Population size multiplier (total = ``popsize × n_params``).
                       Default 12.
        seed:          Random seed for reproducibility. Default 42.
        disp:          If ``True``, print progress each generation. Set to
                       ``False`` when running alongside a tqdm bar. Default ``True``.
    """

    def __init__(
        self,
        label: str,
        initial_guess: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        maxiter: int = 200,
        popsize: int = 12,
        seed: int = 42,
        disp: bool = True,
    ):
        super().__init__(label, initial_guess, lower_bounds, upper_bounds)
        self.maxiter = maxiter
        self.popsize = popsize
        self.seed = seed
        self.disp = disp

    def run(self, residual_fn, args: tuple) -> None:
        """Run differential evolution and store the result.

        Args:
            residual_fn: Callable ``f(params, *args) -> np.ndarray`` returning
                         a residual vector. Wrapped internally as a scalar
                         sum-of-squares for the solver.
            args:        Extra arguments forwarded to ``residual_fn``.
        """
        tqdm.write(f"Starting {self.label}...")

        res = differential_evolution(
            _ScalarResidual(residual_fn, args),
            bounds=list(zip(self.lower_bounds, self.upper_bounds)),
            maxiter=self.maxiter,
            popsize=self.popsize,
            workers=-1,
            polish=False,
            disp=self.disp,
            updating="deferred",
            seed=self.seed,
        )
        self._result = OptimizerResult(
            params=res.x,
            cost=float(res.fun),
            nfev=int(res.nfev),
            success=bool(res.success),
            message=res.message,
        )


def save_run(
    plot_path: Path,
    optimizers: list[TyreOptimizer],
    inputs: dict,
    arrays: dict[str, np.ndarray],
    param_names: list[str] | None = None,
) -> None:
    """Save a single-run result as a JSON metadata file and an NPZ array archive.

    Writes two sibling files next to ``plot_path``, sharing its stem:

    - ``{stem}.json`` — physical inputs and per-optimizer scalar results
      (params, cost, nfev, success, message).
    - ``{stem}.npz``  — all arrays in ``arrays`` plus one ``params_{label}``
      entry per optimizer.

    Only optimizers where ``opt.ran is True`` are included. Optimizers that
    have not been run are silently skipped.

    Args:
        plot_path:   Path to the saved plot file. Sibling output files are
                     derived from this path via ``.with_suffix()``.
        optimizers:  List of optimizer instances to save results for.
        inputs:      Dict of scalar physical inputs to embed in the JSON
                     (e.g. ``{"Fz": 700, "v_tyre": 16, ...}``).
        arrays:      Dict of named numpy arrays to store in the NPZ
                     (e.g. ``{"sigma_x": ..., "Fs_MF": ..., "Fs_BB_...": ...}``).
        param_names: If provided, params are stored as a named dict in the JSON;
                     otherwise stored as a plain list.
    """
    Fs_MF = arrays.get("Fs_MF")
    metadata: dict = {"inputs": inputs, "results": {}}
    for opt in optimizers:
        if not opt.ran:
            continue
        r = opt.result
        params_out = (
            dict(zip(param_names, r.params.tolist())) if param_names else r.params.tolist()
        )
        curve_key = f"Fs_BB_{opt.label.lower().replace(' ', '_')}"
        r2 = (
            float(r2_score(Fs_MF, arrays[curve_key]))
            if Fs_MF is not None and curve_key in arrays
            else None
        )
        metadata["results"][opt.label] = {
            "params": params_out,
            "cost": r.cost,
            "r2": r2,
            "nfev": r.nfev,
            "success": r.success,
            "message": r.message,
        }

    json_path = plot_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2))

    npz_arrays = dict(arrays)
    for opt in optimizers:
        if opt.ran:
            key = opt.label.lower().replace(" ", "_")
            npz_arrays[f"params_{key}"] = opt.result.params

    npz_path = plot_path.with_suffix(".npz")
    np.savez(npz_path, **npz_arrays)

    print(f"Saved: {plot_path.name}, {json_path.name}, {npz_path.name}")


def save_search_results(
    plot_path: Path,
    results: list[TyreOptimizer],
    arrays: dict[str, np.ndarray],
    param_names: list[str] | None = None,
    filename_stem: str = "bounds_search",
) -> None:
    """Save all results from a search run (bounds_search / multi_start) into the run folder.

    Writes two files into the same folder as ``plot_path``:

    - ``{filename_stem}.json`` — ranked list of all results with label, params,
      cost, nfev, success, message, and any clamped parameter names.
    - ``{filename_stem}.npz``  — all arrays from ``arrays`` plus a stacked
      ``params`` matrix of shape ``(n_results, n_params)`` in rank order.

    The NPZ is self-contained for replotting: load ``sigma_x``, ``Fs_MF``, and
    ``Fs_BB[i]`` to reproduce any individual result without re-running the search.

    Args:
        plot_path:      Path to the main plot file; the run folder is its parent.
        results:        List of optimizer instances, ordered by ascending cost
                        (as returned by ``bounds_search`` or ``multi_start``).
        arrays:         Shared arrays to include in the NPZ, e.g.
                        ``{"sigma_x": ..., "Fs_MF": ..., "Fs_BB": ...}``.
                        ``Fs_BB`` should be shape ``(n_results, n_v)`` in the
                        same rank order as ``results``.
        param_names:    If provided, params are stored as a named dict in the JSON;
                        otherwise stored as a plain list.
        filename_stem:  Base name for the output files. Default ``"bounds_search"``.
                        Use ``"multi_start"`` when saving multi-start results.
    """
    run_dir = plot_path.parent
    Fs_MF = arrays.get("Fs_MF")
    Fs_BB_all = arrays.get("Fs_BB")

    metadata = {"results": []}
    for i, opt in enumerate(results):
        r = opt.result
        params_out = (
            dict(zip(param_names, r.params.tolist())) if param_names else r.params.tolist()
        )
        r2 = (
            float(r2_score(Fs_MF, Fs_BB_all[i]))
            if Fs_MF is not None and Fs_BB_all is not None
            else None
        )
        metadata["results"].append({
            "rank": i + 1,
            "label": opt.label,
            "params": params_out,
            "cost": r.cost,
            "r2": r2,
            "nfev": r.nfev,
            "success": r.success,
            "message": r.message,
            "clamped_params": r.clamped_params,
        })

    json_path = run_dir / f"{filename_stem}.json"
    json_path.write_text(json.dumps(metadata, indent=2))

    npz_path = run_dir / f"{filename_stem}.npz"
    np.savez(npz_path, **arrays, params=np.array([opt.result.params for opt in results]))

    print(f"Saved: {json_path.name}, {npz_path.name}")
