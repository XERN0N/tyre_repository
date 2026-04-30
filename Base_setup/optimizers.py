from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares, differential_evolution
from tqdm import tqdm


@dataclass
class OptimizerResult:
    params: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str
    clamped_params: list[str] = field(default_factory=list)


class TyreOptimizer(ABC):
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
        return self._result is not None

    @property
    def result(self) -> OptimizerResult:
        if not self.ran:
            raise RuntimeError(f"'{self.label}' has not been run yet.")
        return self._result

    @abstractmethod
    def run(self, residual_fn, args: tuple) -> None: ...


class LeastSquaresOptimizer(TyreOptimizer):
    def run(self, residual_fn, args: tuple) -> None:
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


class _ScalarResidual:
    """Top-level picklable wrapper so differential_evolution can use workers=-1."""
    def __init__(self, residual_fn, args: tuple):
        self.residual_fn = residual_fn
        self.args = args

    def __call__(self, params) -> float:
        return float(np.sum(self.residual_fn(params, *self.args) ** 2))


class GeneticOptimizer(TyreOptimizer):
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
    metadata: dict = {"inputs": inputs, "results": {}}
    for opt in optimizers:
        if not opt.ran:
            continue
        r = opt.result
        params_out = (
            dict(zip(param_names, r.params.tolist())) if param_names else r.params.tolist()
        )
        metadata["results"][opt.label] = {
            "params": params_out,
            "cost": r.cost,
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
    run_dir = plot_path.parent
    metadata = {"results": []}
    for rank, opt in enumerate(results, 1):
        r = opt.result
        params_out = (
            dict(zip(param_names, r.params.tolist())) if param_names else r.params.tolist()
        )
        metadata["results"].append({
            "rank": rank,
            "label": opt.label,
            "params": params_out,
            "cost": r.cost,
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
