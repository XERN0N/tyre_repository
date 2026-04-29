from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares, differential_evolution


@dataclass
class OptimizerResult:
    params: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str


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
        print(f"Starting {self.label}...")
        res = least_squares(
            residual_fn,
            self.initial_guess,
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
        )


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
    ):
        super().__init__(label, initial_guess, lower_bounds, upper_bounds)
        self.maxiter = maxiter
        self.popsize = popsize
        self.seed = seed

    def run(self, residual_fn, args: tuple) -> None:
        print(f"Starting {self.label}...")

        def scalar_residual(params):
            return float(np.sum(residual_fn(params, *args) ** 2))

        res = differential_evolution(
            scalar_residual,
            bounds=list(zip(self.lower_bounds, self.upper_bounds)),
            maxiter=self.maxiter,
            popsize=self.popsize,
            workers=-1,
            polish=False,
            disp=True,
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
