import numpy as np
from pathlib import Path
from Model_library.MF_model import magic_formula_longitudinal
from Model_library.Basic_brush_model import basic_brush
from optimizers import LeastSquaresOptimizer, GeneticOptimizer, save_run
from utilities import get_plot_path, plot_results

np.set_printoptions(suppress=True, precision=3)


def residual(params, v_rel, v_tyre, n_x, Fz, Fs_MF):
    res_BB = basic_brush(
        contact_len=params[0],
        k_bristle=params[1],
        mu_d=params[2],
        mu_s=params[3],
        vel_stribeck=params[4],
        exp_stribeck=params[5],
        v_rel=v_rel,
        v_tyre=v_tyre,
        num_bristle=n_x,
        load_fz=Fz,
    )
    return (res_BB - Fs_MF) / np.max(np.abs(Fs_MF))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, precision=3)

    # tyre parameters
    L       = 0.1                   # contact patch length          [m]         range [0.05-0.2]
    k_0     = 240                   # Bristle micro-stiffness       [1/m]       range [100-600]
    v_tyre  = 16                    # Tyre rolling speed            [m/s]       range [0.1-100]
    Fz      = 700                   # Normal load                   [N]         range N/A

    # friction parameters
    mu_d    = 0.7                   # Dynamic friction coefficient  [-]         range (0,1]
    mu_s    = 1.2                   # Static friction coefficient   [-]         range [0.4,2]
    v_S     = 3.5                   # Stribeck velocity             [m/s]       range [2-10]
    delta_S = 0.6                   # Stribeck exponent             [-]         range [0.1,2]

    # discretisation
    n_x     = 100                   # Spatial grid
    n_v     = 200                   # Velocity grid

    v_rel   = np.linspace(-1, 0, n_v) * v_tyre
    sigma_x = -v_rel / v_tyre

    Fs_MF = magic_formula_longitudinal(v_rel=v_rel, v_tyre=v_tyre, load_fz=Fz)

    initial_guess = np.array([L, k_0, mu_d, mu_s, v_S, delta_S])
    lower_bounds  = np.array([0.05, 100, 0.7, 1, 0.1, 0.1])
    upper_bounds  = np.array([0.12, 800, 2, 3.5, 20, 2])

    param_names = ["L", "k_0", "mu_d", "mu_s", "v_S", "delta_S"]

    plot_path = get_plot_path(Path("plots"), f"Fz{int(Fz)}_vtyre{int(v_tyre)}")

    optimizers = [
        LeastSquaresOptimizer("Least squares", initial_guess, lower_bounds, upper_bounds),
        GeneticOptimizer("Genetic", initial_guess, lower_bounds, upper_bounds),
    ]

    for opt in optimizers:
        opt.run(residual, args=(v_rel, v_tyre, n_x, Fz, Fs_MF))

    force_curves = {
        opt.label: basic_brush(
            v_rel=v_rel, v_tyre=v_tyre, num_bristle=n_x,
            contact_len=opt.result.params[0],
            k_bristle=opt.result.params[1],
            mu_d=opt.result.params[2],
            mu_s=opt.result.params[3],
            vel_stribeck=opt.result.params[4],
            exp_stribeck=opt.result.params[5],
            return_z=False,
        )
        for opt in optimizers if opt.ran
    }

    print("-" * 50)
    for opt in optimizers:
        if opt.ran:
            print(f"{opt.label}: {dict(zip(param_names, opt.result.params.round(4)))}")

    fig, ax = plot_results(
        sigma_x, Fs_MF, force_curves, optimizers, initial_guess, param_names,
        show_table=True,
    )
    fig.savefig(plot_path, dpi=150)

    save_run(
        plot_path,
        optimizers,
        inputs={
            "Fz": Fz, "v_tyre": v_tyre, "n_x": n_x, "n_v": n_v,
            "L": L, "k_0": k_0, "mu_d": mu_d, "mu_s": mu_s,
            "v_S": v_S, "delta_S": delta_S,
        },
        arrays={
            "sigma_x": sigma_x,
            "Fs_MF": Fs_MF,
            **{
                f"Fs_BB_{opt.label.lower().replace(' ', '_')}": force_curves[opt.label]
                for opt in optimizers if opt.ran
            },
        },
        param_names=param_names,
    )

    plt.show()
