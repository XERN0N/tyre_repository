import numpy as np
from pathlib import Path
from Model_library.MF_model import magic_formula_lateral
from Model_library.Basic_brush_model import basic_brush
from optimizers import LeastSquaresOptimizer, GeneticOptimizer, save_run, ScaledMuResidual
from utilities import get_plot_path, plot_results


def residual(params, v_rel, v_tyre, n_x, Fz, Fs_MF):
    """Compute normalised residual between the brush model and the Magic Formula reference.

    Evaluates ``basic_brush`` at the current parameter vector and returns the
    element-wise difference from the Magic Formula curve, scaled by its peak
    absolute value so all residual components are dimensionless and O(1).

    Args:
        params:  1-D parameter vector ``[L, k_0, mu_d, mu_s, v_S, delta_S]``:

                 - L         — contact patch length [m]
                 - k_0       — bristle stiffness [1/m]
                 - mu_d      — dynamic friction coefficient [-]
                 - mu_s      — static friction coefficient [-]
                 - v_S       — Stribeck velocity [m/s]
                 - delta_S   — Stribeck exponent [-]

        v_rel:   Relative sliding velocity array [m/s], shape ``(n_v,)``.
        v_tyre:  Tyre rolling velocity [m/s], scalar.
        n_x:     Number of bristle discretisation points (spatial grid).
        Fz:      Normal load [N].
        Fs_MF:   Magic Formula reference force curve, shape ``(n_v,)``.

    Returns:
        Normalised residual vector ``(Fs_BB - Fs_MF) / max(|Fs_MF|)``,
        shape ``(n_v,)``.  Passed directly to ``scipy.optimize.least_squares``.
    """
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

    v_rel   = np.linspace(-1, 0, n_v) * v_tyre  # full slip range: 0 (free-rolling) → -v_tyre (locked wheel)
    sigma_y = -v_rel / v_tyre                   # lateral slip σ_y = -v_rel/v_tyre ∈ [0, 1]

    Fs_MF = magic_formula_lateral(v_rel=v_rel, v_tyre=v_tyre, load_fz=Fz)

    initial_guess = np.array([L, k_0, mu_d, mu_s, v_S, delta_S])
    #                          L      k_0   mu_d  mu_s  v_S   delta_S
    lower_bounds  = np.array([0.05,  100,  0.7,  1.0,  2,  0.1])
    upper_bounds  = np.array([0.12,  800,  2.0,  3.5,  20.0, 2.0])

    param_names = ["L", "k_0", "mu_d", "mu_s", "v_S", "delta_S"]

    scaled_param_names = ["L", "k_0", "mu_d", "k_s", "v_S", "delta_S"]
    #                              L      k_0   mu_d        k_s  v_S   delta_S
    scaled_initial_guess = np.array([L, k_0,  mu_d, mu_s/mu_d, v_S,  delta_S])
    scaled_lower_bounds  = np.array([0.05, 100,  0.7,      1.05, 0.1,  0.1])
    scaled_upper_bounds  = np.array([0.12, 800,  2.0,      5.0,  20.0, 2.0])

    plot_path = get_plot_path(Path("plots"), f"Fz{int(Fz)}_vtyre{int(v_tyre)}")

    scaled_optimizers = [
        LeastSquaresOptimizer("LS Scaled Mu", scaled_initial_guess, scaled_lower_bounds, scaled_upper_bounds),
        GeneticOptimizer("Genetic Scaled Mu", scaled_initial_guess, scaled_lower_bounds, scaled_upper_bounds),
    ]

    scaled_residual = ScaledMuResidual(residual)

    for opt in scaled_optimizers:
        opt.run(scaled_residual, args=(v_rel, v_tyre, n_x, Fz, Fs_MF))

    scaled_force_curves = {
        opt.label: basic_brush(
            v_rel=v_rel, v_tyre=v_tyre, num_bristle=n_x,
            contact_len=opt.result.params[0],
            k_bristle=opt.result.params[1],
            mu_d=opt.result.params[2],
            mu_s=opt.result.params[3] * opt.result.params[2],  # k_s * mu_d
            vel_stribeck=opt.result.params[4],
            exp_stribeck=opt.result.params[5],
            return_z=False,
        )
        for opt in scaled_optimizers if opt.ran
    }


    scaled_plot_path = plot_path.with_stem(plot_path.stem + "_scaled_mu")
    save_run(
        scaled_plot_path,
        scaled_optimizers,
        inputs={"Fz": Fz, "v_tyre": v_tyre, "n_x": n_x, "n_v": n_v,
                "L": L, "k_0": k_0, "mu_d": mu_d, "k_s": mu_s/mu_d, "v_S": v_S, "delta_S": delta_S},
        arrays={"sigma_y": sigma_y, "Fs_MF": Fs_MF,
                **{f"Fs_BB_{opt.label.lower().replace(' ', '_')}": scaled_force_curves[opt.label]
                   for opt in scaled_optimizers if opt.ran}},
        param_names=scaled_param_names,
    )
    fig, ax = plot_results(
        sigma_y, Fs_MF, scaled_force_curves,
        scaled_optimizers, scaled_initial_guess, scaled_param_names,
        show_table=True,
    )
    ax.set_title("Linear brush model scaled friction vs magic formula")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()

    # optimizers = [
    #     LeastSquaresOptimizer("Least squares", initial_guess, lower_bounds, upper_bounds),
    #     GeneticOptimizer("Genetic", initial_guess, lower_bounds, upper_bounds),
    # ]
    # for opt in optimizers:
    #     opt.run(residual, args=(v_rel, v_tyre, n_x, Fz, Fs_MF))
    # force_curves = {
    #     opt.label: basic_brush(
    #         v_rel=v_rel, v_tyre=v_tyre, num_bristle=n_x,
    #         contact_len=opt.result.params[0],
    #         k_bristle=opt.result.params[1],
    #         mu_d=opt.result.params[2],
    #         mu_s=opt.result.params[3],
    #         vel_stribeck=opt.result.params[4],
    #         exp_stribeck=opt.result.params[5],
    #         return_z=False,
    #     )
    #     for opt in optimizers if opt.ran
    # }
    # print("-" * 50)
    # for opt in optimizers:
    #     if opt.ran:
    #         print(f"{opt.label}: {dict(zip(param_names, opt.result.params.round(4)))}")
    # for opt in scaled_optimizers:
    #     if opt.ran:
    #         p = opt.result.params
    #         print(f"{opt.label}: {dict(zip(scaled_param_names, p.round(4)))}  mu_s={p[2]*p[3]:.4f}")

    # fig, ax = plot_results(
    #     sigma_y, Fs_MF, {**force_curves, **scaled_force_curves},
    #     optimizers + scaled_optimizers, initial_guess, param_names,
    #     show_table=False,
    # )
    # fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    # save_run(
    #     plot_path,
    #     optimizers,
    #     inputs={"Fz": Fz, "v_tyre": v_tyre, "n_x": n_x, "n_v": n_v,
    #             "L": L, "k_0": k_0, "mu_d": mu_d, "mu_s": mu_s, "v_S": v_S, "delta_S": delta_S},
    #     arrays={"sigma_y": sigma_y, "Fs_MF": Fs_MF,
    #             **{f"Fs_BB_{opt.label.lower().replace(' ', '_')}": force_curves[opt.label]
    #                for opt in optimizers if opt.ran}},
    #     param_names=param_names,
    # )