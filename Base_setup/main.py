import numpy as np
from pathlib import Path
from datetime import datetime
from Model_library.MF_model import magic_formula_longitudinal
from Model_library.Basic_brush_model import basic_brush
from optimizers import LeastSquaresOptimizer, GeneticOptimizer, save_run

np.set_printoptions(suppress=True, precision=3)


def get_plot_path(plot_dir: Path, default_desc: str) -> Path:
    plot_dir.mkdir(exist_ok=True)

    desc = input("Plot name (Enter for auto): ").strip()
    if not desc:
        desc = default_desc

    existing = [f for f in plot_dir.iterdir() if desc in f.stem]
    if existing:
        print(f"  Warning: {len(existing)} existing plot(s) contain '{desc}':")
        for f in sorted(existing):
            print(f"    {f}")
        if input("  Use same name anyway? [y/N]: ").strip().lower() != "y":
            raise SystemExit("Aborted.")

    return plot_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{desc}.png"


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
    import scienceplots
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

    plot_path = get_plot_path(Path("plots"), f"Fz{int(Fz)}_vtyre{int(v_tyre)}")

    optimizers = [
        LeastSquaresOptimizer("Least squares", initial_guess, lower_bounds, upper_bounds),
        GeneticOptimizer("Genetic", initial_guess, lower_bounds, upper_bounds),
    ]

    for opt in optimizers:
        opt.run(residual, args=(v_rel, v_tyre, n_x, Fz, Fs_MF))

    force_curves = {
        opt.label: basic_brush(
            v_rel=v_rel,
            v_tyre=v_tyre,
            num_bristle=n_x,
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

    param_names = ["L", "k_0", "mu_d", "mu_s", "v_S", "delta_S"]

    print("-" * 50)
    for opt in optimizers:
        if opt.ran:
            print(f"{opt.label}: {dict(zip(param_names, opt.result.params.round(4)))}")

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
