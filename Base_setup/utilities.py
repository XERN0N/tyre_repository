from pathlib import Path
from datetime import datetime
import numpy as np


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
