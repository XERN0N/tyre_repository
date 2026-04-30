import sys
from pathlib import Path
import matplotlib.pyplot as plt
from utilities import replot_from_file

PLOTS_DIR = Path("plots")
REPLOT_DIR = Path("re-plots")


def list_runs(plots_dir: Path) -> list[Path]:
    """Return all .json files found recursively under plots_dir, sorted by name."""
    return sorted(plots_dir.rglob("*.json"))


if __name__ == "__main__":
    if not PLOTS_DIR.exists():
        print("No plots folder found — run main.py or single_run.py first.")
        sys.exit(1)

    runs = list_runs(PLOTS_DIR)
    if not runs:
        print("No saved runs found in 'plots/' — run main.py or single_run.py first.")
        sys.exit(1)

    print("Available saved runs:")
    for i, p in enumerate(runs):
        print(f"  [{i}] {p.relative_to(PLOTS_DIR)}")

    raw = input("\nEnter number to replot (or paste a full path): ").strip()

    try:
        path = runs[int(raw)]
    except (ValueError, IndexError):
        path = Path(raw)

    title = input("Plot title (Enter to skip): ").strip()
    save_name = input(f"Save filename (Enter for '{path.stem}'): ").strip()
    if not save_name:
        save_name = path.stem

    fig, ax = replot_from_file(path, top_n=5, show_table=True)
    if title:
        ax.set_title(title)

    REPLOT_DIR.mkdir(exist_ok=True)
    save_path = REPLOT_DIR / f"{save_name}.png"
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()
