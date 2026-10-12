# scripts/analyze_posture_trajectories.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import config as cfg
from core.data_io.load_data import load_data_file, load_protocol
from core.preprocessing.prepare_data import prepare_data
import core.excels.save_quadratic_results as save_res
import core.utils.plot_config as plot_config
from core.analysis.fitting_quadratic_interpolation import reconstruct_aligned_curves

# Configuration
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "Results" / "angles_comparison"

# This workbook must contain the individual quadratic-curve results
# for all postures being compared (70°, 90°, and 120°).
QUADRATIC_RESULTS_FILE = RESULTS_DIR / "quadratic_results.xlsx"

CURVE_ANALYSIS_FILE = RESULTS_DIR / "curve_analysis.xlsx"
MEAN_CURVES_FIGURE = RESULTS_DIR / "mean_curves.png"

POSTURE_COLORS = {
    70: "tab:blue",
    90: "tab:green",
    120: "tab:red",
}


def plot_single_original_curve(df, subject, posture, duration):
    """
    Plot the original quadratic trajectory for a specific subject and condition.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the fitted quadratic coefficients and the
        coordinates of the flexion, center, and extension points.
    subject : str or int
        Identifier of the subject whose trajectory is to be plotted.
    posture : int or float
        Initial elbow angle, in degrees.
    duration : int or float
        Stimulation duration, in seconds.

    Raises
    ------
    ValueError
        If no row matches the specified subject, posture, and duration.

    Returns
    -------
    None
        Displays the quadratic trajectory and its three defining points
        in a Matplotlib figure.
    """

    row = df[(df["subject"] == subject) & (df["posture"] == posture) & (df["duration"] == duration)]

    if row.empty:
        raise ValueError(f"Nessun risultato trovato per " f"{subject}, {posture}°, {duration} s.")

    row = row.iloc[0]

    t = np.linspace(0, 1, 500)

    x = row["a_x"] * t**2 + row["b_x"] * t + row["c_x"]

    y = row["a_y"] * t**2 + row["b_y"] * t + row["c_y"]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(x, y, label="Curva quadratica")

    ax.scatter(row["flexion_x_cm"], row["flexion_y_cm"], s=80, label="Flessione")

    ax.scatter(row["center_x_cm"], row["center_y_cm"], s=80, label="Centro")

    ax.scatter(row["extension_x_cm"], row["extension_y_cm"], s=80, label="Estensione")

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title(f"{subject} — {posture}° — {duration} s")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()

    plt.show()


def main():
    """
    Run the quadratic trajectory analysis across elbow postures.

    Loads and preprocesses the experimental data, fits quadratic curves for
    the current protocol, loads the combined quadratic results, and calculates
    arc lengths, aligned mean trajectories, pairwise trajectory distances,
    and changes in trajectory orientation. It also generates and saves a
    figure comparing the aligned mean trajectories across postures.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Saves the quadratic results, trajectory metrics, and mean-trajectory
        figure to the configured results directory.
    """
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df_main = load_data_file(cfg.DATA_PATH, "Trials")
    df_subjects = load_data_file(cfg.DATA_PATH, "Subjects")

    # --------------------------------------------------------
    # Load protocol
    # --------------------------------------------------------

    protocol = load_protocol(cfg.PROTOCOL_PATH)

    df = prepare_data(df_main, df_subjects, protocol, cfg.DATA_PROCESSED_PATH)

    # --------------------------------------------------------
    # Save quadratic results
    # --------------------------------------------------------
    output_path = RESULTS_DIR
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"quadratic_results_test_{str(protocol["initial_angle"])}.xlsx")

    save_res.save_quadratic_results(
        df=df,
        protocol=protocol,
        output_file=output_file,
        flexion_pattern=cfg.FLEXION_PATTERN_FOR_QUADRATIC_INTERP,
        extension_pattern=cfg.EXTENSION_PATTERN_FOR_QUADRATIC_INTERP,
        durations=[1, 3, 6],
    )

    combined_results_file = QUADRATIC_RESULTS_FILE

    combined_df = load_data_file(combined_results_file)
    save_res.save_arc_length_data(combined_df, combined_results_file)

    print(f"[SUCCESS] Quadratic results saved to: {output_file}")

    curve_results_file = CURVE_ANALYSIS_FILE
    df_curves = reconstruct_aligned_curves(combined_df, curve_results_file)

    mean_curve_A = save_res.calculate_mean_curves(df_curves, curve_results_file)
    mean_curve_B, mean_curve_B_coefficients = save_res.calculate_mean_point_curves(combined_df, curve_results_file)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    plot_config.configure_plot_style()
    durations = protocol["blocks"]["durations"][1:]

    # 2. Loop corretto con zip
    for duration, ax in zip(durations, axes):

        df_plot = mean_curve_B[mean_curve_B["duration"] == duration]

        # Traccia ogni postura
        for posture, group_posture in df_plot.groupby("posture"):
            ax.plot(
                group_posture["x_mean_cm"],
                group_posture["y_mean_cm"],
                color=POSTURE_COLORS[posture],
                alpha=0.7,
            )

        # Punto centrale comune (fuori dal loop delle posture)
        ax.scatter(
            0,
            0,
            color="black",
            marker="x",
            s=80,
        )

        # Configurazione del singolo pannello
        ax.set_title(f"D = {duration} s")
        ax.set_xlabel("Aligned X (cm)")
        ax.set_aspect("equal")
        ax.set_ylim(-15, 15)
        ax.set_xlim(-15, 15)
        ax.invert_yaxis()
        ax.grid(True)

    # Etichetta asse Y solo sul primo pannello
    axes[0].set_ylabel("Aligned Y (cm)")

    # 3. Legenda delle posture
    legend_handles = [
        Line2D([0], [0], color=POSTURE_COLORS[70], lw=2, label="70°"),
        Line2D([0], [0], color=POSTURE_COLORS[90], lw=2, label="90°"),
        Line2D([0], [0], color=POSTURE_COLORS[120], lw=2, label="120°"),
        Line2D([0], [0], color="black", marker="x", linestyle="None", label="Start position (cm)"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Salva l'immagine
    plt.savefig(
        MEAN_CURVES_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    # Calcolo distanze
    save_res.calculate_curve_distances(mean_curve_B, curve_results_file, "Curve_distances")

    df_orientation = save_res.calculate_curve_orientation(mean_curve_B_coefficients, curve_results_file)
    save_res.calculate_orientation_change(df_orientation, curve_results_file)


if __name__ == "__main__":
    main()
