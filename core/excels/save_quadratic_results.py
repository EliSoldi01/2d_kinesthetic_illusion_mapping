# core/excels/save_quadratic_results.py

import pandas as pd
import numpy as np
import core.preprocessing.geometry as geometry
import core.analysis.fitting_quadratic_interpolation as fqi
import core.preprocessing.prepare_data as prepdata

def save_quadratic_results(df, protocol, output_file, flexion_pattern, extension_pattern, durations=None):
    """Fit quadratic parametric curves for each subject and stimulation duration.

    For each subject and duration, computes the mean perceived positions
    associated with flexion and extension patterns, uses the protocol's
    starting position as the center point, fits a quadratic parametric curve,
    and saves the curve coefficients, arc length, and defining points to Excel.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed experimental data containing subject identifiers,
        stimulation durations, pattern pairs, and perceived positions.
    protocol : dict
        Experimental protocol configuration containing the grid definition,
        starting cell, cell size, and initial elbow angle.
    output_file : str or pathlib.Path
        Path to the Excel file where the results will be saved.
    flexion_pattern : str
        Identifier of the stimulation pattern associated with the flexion
        illusion.
    extension_pattern : str
        Identifier of the stimulation pattern associated with the extension
        illusion.
    durations : sequence of int or float, optional
        Stimulation durations, in seconds, for which curves will be fitted.
        If None, uses the protocol's configured durations, excluding the
        first entry.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per subject and duration, including
        posture, arc length, quadratic coefficients, and the coordinates
        of the flexion, center, and extension points.
    """

    x_cell, y_cell = protocol["grid"]["start_cell"]
    center_mean = geometry.grid_point_to_cm(
        x_cell,
        y_cell,
        protocol["grid"]["cell_size_cm"]
    )

    posture = protocol["initial_angle"]

    if durations is None:
        durations = protocol["blocks"]["durations"][1:]

    results = []
    subjects = sorted(df["subject"].dropna().unique())

    for subject in subjects:
        df_subject = df[df["subject"] == subject]

        for duration in durations:

            df_mean = prepdata.prepare_quadratic_interpolation_data(
                df=df_subject,
                duration=duration,
                protocol=protocol
            )

            flexion = df_mean[df_mean["pattern_pair"] == flexion_pattern]
            extension = df_mean[df_mean["pattern_pair"] == extension_pattern]

            flexion_mean = (float(flexion["x"].mean()), float(flexion["y"].mean()))
            extension_mean = (float(extension["x"].mean()), float(extension["y"].mean()),)
            
            result = fqi.fit_quadratic_parametric_curve_safe(
                flexion_mean=flexion_mean,
                center=center_mean,
                extension_mean=extension_mean,
            )

            a_x, b_x, c_x = result["coefficients_x"]
            a_y, b_y, c_y = result["coefficients_y"]

            results.append({
                "subject": subject,
                "posture": posture,
                "duration": duration,
                "arc_length_cm": result["length"],
                "a_x": a_x,
                "b_x": b_x,
                "c_x": c_x,
                "a_y": a_y,
                "b_y": b_y,
                "c_y": c_y,
                "flexion_x_cm": flexion_mean[0],
                "flexion_y_cm": flexion_mean[1],
                "center_x_cm": center_mean[0],
                "center_y_cm": center_mean[1],
                "extension_x_cm": extension_mean[0],
                "extension_y_cm": extension_mean[1],
            })

    results_df = pd.DataFrame(results)

    results_df.to_excel(
        output_file,
        index=False
    )
    return results_df

def save_arc_length_data(df, output_file):
    """Calculate descriptive statistics of fitted arc lengths by posture and duration.

    Groups the fitted arc lengths by initial elbow posture and stimulation
    duration, computes the number of valid observations, mean arc length,
    and standard deviation, and saves the results to the ``Arc_Length``
    worksheet of an existing Excel workbook.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the fitted arc lengths and the corresponding
        posture and duration for each observation. Must include the columns
        ``posture``, ``duration``, and ``arc_length_cm``.
    output_file : str or pathlib.Path
        Path to an existing Excel workbook in which the ``Arc_Length``
        worksheet will be created or replaced.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per posture and duration, with the
        number of valid arc lengths, their mean, and their standard deviation.
    """

    postures = sorted(df["posture"].dropna().unique())
    durations = sorted(df["duration"].dropna().unique())

    result = []
    for posture in postures:
        df_posture = df[df["posture"] == posture]
        for duration in durations:
            df_duration = df_posture[df_posture["duration"] == duration]

            arc_lengths = df_duration["arc_length_cm"].dropna()

            result.append({
                "posture": posture,
                "duration": duration,
                "n_valid": len(arc_lengths),
                "Mean arc_length_cm": arc_lengths.mean(),
                "Std arc_length_cm": arc_lengths.std()
            })

    results_df = pd.DataFrame(result)
    
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_df.to_excel(writer, sheet_name='Arc_Length', index=False)

    return results_df

def calculate_mean_curves(df_curves, output_file):
    """Calculate mean aligned trajectories from individual reconstructed curves.

    Averages the x and y coordinates across subjects at each parameter value
    t, separately for each posture and stimulation duration. Saves the
    resulting mean trajectories to the ``Mean_curves_A`` worksheet.

    Parameters
    ----------
    df_curves : pandas.DataFrame
        DataFrame containing the sampled individual trajectories. Must include
        the columns ``posture``, ``duration``, ``t``, ``x_cm``, ``y_cm``,
        and ``subject``.
    output_file : str or pathlib.Path
        Path to an existing Excel workbook in which the ``Mean_curves_A``
        worksheet will be created or replaced.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the mean x and y coordinates and the number
        of unique subjects contributing to each point, grouped by posture,
        duration, and parameter value t.
    """

    result = []

    grouped = df_curves.groupby(
        ["posture", "duration", "t"],
        as_index=False
    )

    for (posture, duration, t), group in grouped:

        result.append({
            "posture": posture,
            "duration": duration,
            "t": t,
            "x_mean_cm": group["x_cm"].mean(),
            "y_mean_cm": group["y_cm"].mean(),
            "n_subjects": group["subject"].nunique(),
        })

    results_df = pd.DataFrame(result)

    results_df = results_df.sort_values(
        ["posture", "duration", "t"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace"
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name="Mean_curves_A",
            index=False
        )

    return results_df

def calculate_mean_point_curves(df, output_file, n_points=101):
    """Reconstruct aligned mean trajectories from group-averaged defining points.

    For each posture and stimulation duration, averages the flexion, center,
    and extension coordinates across subjects, fits a quadratic parametric
    curve through these group-level points, and translates the curve so that
    its center point is at the common reference position (0, 0). Saves the
    sampled mean trajectories and their aligned quadratic coefficients to
    separate worksheets in the Excel workbook.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing subject-level quadratic results and the
        coordinates of the flexion, center, and extension points. Must include
        ``posture``, ``duration``, ``subject``, ``flexion_x_cm``,
        ``flexion_y_cm``, ``center_x_cm``, ``center_y_cm``,
        ``extension_x_cm``, and ``extension_y_cm``.
    output_file : str or pathlib.Path
        Path to an existing Excel workbook in which the ``Mean_curves_B``
        and ``Mean_curve_coefficients`` worksheets will be created or replaced.
    n_points : int, optional
        Number of parameter values sampled between t = 0 and t = 1 for
        each reconstructed curve. Defaults to 101.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing:
        - results_df: sampled, aligned mean trajectories, including posture,
          duration, parameter value, x and y coordinates, and number of subjects.
        - coefficients_df: quadratic coefficients of the aligned mean
          trajectories, together with posture, duration, and number of subjects.

    Notes
    -----
    Curves for which the quadratic fitting procedure returns an invalid
    length are skipped. Translation changes the constant coefficients
    c_x and c_y but leaves the quadratic and linear coefficients unchanged.
    """

    result = []
    coefficients_result = []

    grouped = df.groupby(
        ["posture", "duration"]
    )

    for (posture, duration), group in grouped:

        flexion_mean = (
            group["flexion_x_cm"].mean(),
            group["flexion_y_cm"].mean(),
        )

        center_mean = (
            group["center_x_cm"].mean(),
            group["center_y_cm"].mean(),
        )

        extension_mean = (
            group["extension_x_cm"].mean(),
            group["extension_y_cm"].mean(),
        )

        curve = fqi.fit_quadratic_parametric_curve_safe(
            flexion_mean=flexion_mean,
            center=center_mean,
            extension_mean=extension_mean,
            n_points=n_points,
        )

        if np.isnan(curve["length"]):
            continue

        # Coefficienti della curva prima dell'allineamento
        a_x, b_x, c_x = curve["coefficients_x"]
        a_y, b_y, c_y = curve["coefficients_y"]

        # Trova il valore di t corrispondente al centro
        distance_01 = np.linalg.norm(
            np.array(center_mean) - np.array(flexion_mean)
        )

        distance_12 = np.linalg.norm(
            np.array(extension_mean) - np.array(center_mean)
        )

        t_center = distance_01 / (
            distance_01 + distance_12
        )

        # Coordinate del centro sulla curva
        x_center = (
            a_x * t_center**2
            + b_x * t_center
            + c_x
        )

        y_center = (
            a_y * t_center**2
            + b_y * t_center
            + c_y
        )

        # Allineamento al centro comune (0, 0)
        x_aligned = curve["x"] - x_center
        y_aligned = curve["y"] - y_center

        # Dopo una traslazione:
        # a e b rimangono invariati,
        # mentre c viene traslato.
        c_x_aligned = c_x - x_center
        c_y_aligned = c_y - y_center

        # Salva i coefficienti della curva media allineata
        coefficients_result.append({
            "posture": posture,
            "duration": duration,
            "a_x": a_x,
            "b_x": b_x,
            "c_x": c_x_aligned,
            "a_y": a_y,
            "b_y": b_y,
            "c_y": c_y_aligned,
            "n_subjects": group["subject"].nunique(),
        })

        t_values = np.linspace(
            0,
            1,
            n_points
        )

        for t, x, y in zip(
            t_values,
            x_aligned,
            y_aligned
        ):
            result.append({
                "posture": posture,
                "duration": duration,
                "t": t,
                "x_mean_cm": x,
                "y_mean_cm": y,
                "n_subjects": group["subject"].nunique(),
            })

    results_df = pd.DataFrame(result)

    results_df = results_df.sort_values(
        ["posture", "duration", "t"]
    ).reset_index(drop=True)

    coefficients_df = pd.DataFrame(
        coefficients_result
    )

    coefficients_df = coefficients_df.sort_values(
        ["posture", "duration"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace"
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name="Mean_curves_B",
            index=False
        )

        coefficients_df.to_excel(
            writer,
            sheet_name="Mean_curve_coefficients",
            index=False
        )

    return results_df, coefficients_df


def calculate_curve_distances(df_curves, output_file, sheet_name):
    """Calculate pairwise spatial distances between aligned mean trajectories.

    For each stimulation duration, compares the aligned mean trajectories
    for the posture pairs 70°–90°, 70°–120°, and 90°–120°. Computes the
    Euclidean distance between corresponding points sampled at matching
    parameter values t, then summarizes the distances using their mean
    and maximum. Saves the results to the specified worksheet.

    Parameters
    ----------
    df_curves : pandas.DataFrame
        DataFrame containing aligned mean trajectories. Must include the
        columns ``posture``, ``duration``, ``t``, ``x_mean_cm``, and
        ``y_mean_cm``.
    output_file : str or pathlib.Path
        Path to an existing Excel workbook in which the results will be saved.
    sheet_name : str
        Name of the worksheet to create or replace with the distance results.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per posture pair and duration, with
        the mean and maximum pointwise Euclidean distances in centimeters.

    Raises
    ------
    ValueError
        If two trajectories being compared have different sampled values
        of the parameter t.

    Notes
    -----
    The distances are computed between corresponding parameter values,
    not by finding the minimum distance between arbitrary points on the
    two curves. Comparisons are skipped when either trajectory is missing.
    """

    result = []

    posture_pairs = [
        (70, 90),
        (70, 120),
        (90, 120),
    ]

    durations = sorted(
        df_curves["duration"].dropna().unique()
    )

    for duration in durations:

        df_duration = df_curves[
            df_curves["duration"] == duration
        ]

        for posture_1, posture_2 in posture_pairs:

            curve_1 = df_duration[
                df_duration["posture"] == posture_1
            ].sort_values("t")

            curve_2 = df_duration[
                df_duration["posture"] == posture_2
            ].sort_values("t")

            if curve_1.empty or curve_2.empty:
                continue

            # Controllo che le due curve abbiano gli stessi valori di t
            if not np.allclose(
                curve_1["t"].values,
                curve_2["t"].values
            ):
                raise ValueError(
                    "The two curves do not have matching t values."
                )

            distance = np.sqrt(
                (
                    curve_1["x_mean_cm"].values
                    - curve_2["x_mean_cm"].values
                )**2
                +
                (
                    curve_1["y_mean_cm"].values
                    - curve_2["y_mean_cm"].values
                )**2
            )

            result.append({
                "posture_1": posture_1,
                "posture_2": posture_2,
                "duration": duration,
                "mean_distance_cm": distance.mean(),
                "max_distance_cm": distance.max(),
            })

    results_df = pd.DataFrame(result)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace"
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

    return results_df

def calculate_curve_orientation(df_coefficients, output_file, n_points=101):
    """Calculate tangent orientation along aligned mean parametric trajectories.

    Uses the first derivatives of the quadratic x(t) and y(t) functions
    to calculate the tangent orientation at uniformly sampled parameter
    values. Applies phase unwrapping to avoid artificial discontinuities
    caused by angle representations crossing ±180°, and computes the change
    in orientation relative to the initial tangent. Saves the results to
    the ``Curve_orientation`` worksheet.

    Parameters
    ----------
    df_coefficients : pandas.DataFrame
        DataFrame containing the quadratic coefficients for each mean curve.
        Must include ``posture``, ``duration``, ``a_x``, ``b_x``, ``a_y``,
        ``b_y``, and ``n_subjects``.
    output_file : str or pathlib.Path
        Path to an existing Excel workbook in which the
        ``Curve_orientation`` worksheet will be created or replaced.
    n_points : int, optional
        Number of uniformly spaced parameter values between t = 0 and t = 1
        at which tangent orientations are evaluated. Defaults to 101.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the tangent orientation and its change relative
        to the initial orientation, in degrees, for each posture, duration,
        and parameter value.

    Notes
    -----
    The orientation is calculated as atan2(dy/dt, dx/dt), expressed in
    degrees. The orientation change describes the change in tangent angle
    along the curve; it is not the geometric curvature.
    """

    result = []

    t_values = np.linspace(0, 1, n_points)

    for _, row in df_coefficients.iterrows():

        a_x = row["a_x"]
        b_x = row["b_x"]

        a_y = row["a_y"]
        b_y = row["b_y"]

        # Derivate della curva parametrica
        dx_dt = 2 * a_x * t_values + b_x
        dy_dt = 2 * a_y * t_values + b_y

        # Orientamento della tangente
        theta = np.degrees(
            np.arctan2(dy_dt, dx_dt)
        )

        # Evita discontinuità dovute al passaggio ±180°
        theta_unwrapped = np.degrees(
            np.unwrap(np.radians(theta))
        )

        theta_initial = theta_unwrapped[0]

        delta_theta = theta_unwrapped - theta_initial

        for t, angle, delta in zip(
            t_values,
            theta_unwrapped,
            delta_theta
        ):
            result.append({
                "posture": row["posture"],
                "duration": row["duration"],
                "t": t,
                "theta_deg": angle,
                "delta_theta_deg": delta,
                "n_subjects": row["n_subjects"],
            })

    results_df = pd.DataFrame(result)

    results_df = results_df.sort_values(
        ["posture", "duration", "t"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace"
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name="Curve_orientation",
            index=False
        )

    return results_df


def calculate_orientation_change(df_orientation, output_file):
    """Summarize the tangent orientation change between the start and end of each curve.

    Groups the sampled orientation data by posture and stimulation duration,
    orders the samples by parameter value, and calculates the initial and
    final tangent orientations and their difference. Saves the summary to
    the ``Orientation_change`` worksheet.

    Parameters
    ----------
    df_orientation : pandas.DataFrame
        DataFrame containing the sampled tangent orientations. Must include
        ``posture``, ``duration``, ``t``, ``theta_deg``, and ``n_subjects``.
    output_file : str or pathlib.Path
        Path to an existing Excel workbook in which the
        ``Orientation_change`` worksheet will be created or replaced.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per posture and duration, with the
        initial tangent orientation, final tangent orientation, total
        orientation change, and number of subjects contributing to the
        corresponding mean curve.

    Notes
    -----
    The orientation change is defined as the unwrapped final tangent angle
    minus the unwrapped initial tangent angle, in degrees. It represents
    the net change in tangent orientation over the parameter interval
    t = 0 to t = 1, not the geometric curvature.
    """

    result = []

    grouped = df_orientation.groupby(
        ["posture", "duration"]
    )

    for (posture, duration), group in grouped:

        group = group.sort_values("t")

        theta_initial = group["theta_deg"].iloc[0]
        theta_final = group["theta_deg"].iloc[-1]

        delta_theta = (
            theta_final
            - theta_initial
        )

        result.append({
            "posture": posture,
            "duration": duration,
            "theta_initial_deg": theta_initial,
            "theta_final_deg": theta_final,
            "delta_theta_deg": delta_theta,
            "n_subjects": group["n_subjects"].iloc[0],
        })

    results_df = pd.DataFrame(result)

    results_df = results_df.sort_values(
        ["posture", "duration"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace"
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name="Orientation_change",
            index=False
        )

    return results_df

