# core/analysi/fitting_quadratic_interpolation.py

import numpy as np
import pandas as pd

import core.preprocessing.geometry as geometry


def fit_quadratic_parametric_curve(
    flexion_mean,
    center,
    extension_mean,
    n_points=500,
):
    """
    Fit a parametric quadratic curve through three points.

    The three points are expressed in cm. The curve is parameterized by
    t in [0, 1]; the parameter value of the center point is proportional
    to its distance from the flexion point along the polyline
    flexion -> center -> extension. The x(t) and y(t) quadratics are
    obtained by solving the exact interpolation system.

    Parameters
    ----------
    flexion_mean : array-like
        Mean flexion position (x_cm, y_cm), corresponding to t = 0.

    center : array-like
        Center position (x_cm, y_cm) the curve passes through.

    extension_mean : array-like
        Mean extension position (x_cm, y_cm), corresponding to t = 1.

    n_points : int, optional
        Number of points used to sample the curve. Must be at least 3.
        Defaults to 500.

    Returns
    -------
    dict
        Dictionary containing:

        x
            x coordinates of the sampled curve.
        y
            y coordinates of the sampled curve.
        length
            Arc length of the curve, in cm.
        coefficients_x
            Coefficients (a, b, c) of x(t) = a*t² + b*t + c.
        coefficients_y
            Coefficients (a, b, c) of y(t) = a*t² + b*t + c.
        flexion_mean
        center
        extension_mean
            The three input points.

    Raises
    ------
    ValueError
        If n_points is smaller than 3 or if the three points
        are coincident.
    """

    point_0 = np.asarray(flexion_mean, dtype=float)

    point_1 = np.asarray(center, dtype=float)

    point_2 = np.asarray(extension_mean, dtype=float)

    if n_points < 3:
        raise ValueError("n_points must be at least 3.")

    distance_01 = np.linalg.norm(point_1 - point_0)

    distance_12 = np.linalg.norm(point_2 - point_1)

    total_distance = distance_01 + distance_12

    if total_distance == 0:
        raise ValueError("The three points cannot be coincident.")

    t_known = np.array([0.0, distance_01 / total_distance, 1.0])

    basis = np.column_stack([t_known**2, t_known, np.ones(3)])

    coefficients_x = np.linalg.solve(basis, np.array([point_0[0], point_1[0], point_2[0]]))

    coefficients_y = np.linalg.solve(basis, np.array([point_0[1], point_1[1], point_2[1]]))

    t = np.linspace(0.0, 1.0, n_points)

    x_curve = coefficients_x[0] * t**2 + coefficients_x[1] * t + coefficients_x[2]

    y_curve = coefficients_y[0] * t**2 + coefficients_y[1] * t + coefficients_y[2]

    dx_dt = np.gradient(x_curve, t)

    dy_dt = np.gradient(y_curve, t)

    arc_length = np.trapezoid(np.hypot(dx_dt, dy_dt), t)

    return {
        "x": x_curve,
        "y": y_curve,
        "length": float(arc_length),
        "coefficients_x": tuple(coefficients_x),
        "coefficients_y": tuple(coefficients_y),
        "flexion_mean": tuple(point_0),
        "center": tuple(point_1),
        "extension_mean": tuple(point_2),
    }


def fit_quadratic_parametric_curve_safe(
    flexion_mean,
    center,
    extension_mean,
    n_points=500,
):
    """
    Fit a parametric quadratic curve through three points,
    returning NaN values when a unique curve cannot be determined.

    The three points are expressed in cm. The curve is parameterized
    by t in [0, 1], with the parameter value of the center point
    proportional to its distance from the flexion point along the
    polyline flexion -> center -> extension.

    Unlike fit_quadratic_parametric_curve(), this function handles
    cases in which the center point coincides with the flexion or
    extension point. In these cases, the three-point interpolation
    problem does not define a unique quadratic curve. Instead of
    raising a Singular Matrix error, the function returns NaN for
    the curve coefficients, sampled coordinates, and arc length.

    Parameters
    ----------
    flexion_mean : array-like
        Mean flexion position (x_cm, y_cm), corresponding to t = 0.

    center : array-like
        Center position (x_cm, y_cm) that the curve should pass through.

    extension_mean : array-like
        Mean extension position (x_cm, y_cm), corresponding to t = 1.

    n_points : int, optional
        Number of points used to sample the curve. Must be at least 3.
        Defaults to 500.

    Returns
    -------
    dict
        Dictionary containing:

        x
            x coordinates of the sampled curve. Contains NaN values
            if a unique quadratic curve cannot be determined.

        y
            y coordinates of the sampled curve. Contains NaN values
            if a unique quadratic curve cannot be determined.

        length
            Arc length of the curve in cm. NaN if the curve cannot
            be uniquely determined.

        coefficients_x
            Coefficients (a, b, c) of x(t) = a*t² + b*t + c.
            Contains NaN values if the curve cannot be determined.

        coefficients_y
            Coefficients (a, b, c) of y(t) = a*t² + b*t + c.
            Contains NaN values if the curve cannot be determined.

        flexion_mean
            Input flexion point.

        center
            Input center point.

        extension_mean
            Input extension point.

    Raises
    ------
    ValueError
        If n_points is smaller than 3.
    """

    point_0 = np.asarray(flexion_mean, dtype=float)

    point_1 = np.asarray(center, dtype=float)

    point_2 = np.asarray(extension_mean, dtype=float)

    if n_points < 3:
        raise ValueError("n_points must be at least 3.")

    distance_01 = np.linalg.norm(point_1 - point_0)

    distance_12 = np.linalg.norm(point_2 - point_1)

    total_distance = distance_01 + distance_12

    if distance_01 == 0 or distance_12 == 0:
        return {
            "x": np.full(n_points, np.nan),
            "y": np.full(n_points, np.nan),
            "length": np.nan,
            "coefficients_x": (np.nan, np.nan, np.nan),
            "coefficients_y": (np.nan, np.nan, np.nan),
            "flexion_mean": tuple(point_0),
            "center": tuple(point_1),
            "extension_mean": tuple(point_2),
        }

    t_known = np.array([0.0, distance_01 / total_distance, 1.0])

    basis = np.column_stack([t_known**2, t_known, np.ones(3)])

    coefficients_x = np.linalg.solve(basis, np.array([point_0[0], point_1[0], point_2[0]]))

    coefficients_y = np.linalg.solve(basis, np.array([point_0[1], point_1[1], point_2[1]]))

    t = np.linspace(0.0, 1.0, n_points)

    x_curve = coefficients_x[0] * t**2 + coefficients_x[1] * t + coefficients_x[2]

    y_curve = coefficients_y[0] * t**2 + coefficients_y[1] * t + coefficients_y[2]

    dx_dt = np.gradient(x_curve, t)

    dy_dt = np.gradient(y_curve, t)

    arc_length = np.trapezoid(np.hypot(dx_dt, dy_dt), t)

    return {
        "x": x_curve,
        "y": y_curve,
        "length": float(arc_length),
        "coefficients_x": tuple(coefficients_x),
        "coefficients_y": tuple(coefficients_y),
        "flexion_mean": tuple(point_0),
        "center": tuple(point_1),
        "extension_mean": tuple(point_2),
    }


def get_pattern_means(
    df_mean, flexion_pattern, extension_pattern, pattern_column="pattern_pair", x_column="x", y_column="y", cell_size_cm=4
):
    """
    Return mean flexion and extension positions.

    The dataframe coordinates are in grid-cell units.
    The returned coordinates are converted to cm.

    Parameters
    ----------
    df_mean : pandas.DataFrame
        Data containing the pattern, x and y columns.

    flexion_pattern : str
        Pattern pair used as flexion reference.

    extension_pattern : str
        Pattern pair used as extension reference.

    pattern_column : str, optional
        Name of the column containing the pattern pairs.
        Defaults to "pattern_pair".

    x_column : str, optional
        Name of the column containing the x coordinates. Defaults to "x".

    y_column : str, optional
        Name of the column containing the y coordinates. Defaults to "y".

    cell_size_cm : float, optional
        Side length of a grid cell, in cm. Defaults to 4.

    Returns
    -------
    flexion_mean : tuple of float
        Mean flexion position (x_cm, y_cm).

    extension_mean : tuple of float
        Mean extension position (x_cm, y_cm).

    Raises
    ------
    ValueError
        If the dataframe does not contain both the flexion and the
        extension pattern.
    """

    flexion = df_mean[df_mean[pattern_column] == flexion_pattern]

    extension = df_mean[df_mean[pattern_column] == extension_pattern]

    if flexion.empty or extension.empty:
        raise ValueError("The dataframe must contain both " "flexion (000_111) and extension (111_000).")

    flexion_mean_cell = (
        float(flexion[x_column].mean()),
        float(flexion[y_column].mean()),
    )

    extension_mean_cell = (
        float(extension[x_column].mean()),
        float(extension[y_column].mean()),
    )

    flexion_mean = geometry.grid_point_to_cm(*flexion_mean_cell, cell_size_cm)

    extension_mean = geometry.grid_point_to_cm(*extension_mean_cell, cell_size_cm)

    return (flexion_mean, extension_mean)


def reconstruct_aligned_curves(df, output_file, n_points=101):
    """Reconstruct and align individual quadratic trajectories to a common center.

    Evaluates each subject-level quadratic parametric curve at uniformly
    spaced parameter values between t = 0 and t = 1. Estimates the parameter
    value corresponding to the center point from the relative Euclidean
    distances between the flexion, center, and extension points, then
    translates the reconstructed curve so that this center point coincides
    with the common reference position (0, 0). Saves the sampled aligned
    trajectories to an Excel worksheet.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing subject-level quadratic coefficients and the
        coordinates of the flexion, center, and extension points. Must include
        the columns ``subject``, ``posture``, ``duration``, ``a_x``, ``b_x``,
        ``c_x``, ``a_y``, ``b_y``, ``c_y``, ``flexion_x_cm``,
        ``flexion_y_cm``, ``center_x_cm``, ``center_y_cm``,
        ``extension_x_cm``, and ``extension_y_cm``.
    output_file : str or pathlib.Path
        Path to an existing Excel workbook in which the
        ``Reconstructed_aligned_curves`` worksheet will be created or replaced.
    n_points : int, optional
        Number of uniformly spaced parameter values between t = 0 and t = 1
        used to sample each reconstructed trajectory. Defaults to 101.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the sampled aligned trajectories, with one row
        per subject, posture, duration, and parameter value. Includes the
        columns ``subject``, ``posture``, ``duration``, ``t``, ``x_cm``,
        and ``y_cm``.

    Notes
    -----
    The alignment removes the positional offset between trajectories by
    translating each curve so that its estimated center point is at (0, 0).
    It does not rotate, scale, or otherwise deform the curves.

    Curves for which the sum of the distances from flexion to center and
    center to extension is zero are skipped because the center parameter
    value cannot be determined.

    The center parameter is estimated from the distances between the three
    defining points; it is not independently optimized from the quadratic
    coefficients.
    """
    t_values = np.linspace(0, 1, n_points)
    curves = []

    for _, row in df.iterrows():
        # Ricostruisci la curva dalle equazioni quadratiche
        x = row["a_x"] * t_values**2 + row["b_x"] * t_values + row["c_x"]
        y = row["a_y"] * t_values**2 + row["b_y"] * t_values + row["c_y"]

        # Calcola il parametro t del punto centrale
        flexion = np.array([row["flexion_x_cm"], row["flexion_y_cm"]])
        center = np.array([row["center_x_cm"], row["center_y_cm"]])
        extension = np.array([row["extension_x_cm"], row["extension_y_cm"]])

        distance_01 = np.linalg.norm(flexion - center)
        distance_12 = np.linalg.norm(center - extension)
        total_distance = distance_01 + distance_12

        if total_distance == 0:
            continue

        t_center = distance_01 / total_distance

        if row["subject"] == "S02" and row["posture"] == 70 and row["duration"] == 1:

            x_center = row["a_x"] * t_center**2 + row["b_x"] * t_center + row["c_x"]

            y_center = row["a_y"] * t_center**2 + row["b_y"] * t_center + row["c_y"]

            print("\n--- CONTROLLO S02 / 70° / 1 s ---")

            print("Flessione:")
            print("  Excel:", row["flexion_x_cm"], row["flexion_y_cm"])
            print("  Curva:", x[0], y[0])

            print("Centro:")
            print("  Excel:", row["center_x_cm"], row["center_y_cm"])
            print("  Curva:", x_center, y_center)

            print("Estensione:")
            print("  Excel:", row["extension_x_cm"], row["extension_y_cm"])
            print("  Curva:", x[-1], y[-1])

            print("t_center:", t_center)

        # Trasla la curva affinché il centro coincida con (0, 0)
        x_center = row["a_x"] * t_center**2 + row["b_x"] * t_center + row["c_x"]
        y_center = row["a_y"] * t_center**2 + row["b_y"] * t_center + row["c_y"]

        x_aligned = x - x_center
        y_aligned = y - y_center

        for i, t in enumerate(t_values):
            curves.append(
                {
                    "subject": row["subject"],
                    "posture": row["posture"],
                    "duration": row["duration"],
                    "t": t,
                    "x_cm": x_aligned[i],
                    "y_cm": y_aligned[i],
                }
            )

    results_df = pd.DataFrame(curves)
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        results_df.to_excel(writer, sheet_name="Reconstructed_aligned_curves", index=False)
    return results_df
