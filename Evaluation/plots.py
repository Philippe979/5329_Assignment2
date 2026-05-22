def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it before calling plot functions."
        ) from error
    return plt


def _save_or_show(plt, output_path: str = None):
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_accuracy_curves(
    epochs,
    clean_accs,
    robust_accs,
    title: str = "Clean vs Robust Accuracy",
    output_path: str = None,
):
    """
    Plot clean and robust accuracy curves.
    """
    plt = _get_plt()
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, clean_accs, label="Clean Accuracy")
    plt.plot(epochs, robust_accs, label="Robust Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    _save_or_show(plt, output_path)


def plot_gap_curve(
    epochs,
    gaps,
    title: str = "Robustness Gap",
    output_path: str = None,
):
    """
    Plot clean-robust gap curve.
    """
    plt = _get_plt()
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, gaps, label="Clean - Robust")
    plt.xlabel("Epoch")
    plt.ylabel("Gap")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    _save_or_show(plt, output_path)


def plot_trajectory(
    rows,
    title: str = "Dynamic Robustness Trajectory",
    output_path: str = None,
):
    """
    Plot CA(p), RA(p), and Delta(p) for one model.
    """
    plt = _get_plt()
    rows = sorted(rows, key=lambda row: float(row["progress"]))
    progress = [float(row["progress"]) for row in rows]
    clean_acc = [float(row["test_clean_acc"]) for row in rows]
    robust_acc = [float(row["test_robust_acc"]) for row in rows]
    gaps = [float(row["gap"]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(progress, clean_acc, marker="o", label="Clean Accuracy")
    plt.plot(progress, robust_acc, marker="o", label="FGSM Robust Accuracy")
    plt.plot(progress, gaps, marker="o", label="Clean-Robust Gap")
    plt.xlabel("Training Progress")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_or_show(plt, output_path)


def plot_multi_model_robustness(
    grouped_rows,
    title: str = "Robust Accuracy Across Models",
    output_path: str = None,
):
    """
    Plot RA(p) for multiple models.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(11, 6))
    model_count = max(len(grouped_rows), 1)
    cmap = plt.get_cmap("tab20")

    for index, (model_name, rows) in enumerate(grouped_rows.items()):
        rows = sorted(rows, key=lambda row: float(row["progress"]))
        progress = [float(row["progress"]) for row in rows]
        robust_acc = [float(row["test_robust_acc"]) for row in rows]
        design_type = rows[0].get("design_type", "")
        linestyle = {
            "CIFAR-friendly": "-",
            "General-adapted": "--",
            "MLP-adapted": "-.",
            "Transformer-adapted": ":",
        }.get(design_type, "-")
        ax.plot(
            progress,
            robust_acc,
            marker="o",
            markersize=4,
            linewidth=1.8,
            alpha=0.82,
            linestyle=linestyle,
            color=cmap(index % 20),
            label=model_name,
        )

    ax.set_xlabel("Training Progress")
    ax.set_ylabel("FGSM Robust Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    legend_cols = 1 if model_count <= 12 else 2
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        ncol=legend_cols,
        fontsize=8,
    )
    _save_or_show(plt, output_path)


def plot_gravity_collapse(
    rows,
    title: str = "Gravity Collapse Plot",
    output_path: str = None,
):
    """
    Plot the CA-RA trajectory. Larger points indicate a larger robustness gap.
    """
    plt = _get_plt()
    rows = sorted(rows, key=lambda row: float(row["progress"]))
    clean_acc = [float(row["test_clean_acc"]) for row in rows]
    robust_acc = [float(row["test_robust_acc"]) for row in rows]
    progress = [float(row["progress"]) for row in rows]
    gaps = [float(row["gap"]) for row in rows]
    sizes = [40 + 500 * gap for gap in gaps]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        clean_acc,
        robust_acc,
        c=progress,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )

    for i in range(len(rows) - 1):
        plt.annotate(
            "",
            xy=(clean_acc[i + 1], robust_acc[i + 1]),
            xytext=(clean_acc[i], robust_acc[i]),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "0.25"},
        )

    plt.xlabel("Clean Accuracy")
    plt.ylabel("FGSM Robust Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Training Progress")
    _save_or_show(plt, output_path)


def plot_3d_gravity_collapse(
    grouped_rows,
    title: str = "3D Gravity Collapse Map",
    output_path: str = None,
):
    """
    Plot all model trajectories in a shared 3D clean-robust-gap space.

    Axes:
        x = clean accuracy
        y = FGSM robust accuracy
        z = clean-robust gap

    Color encodes training progress and marker shape encodes model level.
    """
    plt = _get_plt()
    from matplotlib.lines import Line2D
    import math

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    level_markers = {
        "Level 1 - Baseline": "o",
        "Level 2 - Median": "^",
        "Level 3 - Strong": "s",
    }
    design_linestyles = {
        "CIFAR-friendly": "-",
        "General-adapted": "--",
        "MLP-adapted": "-.",
        "Transformer-adapted": ":",
    }
    model_cmap = plt.get_cmap("tab20")

    scatter_for_colorbar = None
    legend_handles = []

    for index, (model_name, rows) in enumerate(grouped_rows.items()):
        rows = sorted(rows, key=lambda row: float(row["progress"]))
        clean_acc = [float(row["test_clean_acc"]) for row in rows]
        robust_acc = [float(row["test_robust_acc"]) for row in rows]
        gaps = [float(row["gap"]) for row in rows]
        progress = [float(row["progress"]) for row in rows]
        level = rows[0].get("level", "")
        design_type = rows[0].get("design_type", "")
        marker = level_markers.get(level, "o")
        line_color = model_cmap(index % 20)
        linestyle = design_linestyles.get(design_type, "-")

        scatter_for_colorbar = ax.scatter(
            clean_acc,
            robust_acc,
            gaps,
            c=progress,
            cmap="viridis",
            marker=marker,
            s=45,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.plot(
            clean_acc,
            robust_acc,
            gaps,
            color=line_color,
            linestyle=linestyle,
            linewidth=1.9,
            alpha=0.9,
        )

        for i in range(len(rows) - 1):
            dx = clean_acc[i + 1] - clean_acc[i]
            dy = robust_acc[i + 1] - robust_acc[i]
            dz = gaps[i + 1] - gaps[i]
            ax.quiver(
                clean_acc[i],
                robust_acc[i],
                gaps[i],
                dx,
                dy,
                dz,
                color=line_color,
                arrow_length_ratio=0.18,
                linewidth=0.8,
                alpha=0.7,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=line_color,
                marker=marker,
                linestyle=linestyle,
                label=model_name,
                markersize=6,
            )
        )

    ax.set_xlabel("Clean Accuracy")
    ax.set_ylabel("FGSM Robust Accuracy")
    ax.set_zlabel("Clean-Robust Gap")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-55)
    legend_cols = 3 if len(legend_handles) <= 12 else 4
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8,
        ncol=legend_cols,
        frameon=True,
    )
    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=ax, shrink=0.62, pad=0.08)
        cbar.set_label("Training Progress")
    _save_or_show(plt, output_path)


def plot_fitted_collapse_surface(
    grouped_rows,
    title: str = "Fitted 3D Robustness Collapse Surface",
    output_path: str = None,
):
    """
    Plot an interpretive fitted surface of the overall collapse trend.

    Axes:
        x = training progress
        y = clean accuracy
        z = clean-robust gap

    The surface is generated by Gaussian kernel smoothing over all observed
    checkpoints. Raw points are overlaid to keep the fit auditable.
    """
    plt = _get_plt()
    import numpy as np

    progress_values = []
    clean_values = []
    gap_values = []
    for rows in grouped_rows.values():
        for row in rows:
            progress_values.append(float(row["progress"]))
            clean_values.append(float(row["test_clean_acc"]))
            gap_values.append(float(row["gap"]))

    if not progress_values:
        raise ValueError("No rows available for fitted collapse surface.")

    progress_values = np.array(progress_values)
    clean_values = np.array(clean_values)
    gap_values = np.array(gap_values)

    x_grid = np.linspace(progress_values.min(), progress_values.max(), 70)
    y_grid = np.linspace(clean_values.min(), clean_values.max(), 70)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = np.zeros_like(xx)

    sigma_x = 0.12
    sigma_y = 0.08
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            weights = np.exp(
                -0.5 * (
                    ((progress_values - xx[i, j]) / sigma_x) ** 2
                    + ((clean_values - yy[i, j]) / sigma_y) ** 2
                )
            )
            weight_sum = weights.sum()
            zz[i, j] = (weights * gap_values).sum() / weight_sum if weight_sum > 0 else 0.0

    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap="viridis",
        alpha=0.82,
        linewidth=0.15,
        edgecolor="0.35",
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.scatter(
        progress_values,
        clean_values,
        gap_values,
        c=gap_values,
        cmap="viridis",
        s=16,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.75,
    )

    ax.set_xlabel("Training Progress")
    ax.set_ylabel("Clean Accuracy")
    ax.set_zlabel("Clean-Robust Gap")
    ax.set_title(title)
    ax.view_init(elev=28, azim=-128)
    cbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label("Fitted Collapse Intensity")
    _save_or_show(plt, output_path)


def plot_phase_timeline(
    grouped_rows,
    title: str = "Robustness Phase Timeline",
    output_path: str = None,
):
    """
    Plot phase assignments for every model across training progress.
    """
    plt = _get_plt()
    import matplotlib.patches as mpatches

    phase_colors = {
        "Pre-learning": "#9d9d9d",
        "Robustness Formation": "#4c78a8",
        "Robustness Maturation": "#72b7b2",
        "Robustness Stabilization / Peak": "#59a14f",
        "Early Robustness Decay": "#8cd17d",
        "Late Degradation": "#e15759",
        "Late Divergence": "#b07aa1",
        "Late Stabilization": "#f28e2b",
    }
    model_names = list(grouped_rows.keys())
    model_count = max(len(model_names), 1)

    fig_height = max(5.2, 0.48 * model_count + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    for y_index, model_name in enumerate(model_names):
        rows = sorted(grouped_rows[model_name], key=lambda row: float(row["progress"]))
        for i, row in enumerate(rows):
            start = float(row["progress"])
            if i < len(rows) - 1:
                end = float(rows[i + 1]["progress"])
            else:
                end = 1.0
            width = max(end - start, 0.012)
            phase = row.get("phase", "")
            ax.barh(
                y_index,
                width,
                left=start,
                height=0.55,
                color=phase_colors.get(phase, "0.7"),
                edgecolor="white",
                linewidth=0.7,
            )

    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Training Progress")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    legend_handles = [
        mpatches.Patch(color=color, label=phase)
        for phase, color in phase_colors.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
    )
    _save_or_show(plt, output_path)


def plot_blackhole_gravity_collapse(
    grouped_rows,
    title: str = "Black-Hole Style Robustness Collapse Map",
    output_path: str = None,
):
    """
    Plot a derived cylindrical 3D collapse map.

    Coordinates:
        theta = model identity
        radius = robustness retention = RA(t) / RA_peak
        z = - clean-robust gap

    This is an interpretive visualization for trajectory direction, not a raw
    CA/RA/gap coordinate plot.
    """
    plt = _get_plt()
    import math
    import numpy as np
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    model_names = list(grouped_rows.keys())
    level_colors = {
        "Level 1 - VGG-style CNN": "#1f77b4",
        "Level 2 - Residual CNN": "#2ca02c",
        "Level 3 - Wide Residual CNN": "#d62728",
    }
    phase_markers = {
        "Pre-learning": "x",
        "Robustness Formation": "o",
        "Robustness Stabilization / Peak": "*",
        "Late Degradation": "v",
        "Late Divergence": "s",
        "Late Stabilization": "D",
    }

    # Draw a faint event-horizon style guide ring near the early robust region.
    theta_grid = np.linspace(0, 2 * math.pi, 240)
    ring_x = np.cos(theta_grid)
    ring_y = np.sin(theta_grid)
    ax.plot(ring_x, ring_y, [0.0] * len(theta_grid), color="0.2", lw=1.0, alpha=0.25)
    ax.plot(0.35 * ring_x, 0.35 * ring_y, [-0.85] * len(theta_grid), color="0.1", lw=1.0, alpha=0.18)

    legend_handles = []

    for index, model_name in enumerate(model_names):
        rows = sorted(grouped_rows[model_name], key=lambda row: float(row["progress"]))
        theta = 2 * math.pi * index / len(model_names)
        robust_acc = [float(row["test_robust_acc"]) for row in rows]
        gaps = [float(row["gap"]) for row in rows]
        progress = [float(row["progress"]) for row in rows]
        phases = [row.get("phase", "") for row in rows]
        level = rows[0].get("level", "")
        color = level_colors.get(level, "0.4")
        ra_peak = max(robust_acc) if robust_acc else 1.0
        retention = [value / ra_peak if ra_peak > 0 else 0.0 for value in robust_acc]

        x_values = [radius * math.cos(theta) for radius in retention]
        y_values = [radius * math.sin(theta) for radius in retention]
        z_values = [-gap for gap in gaps]

        ax.plot(x_values, y_values, z_values, color=color, linewidth=2.0, alpha=0.9)
        for i, row in enumerate(rows):
            marker = phase_markers.get(phases[i], "o")
            size = 85 if phases[i] == "Robustness Stabilization / Peak" else 38
            ax.scatter(
                x_values[i],
                y_values[i],
                z_values[i],
                c=[progress[i]],
                cmap="viridis",
                vmin=0,
                vmax=1,
                marker=marker,
                s=size,
                edgecolors=color,
                linewidths=0.8,
                alpha=0.95,
            )

        for i in range(len(rows) - 1):
            dx = x_values[i + 1] - x_values[i]
            dy = y_values[i + 1] - y_values[i]
            dz = z_values[i + 1] - z_values[i]
            ax.quiver(
                x_values[i],
                y_values[i],
                z_values[i],
                dx,
                dy,
                dz,
                color=color,
                arrow_length_ratio=0.18,
                linewidth=0.9,
                alpha=0.75,
            )

        ax.text(
            1.1 * math.cos(theta),
            1.1 * math.sin(theta),
            0.04,
            model_name,
            fontsize=8,
            ha="center",
            va="center",
        )
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=model_name)
        )

    ax.set_xlabel("Model Identity Ring X")
    ax.set_ylabel("Model Identity Ring Y")
    ax.set_zlabel("- Clean-Robust Gap")
    ax.set_title(title)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-0.9, 0.05)
    ax.view_init(elev=28, azim=-62)

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_clim(0, 1)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label("Training Progress")

    phase_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="0.3", markeredgecolor="0.3", label="Robustness Stabilization / Peak", markersize=10),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markeredgecolor="0.6", label="Other checkpoints", markersize=7),
    ]
    ax.legend(handles=legend_handles + phase_handles, loc="upper left", bbox_to_anchor=(1.03, 1.0))
    _save_or_show(plt, output_path)


def plot_blackhole_gravity_collapse_v2(
    grouped_rows,
    title: str = "Normalized Black-Hole Robustness Collapse Map",
    output_path: str = None,
    inward_strength: float = 0.68,
):
    """
    Plot a cleaner derived collapse-coordinate map.

    Coordinates:
        theta = model identity
        normalized_gap(t) = Delta(t) / Delta_final
        radius = 1 - inward_strength * normalized_gap(t)
        z = - normalized_gap(t)

    Visual encoding:
        line gradient = training progress
        start marker edge color = model level
        star = RA peak checkpoint
    """
    plt = _get_plt()
    import math
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    model_names = list(grouped_rows.keys())
    level_colors = {
        "Level 1 - VGG-style CNN": "#1f77b4",
        "Level 2 - Residual CNN": "#2ca02c",
        "Level 3 - Wide Residual CNN": "#d62728",
    }
    level_labels = {
        "Level 1 - VGG-style CNN": "Baseline / VGG",
        "Level 2 - Residual CNN": "Mid / ResNet",
        "Level 3 - Wide Residual CNN": "Strong / WRN",
    }
    cmap = plt.get_cmap("viridis")

    theta_grid = np.linspace(0, 2 * math.pi, 260)
    outer_x = np.cos(theta_grid)
    outer_y = np.sin(theta_grid)
    inner_x = (1 - inward_strength) * np.cos(theta_grid)
    inner_y = (1 - inward_strength) * np.sin(theta_grid)
    ax.plot(outer_x, outer_y, np.zeros_like(theta_grid), color="0.15", lw=1.1, alpha=0.28)
    ax.plot(inner_x, inner_y, -np.ones_like(theta_grid), color="0.15", lw=1.1, alpha=0.16)

    # Faint funnel ribs.
    for theta in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        ax.plot(
            [math.cos(theta), (1 - inward_strength) * math.cos(theta)],
            [math.sin(theta), (1 - inward_strength) * math.sin(theta)],
            [0, -1],
            color="0.25",
            lw=0.45,
            alpha=0.12,
        )

    for index, model_name in enumerate(model_names):
        rows = sorted(grouped_rows[model_name], key=lambda row: float(row["progress"]))
        theta = 2 * math.pi * index / len(model_names)
        progress = np.array([float(row["progress"]) for row in rows])
        gaps = np.array([float(row["gap"]) for row in rows])
        robust_acc = np.array([float(row["test_robust_acc"]) for row in rows])
        level = rows[0].get("level", "")
        level_color = level_colors.get(level, "0.3")
        final_gap = gaps[-1] if gaps[-1] > 0 else max(gaps.max(), 1.0)
        normalized_gap = np.clip(gaps / final_gap, 0.0, 1.0)
        radius = 1.0 - inward_strength * normalized_gap
        x_values = radius * math.cos(theta)
        y_values = radius * math.sin(theta)
        z_values = -normalized_gap

        for i in range(len(rows) - 1):
            segment_color = cmap((progress[i] + progress[i + 1]) / 2)
            ax.plot(
                x_values[i:i + 2],
                y_values[i:i + 2],
                z_values[i:i + 2],
                color=segment_color,
                lw=2.6,
                alpha=0.95,
            )
            ax.quiver(
                x_values[i],
                y_values[i],
                z_values[i],
                x_values[i + 1] - x_values[i],
                y_values[i + 1] - y_values[i],
                z_values[i + 1] - z_values[i],
                color=segment_color,
                arrow_length_ratio=0.12,
                linewidth=0.7,
                alpha=0.65,
            )

        point_colors = [cmap(value) for value in progress]
        ax.scatter(
            x_values,
            y_values,
            z_values,
            c=point_colors,
            s=38,
            edgecolors="0.25",
            linewidths=0.45,
            alpha=0.95,
        )

        peak_index = int(np.argmax(robust_acc))
        ax.scatter(
            [x_values[peak_index]],
            [y_values[peak_index]],
            [z_values[peak_index]],
            marker="*",
            s=160,
            c=[point_colors[peak_index]],
            edgecolors="black",
            linewidths=0.9,
            alpha=1.0,
        )

        ax.scatter(
            [x_values[0]],
            [y_values[0]],
            [z_values[0]],
            s=180,
            facecolors="none",
            edgecolors=level_color,
            linewidths=2.8,
        )
        ax.scatter(
            [x_values[-1]],
            [y_values[-1]],
            [z_values[-1]],
            s=90,
            facecolors=point_colors[-1],
            edgecolors="black",
            linewidths=1.2,
        )

        ax.text(
            1.12 * math.cos(theta),
            1.12 * math.sin(theta),
            0.04,
            model_name,
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_xlabel("Model Identity Ring X")
    ax.set_ylabel("Model Identity Ring Y")
    ax.set_zlabel("- Normalized Clean-Robust Gap")
    ax.set_title(title)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_zlim(-1.05, 0.06)
    ax.view_init(elev=30, azim=-58)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0, 1)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label("Training Progress")

    used_levels = []
    for rows in grouped_rows.values():
        level = rows[0].get("level", "")
        if level not in used_levels:
            used_levels.append(level)
    level_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=level_colors.get(level, "0.3"),
            markeredgewidth=2.6,
            label=level_labels.get(level, level),
            markersize=9,
        )
        for level in used_levels
    ]
    phase_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="0.7", markeredgecolor="black", label="RA peak", markersize=11),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7", markeredgecolor="black", label="Final checkpoint", markersize=8),
    ]
    ax.legend(handles=level_handles + phase_handles, loc="lower left", bbox_to_anchor=(-0.02, -0.02), frameon=True)
    _save_or_show(plt, output_path)
