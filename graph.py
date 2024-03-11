from typing import Optional, List, Dict, Tuple

import pandas
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis, normaltest, chi2_contingency, pointbiserialr, spearmanr, kruskal
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    fbeta_score,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler

import shared.stats_utils as stats_utils
from shared.ml_config_core import CMResultsData, ModelTrainingResult

from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
from matplotlib.ticker import AutoLocator, AutoMinorLocator, ScalarFormatter, FuncFormatter

filter_samples_above_threshold = stats_utils.filter_samples_above_threshold
calculate_threshold_metrics = stats_utils.calculate_threshold_metrics


class CustomPowerScale(ScaleBase):
    name = 'custom_power'

    def __init__(self, axis, *, factor=10, **kwargs):
        super().__init__(axis)
        self.factor = factor

    def get_transform(self):
        return self.CustomPowerTransform(self.factor)

    def set_default_locators_and_formatters(self, axis):
        """
        Set default locators and formatters to the axis.
        """
        # AutoLocator automatically determines the tick locations.
        axis.set_major_locator(AutoLocator())
        # AutoMinorLocator automatically determines the minor tick locations.
        axis.set_minor_locator(AutoMinorLocator())
        # ScalarFormatter formats the ticks as scalar values.
        axis.set_major_formatter(ScalarFormatter())

    class CustomPowerTransform(Transform):
        input_dims = output_dims = 1

        def __init__(self, factor):
            super().__init__()
            self.factor = factor

        def transform_non_affine(self, y):
            # Ensure y is an array
            y = np.asarray(y)
            # Initialize the output array
            x = np.zeros_like(y)

            for i, yi in np.ndenumerate(y):
                # Apply the minimization to each element individually
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(
                    lambda xi: (np.power(xi, (1 / (1 + self.factor * xi))) - yi) ** 2,
                    bounds=(0, 1),
                    method='bounded'
                )
                x[i] = res.x
            return x

        def inverted(self):
            return CustomPowerScale.CustomPowerTransform(self.factor)

    class InvertedCustomPowerTransform(Transform):
        input_dims = output_dims = 1

        def __init__(self, factor):
            super().__init__()
            self.factor = factor

        def transform_non_affine(self, y):
            # Function to invert the transformation for a single value
            def invert_single_value(yi):
                res = minimize_scalar(lambda xi: (np.power(xi, (1 / (1 + self.factor * xi))) - yi) ** 2, bounds=(0, 1),
                                      method='bounded')
                return res.x

            # Apply the inversion function to each element if y is an array
            if np.iterable(y):
                return np.array([invert_single_value(yi) for yi in y])
            else:
                # Single value case
                return invert_single_value(y)

        def inverted(self):
            return CustomPowerScale.CustomPowerTransform(self.factor)


# Register the custom scale
register_scale(CustomPowerScale)


def render_pca_component_plot(_explained_pc, title=None):
    plt.figure(figsize=(12, 6))
    ax_b = barplot = sns.barplot(
        x="PC",
        y="var",
        data=_explained_pc,
        color="skyblue",
        label="Individual Variance",
        legend=False,  # Disable the
    )
    ax_b.set_ylim([0, 1.1])  # Adjust as needed

    barplot.set_ylim([0, 1])  # Adjust as needed
    # barplot.legend_.remove()

    lineplot_ax = plt.twinx()

    lineplot = sns.lineplot(
        x="PC",
        y="cum_var",
        data=_explained_pc,
        marker="o",
        sort=False,
        color="red",
        label="Cumulative Variance",
        ax=lineplot_ax,
    )
    ax_b.grid(False)
    lineplot_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax_b.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    barplot.set_ylabel("Individual Variance")
    lineplot_ax.set_ylabel("Cumulative Variance")
    plt.title(title if title else "PCA Explained Variance and Cumulative Variance")
    handles, labels = [], []

    for ax in [barplot, lineplot_ax]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    plt.legend(handles, labels, loc="center right")

    plt.show()


def stacked_box_plot(proportions, total_players_by_league):
    fig, ax = plt.subplots(figsize=(10, 6))
    proportions.T.plot(kind="barh", stacked=True, ax=ax)

    for i, league in enumerate(proportions.columns):
        total_players = total_players_by_league[league]
        ax.text(1.01, i, f"n={total_players}", va="center")

    ax.set_xlabel("Percentage of Goals")
    ax.set_ylabel("League")
    ax.set_title("Proportion of Goals by Player Categories in Each League")
    ax.text(
        0.5,
        -2.05,
        f"*Poland and Scotland excluded due to missing data",
        fontsize="medium",
        ha="right",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=len(proportions.columns),
        frameon=False,
    )

    plt.show()


def confusion_matrix_plot(
        model_info,
        title=None,
        axis_label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        ax=None,
        annotations: str = None,
        include_sample_count=False,
        cbar=False,
        subtitle=None,
):
    if ax is None:
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 8))

    conf_matrix = confusion_matrix(model_info.y_test, model_info.predictions)
    conf_matrix_percentage = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    )

    if labels is None:
        labels = np.unique(model_info.y_test)

    # Prepare annotations for heatmap
    if include_sample_count:
        annot = np.array(
            [
                f"{pct:.2%}\n(n={count})"
                for pct, count in np.nditer([conf_matrix_percentage, conf_matrix])
            ]
        )
        annot = annot.reshape(conf_matrix.shape)
    else:
        annot = True  # Or use .2% format if you prefer

    sns.heatmap(
        conf_matrix_percentage,
        annot=annot,
        fmt="" if include_sample_count else ".2%",
        cmap=sns.diverging_palette(220, 20, as_cmap=True).reversed(),
        cbar=cbar,
        ax=ax,
    )
    ax.set_xlabel(f"Predicted Labels{'' if axis_label is None else f' ({axis_label})'}")
    ax.set_ylabel(f"True Labels{'' if axis_label is None else f' ({axis_label})'}")

    if title and subtitle:
        ax.set_title(title, pad=35)
        ax.text(
            0.5,
            1.055,
            f"(Model: {subtitle})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        ax.set_title(title if title else "Confusion Matrix with Percentage Accuracy")

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    if annotations:
        ax.text(
            0.5, -0.1, annotations, ha="center", va="center", transform=ax.transAxes
        )

    return ax


def confusion_matrix_plot_v2(
        cm_data, ax=None, title=None, subtitle=None, annotations: str = None, regressor_input: bool = False
):
    predictions = cm_data.predictions

    if regressor_input:
        predictions = predictions.round()
        predictions = predictions.clip(lower=cm_data.y_test.min(), upper=cm_data.y_test.max())

    cm = confusion_matrix(cm_data.y_test, predictions)
    # cm_normalized = cm[:, np.newaxis]
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    class_accuracies = np.diag(cm_normalized)
    precision, recall, f1, _ = precision_recall_fscore_support(
        cm_data.y_test, predictions
    )

    mask_correct = np.eye(cm.shape[0], dtype=bool)
    mask_incorrect = ~mask_correct

    cmap_incorrect = sns.diverging_palette(220, 20, as_cmap=True)
    cmap_correct = cmap_incorrect.copy().reversed()
    # cmap_correct = sns.light_palette("seagreen", as_cmap=True)

    sns.heatmap(
        cm_normalized,
        vmin=0,
        vmax=1,
        mask=mask_incorrect,
        cmap=cmap_correct,
        annot=False,
        cbar=False,
        ax=ax,
    )

    sns.heatmap(
        cm_normalized,
        mask=mask_correct,
        cmap=cmap_incorrect,
        annot=False,
        alpha=0.5,
        cbar=False,
        ax=ax,
    )

    total_rows = len(cm_data.y_test)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            expected = cm_data.y_test.value_counts()[i]

            headline_anno = None
            annotation_1 = ""
            annotation_2 = ""
            annotation_2_1 = ""
            annotation_3 = ""

            if i == j:
                headline_anno = f"Recall:\n{class_accuracies[i]:.1%}\n"
                annotation_2 += f"F1: {f1[i]:.1%}"

                annotation_2_1 += f"Precision: {precision[i]:.1%}, "

            else:
                annotation_2 = f"Missed:\n{cm_normalized[i, j]:.1%}\n"

            annotation_3 += f"n={cm[i, j]} / {expected}"
            color = "black" if i == j else "white"

            if i == j:
                random_guess_accuracy = cm_data.y_test.value_counts()[j] / total_rows
                annotation_3 += f"\n\nExpected: {random_guess_accuracy:.1%}\n"

            if headline_anno:
                ax.text(
                    j + 0.5,
                    i + 0.3,
                    headline_anno,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=18,
                )
                ax.text(
                    j + 0.5,
                    i + 0.45,
                    annotation_2,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=16,
                )
                ax.text(
                    j + 0.5,
                    i + 0.525,
                    f"({annotation_2_1})",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=14,
                )
                ax.text(
                    j + 0.5,
                    i + 0.755,
                    annotation_3,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=16,
                )
            else:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    annotation_2,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=16,
                )
                ax.text(
                    j + 0.5,
                    i + 0.65,
                    annotation_3,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=16,
                )

    # Adding diagonal stripes pattern for incorrect predictions
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if mask_incorrect[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=False,
                        hatch="//",
                        edgecolor="black",
                        lw=0,
                        alpha=0.5,
                    )
                )

    if title and subtitle:
        ax.set_title(title, pad=35)
        ax.text(
            0.5,
            1.055,
            f"(Model: {subtitle})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        ax.set_title(title if title else "Confusion Matrix with Percentage Accuracy")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if annotations:
        ax.text(
            0.5, -0.1, annotations, ha="center", va="center", transform=ax.transAxes
        )


def render_feature_importances_chart(
        feature_importances,
        title="Feature Importance",
        subtitle=None,
):
    # Feature Importance Plot

    # feature_importances = benchmark_model_target_xgb.feature_importances_
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=feature_importances, y="Feature", x="Importance")
    ax.tick_params(axis="x", which="major", labelsize=16)

    def custom_label_function(original_label):
        modified_label = original_label.replace("_", " ")
        modified_label = modified_label.replace("league name", " ")
        return modified_label.title()

    ax.set_yticklabels(
        [
            custom_label_function(label)
            for label in [item.get_text() for item in ax.get_yticklabels()]
        ]
    )

    if title and subtitle:
        plt.title(title, pad=30)
        plt.text(
            0.5,
            1.035,
            f"(Model: {subtitle})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        plt.title(title if title else "Confusion Matrix with Percentage Accuracy")

    plt.show()


def roc_curve_plot(y_true, y_probs, ax, labels, annotations, class_idx, n):
    """
    Plot ROC curve for a specific class in a multi-class classification.

    Args:
    y_true (pd.Series): True labels.
    y_probs (pd.Series): Probability predictions for the specific class.
    ax (matplotlib.axes.Axes): Axes object to plot on.
    labels (list): Class labels.
    annotations (str): Text for annotations.
    class_idx (int): Index of the class to plot ROC for.
    n (int): Number of rows.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot
    ax.plot(fpr, tpr, label=f"Class {labels[class_idx]} (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: Class {labels[class_idx]}")
    ax.legend(loc="lower right")
    annotations += f", n={n}"
    # ax.text(0.5, 0.2, annotations, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, -0.2, annotations, ha="center", va="center", transform=ax.transAxes)


def get_counts_for_section(data, start_threshold, end_threshold, inverse=False):
    section_data = data[
        (data["probabilities"] >= start_threshold)
        & (data["probabilities"] < end_threshold)
        ]
    predictions = (data["probabilities"] > start_threshold).astype(int)

    y_test = section_data["y_test"]
    target_total = np.sum(y_test == 1)

    if inverse:
        tp = np.sum((y_test == 1))
        fp = np.sum((y_test == 0))
        total = len(y_test)
        fn, recall, precision = None, None, None
    else:
        tp = np.sum((predictions == 1) & (y_test == 1))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        total = len(section_data)
    return tp, fp, fn, total, target_total, recall, precision


# TODO: Specifically used by the defaults dataset to draw "risk groups" but should be made generic
def add_threshold_sections(ax, sections, data):
    total_rows = len(data["probabilities"])
    for section in sections:
        start = section["start"]
        start_data = section.get(
            "start_data", start
        )  # Use 'start_data' if available, otherwise use 'start'
        end = section["end"]
        end_data = section.get(
            "end_data", end
        )  # Use 'start_data' if available, otherwise use 'start'

        inverse = section.get(
            "inverse", False
        )  # Use 'start_data' if available, otherwise use 'start'

        color = section["color"]
        label = section["label"]
        description = section["description"]

        ax.axvspan(start, end, color=color, alpha=0.04)
        # Add vertical line at the start of the section
        if not inverse:
            ax.axvline(x=start, color="grey", linestyle="--", linewidth=2)

        # TODO: hack, actually add a dataclass for sections defintions and an enum for inverse/type
        tp, fp, fn, total, target_total, recall, precision = get_counts_for_section(
            data, start_data, end_data, inverse=inverse
        )

        # total_observed = tp + fn

        desc = f"{description}\nTotal: {total}\n({(total / total_rows):.1%}) \n Default R.:\n{(target_total / total):.2%}"
        # desc = f"{description}\nTotal: {total} \n tp:{tp} \n fn:{fn} \n fp:{fp} \n target_total:{target_total}"
        # desc = f"{description}\nTotal: {total}\n Default Rate:\n{(55 / total):.2%}"
        # desc=  f"{description}\nTotal: {total}\n Default Rate:\n{(total_observed / total):.2%}",

        # Place label and description as annotations
        ax.text(
            (start + end) / 2,
            0.985,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5),
        )

        if precision is None:
            ax.text(
                (start + end) / 2,
                0.875,
                desc,
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

        else:
            ax.text(
                (start + end) / 2,
                0.875,
                desc,
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )


def plot_threshold_metrics_v2(
        model_training_result: CMResultsData,
        min_threshold,
        max_threshold,
        model_name=None,
        class_pos=None,
        include_vars: Optional[List[str]] = None,
        show_threshold_n=False,
        sections=None,
        log_x=False,
):
    """
    Plots accuracy, F1 score, and sample count against various threshold values.

    Parameters:
    model_training_result: Result from the model training.
    min_threshold: Minimum threshold value.
    max_threshold: Maximum threshold value.
    """

    T_values = np.arange(min_threshold, max_threshold, 0.0125)

    last_filled_threshold = 0

    metrics_results = {}

    if include_vars is None:
        include_vars = ["accuracy", "f1", "log_loss"]

    default_vars = ["sample_count", "t_values_included"]
    include_vars = [*include_vars, *default_vars]
    for v in include_vars:
        metrics_results[v] = []

    for T in T_values:
        try:
            if class_pos is None:
                # TODO: this currently works fine, but it filters out the samples where any prob < threshold
                # TODO: so it's used for multicassificaiton problems

                raise NotImplementedError()
                filtered_result = filter_samples_above_threshold(
                    model_training_result, T
                )

                accuracy = filtered_result.metrics["accuracy"]

                recall = filtered_result.metrics["recall"]
                precision = filtered_result.metrics["precision"]

                f1_score = filtered_result.metrics["f1"]
                logs_loss = filtered_result.metrics["log_loss"]
                sample_count = len(filtered_result.y_test)
            else:
                filtered_result = calculate_threshold_metrics(
                    model_training_result, T, class_pos
                )

                # TODO: add support for multi class
                # recall_class_1 precision_class_1 f1_class_1 fbeta_25
                accuracy = filtered_result.metrics["recall_class_1"]
                recall = filtered_result.metrics["recall_class_1"]
                precision = filtered_result.metrics["precision_class_1"]

                f1_score = filtered_result.metrics["f1_class_1"]
                logs_loss = filtered_result.metrics["log_loss"]

                sample_count = sum(filtered_result.predictions)

            if "precision" in metrics_results:
                metrics_results["precision"].append(precision)

            if "recall" in metrics_results:
                metrics_results["recall"].append(recall)

            if "log_loss" in metrics_results:
                metrics_results["log_loss"].append(logs_loss if logs_loss < 2 else 2)

            if "accuracy" in metrics_results:
                metrics_results["accuracy"].append(accuracy)

            if "f1" in metrics_results:
                metrics_results["f1"].append(f1_score)

            if "log_loss" in metrics_results:
                metrics_results["t_values_included"].append(T)

            if sample_count > 0:
                last_filled_threshold = T

            # Always include
            metrics_results["sample_count"].append(sample_count)
            metrics_results["t_values_included"].append(T)

        except Exception as ex:
            # In case there are no predictions above a given threshold (e.g. using the betting odds model)
            raise ex

    fig, ax1 = plt.subplots(figsize=(17, 7))
    lines = []
    for value in include_vars:
        if value not in default_vars and not value == "log_loss":
            try:
                color = None
                alpha = None

                if value == "f1":
                    color = "grey"
                    alpha = 0.4

                (_line,) = ax1.plot(
                    metrics_results["t_values_included"],
                    metrics_results[value],
                    label=value.title(),
                    color=color,
                    alpha=alpha,
                )
                lines.append(_line)
            except Exception as ex:
                raise ex
                # raise Exception(value)

    ax1.set_xlabel("Threshold (T)")
    ax1.set_ylabel("Performance")
    ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax1.set_ylim([0, 1.05])

    tick_step = 4

    x_ticks = []
    x_ticks_labels = []

    if not log_x:
        for i, v in enumerate(T_values):
            if i % tick_step != 0:
                continue
            else:
                x_tick = str(round(v, 2))
                if show_threshold_n:
                    if i == 0:
                        x_tick = f"{x_tick}\nn="
                    else:
                        x_tick = x_tick + f"\n{metrics_results['sample_count'][i]}"
                x_ticks.append(v)
                x_ticks_labels.append(x_tick)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks_labels)
        ax1.set_xlim([T_values[0], last_filled_threshold])
    else:
        # FIX:        # ax1.set_xscale('custom_power', factor=-0.75)
        ax1.set_xscale('log')
        tick_positions = [.025, .05, .1, .2, .3, .4, .5, .7, .9]

        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([str(tick) for tick in tick_positions])
        ax1.set_xlim(0, 1)

        # TODO: HACK: need to make ticks scale with the custom scale
        # ax1.set_xticks([.025, .05, .1, .15, .2, .3, .4, .5, .7, .9])

    metric_axes = [ax1]

    if "log_los" in include_vars:
        ax1.spines["left"].set_position(("axes", -0.1))

        ax3 = ax1.twinx()
        ax3.spines["left"].set_visible(True)
        ax3.spines["left"].set_position(("axes", 0.0))
        ax3.yaxis.set_label_position("left")
        ax3.yaxis.tick_left()

        ax3.set_ylabel("")
        ax3.plot(
            metrics_results["t_values_included"],
            metrics_results["log_loss"],
            label="Log Loss",
            color="purple",
        )
        ax3.set_yticks([0.3, 0.5, 1, 1.5, 2])
        ax3.set_yticklabels(["Log Loss", 0.5, 1, 1.5, ">=2"])
        ax3.set_ylim([0, 2.1])
        ax3.grid(False)
        metric_axes.append(ax3)

        (line_log_loss,) = ax3.plot(
            metrics_results["t_values_included"],
            metrics_results["log_loss"],
            label="Log Loss",
            color="purple",
        )
        lines.append(line_log_loss)
    else:
        ax1.spines["left"].set_position(("axes", -0.01))

    for _ax in metric_axes:
        for label in _ax.get_yticklabels():
            label.set_rotation(90)
            break

    if sections:
        data_for_sections = pd.DataFrame(
            {
                # "predictions": model_training_result.predictions,
                "y_test": model_training_result.y_test,
                "probabilities": model_training_result.probabilities.iloc[:, 1],
            }
        )

        add_threshold_sections(ax1, sections, data_for_sections)

    if class_pos is None:
        ax2 = ax1.twinx()
        scatter = ax2.scatter(
            metrics_results["t_values_included"],
            metrics_results["sample_count"],
            color="green",
            label="Sample Count",
        )

        ax2.set_ylabel("Number of Samples (log)")
        ax2.set_yticks([10, 100, 1000, 5000])
        ax2.grid(False)

        ax2.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        for T, count in zip(
                metrics_results["t_values_included"], metrics_results["sample_count"]
        ):
            if count < 500:
                ax2.annotate(
                    f"{count}",
                    (T, count),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )
    else:
        scatter = None

    fig.legend(
        handles=[*lines, *([] if scatter is None else [])],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),  # Positioning relative to the entire figure
        ncol=4,
    )

    fig.text(
        0.95,
        -0.05,
        f"n={len(model_training_result.predictions)} (target_class_n={model_training_result.y_test.sum()})",
        ha="right",
        va="top",
        fontsize="small",
    )

    title = "Model Performance By Prob. Threshold"
    if model_name:
        plt.title(title, pad=35)
        plt.text(
            0.5,
            1.055,
            f"(Model: {model_name})",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize="medium",
        )
    else:
        plt.title("Model Performance By Prob. Threshold")

    plt.tight_layout()  # Adjust layout

    plt.show()


def make_annotations(cv_info: dict):
    log_loss = f"log_loss={cv_info['log_loss']:.2f}, " if "log_loss" in cv_info else ""
    f1 = f"f1={cv_info['f1_macro']:.2f}, " if "f1_macro" in cv_info else ""
    precision = f"precision={cv_info['precision_macro']:.2f}, " if "precision_macro" in cv_info else ""
    recall = f"recall={cv_info['recall_macro']:.2f}, " if "recall_macro" in cv_info else ""
    accuracy = f"accuracy={cv_info['accuracy']:.2f}" if "accuracy" in cv_info else ""
    return f"macro: {log_loss}{f1}{precision}{recall}{accuracy}"


def roc_precision_recal_grid_plot(
        # confusion_matrices,
        model_training_results: Dict[str, ModelTrainingResult],
        add_fbeta_25=False,
        threshold_x_axis=True
):
    n = len(model_training_results)
    columns = 2
    rows = n
    height = 8
    width = height * columns

    fig, axes = plt.subplots(
        rows, columns, figsize=(width, height * rows), constrained_layout=True
    )

    # for model_key, model_training_result in model_training_results.items():
    # for i, model_key in enumerate(confusion_matrices):
    for i, model_key in enumerate(model_training_results.keys()):
        model_training_result = model_training_results[model_key]
        #     matrix_data = confusion_matrices[model_key][1]
        matrix_data = model_training_result.test_data
        # matrix_data = model_training_result.cm_data
        probabilities = matrix_data.probabilities
        y_test = matrix_data.y_test

        # ROC Curve for Class=1
        fpr, tpr, _ = roc_curve(y_test, probabilities.iloc[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        axes[i, 0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
        axes[i, 0].plot([0, 1], [0, 1], "k--")
        axes[i, 0].set_xlim([0.0, 1.0])
        axes[i, 0].set_xticks(list(np.arange(0, 1, 0.1)))
        axes[i, 0].set_ylim([0.0, 1.05])
        axes[i, 0].set_xlabel("False Positive Rate")
        axes[i, 0].set_ylabel("True Positive Rate")
        axes[i, 0].set_title(f"{model_key} ROC", fontsize=16)
        axes[i, 0].legend(loc="lower right")

        if threshold_x_axis:
            precision, recall, thresholds = precision_recall_curve(
                y_test, probabilities.iloc[:, 1]
            )
            pr_auc = average_precision_score(y_test, probabilities.iloc[:, 1])

            thresholds = np.append(thresholds, 1)

            axes[i, 1].plot(thresholds, precision, label="Precision", linestyle="-")
            axes[i, 1].plot(thresholds, recall, label="Recall", linestyle="-")

            axes[i, 1].set_xlabel("Threshold")
            axes[i, 1].set_ylabel("Value")
            axes[i, 1].set_title(
                f"{model_key} Precision-Recall vs. Threshold", fontsize=16
            )

            axes[i, 1].set_xlim([0.0, 1.0])
            axes[i, 1].set_xticks(list(np.arange(0, 1, 0.1)))

            axes[i, 1].set_ylim([0.0, 1.05])

            no_skill = len(y_test[y_test == 1]) / len(y_test)
            axes[i, 1].plot(
                [0, 1], [no_skill, no_skill], linestyle="--", label="No Skill"
            )
            axes[i, 1].legend(loc="best")

        else:
            # Precision-Recall Curve for Class=1
            precision, recall, _ = precision_recall_curve(
                y_test, probabilities.iloc[:, 1]
            )
            pr_auc = average_precision_score(y_test, probabilities.iloc[:, 1])
            axes[i, 1].plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
            axes[i, 1].set_xlim([0.0, 1.0])
            axes[i, 1].set_xticks(list(np.arange(0, 1, 0.1)))

            axes[i, 1].set_ylim([0.0, 1.05])
            axes[i, 1].set_xlabel("Recall")
            axes[i, 1].set_ylabel("Precision")
            axes[i, 1].set_title(f"{model_key} Precision-Recall", fontsize=16)
            no_skill = len(y_test[y_test == 1]) / len(y_test)
            axes[i, 1].plot(
                [0, 1], [no_skill, no_skill], linestyle="--", label="No Skill"
            )

            f2_5_scores = (
                    (1 + 2.5 ** 2)
                    * (precision * recall)
                    / ((2.5 ** 2 * precision) + recall + np.finfo(float).eps)
            )
            f2_5_scores = np.nan_to_num(f2_5_scores)

            axes[i, 1].plot(
                recall, f2_5_scores, color="grey", label=f"F2.5 Score", alpha=0.2
            )
            axes[i, 1].legend(loc="upper right")

        annotations = make_annotations(model_training_result.cv_metrics)
        axes[i, 0].text(
            0.0,
            -0.1,
            annotations + "\n",
            ha="left",
            va="center",
            fontsize=12,
            transform=axes[i, 0].transAxes,
        )

    plt.suptitle("ROC / Precision-recall Curves (default = 1)", fontsize=20, y=1.0005)
    plt.tight_layout()
    plt.show()


def clean_tick_label(v):
    label = str(v).split(" ")[0]
    label = label.replace("_", " ")
    label = label.title()

    label = "Yes" if label == "1" else label
    label = "No" if label == "0" else label

    return label


def _group_small_internal(proportions: pd.DataFrame, absolute=False):
    if absolute:
        threshold = 0.07 * proportions.sum()
    else:
        threshold = 0.07  # Set the threshold for grouping small values (10%)
    small_proportions = proportions[proportions < threshold]
    other_proportion = small_proportions.sum()
    proportions = proportions[proportions >= threshold].copy()
    if other_proportion > 0:
        proportions["Others"] = other_proportion
    return proportions


# def _group_small_internal_absolute(proportions: pd.DataFrame):
#     threshold = 0.07 * proportions.sum()
#     small_proportions = proportions[proportions < threshold]
#
#     other_proportion = small_proportions.sum()
#     proportions = proportions[proportions >= threshold].copy()
#     if other_proportion > 0:  # Check if there are small values to group
#         proportions["Others"] = other_proportion  # Add the grouped "Others" category
#     return proportions


def _summary_features_pie_chart(col_vals: pd.Series, source_df_no_cat: pd.DataFrame, axes, variable: str,
                                group_small=True):
    unique_values = col_vals.unique()

    counts = source_df_no_cat[variable].value_counts()
    count_plot_df = source_df_no_cat[
        source_df_no_cat[variable].isin(counts[counts >= 50].index)
    ]
    count_plot_df[variable] = count_plot_df[variable].astype("str")
    sns.countplot(x=variable, data=count_plot_df, ax=axes[0], width=0.55)

    axes[0].set_title("Frequency Distribution", fontsize=12)

    if len(count_plot_df[variable].unique()) <= 10:

        tick_labels = [
            clean_tick_label(v) for v in list(count_plot_df[variable].unique())
        ]

        axes[0].set_xticklabels(tick_labels)
    else:
        axes[0].set_xticklabels([])  # Hide tick labels if there are too many X values

    axes[0].set_xlabel("")

    most_common_group_count = source_df_no_cat[variable].value_counts().iloc[0]
    y_limit_0 = (((most_common_group_count // 500) + 1) * 500) + 1000

    non_nan_count = source_df_no_cat[variable].count()

    y_limit_1 = ((non_nan_count // 500) + 1) * 500

    # If high number of values very srpead of out base limit on top group
    axes[0].set_ylim([0, min(y_limit_0, y_limit_1)])

    proportions = col_vals.value_counts(normalize=True)
    if group_small:
        proportions = _group_small_internal(proportions)
        # threshold = 0.07  # Set the threshold for grouping small values (10%)
        # small_proportions = proportions[proportions < threshold]
        # other_proportion = small_proportions.sum()
        # proportions = proportions[proportions >= threshold]
        # if other_proportion > 0:  # Check if there are small values to group
        #     proportions["Others"] = other_proportion  # Add the grouped "Others" category

    explode = [0.02] * len(proportions)
    wedges, _ = axes[1].pie(
        proportions,
        labels=None,
        autopct=None,
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=explode,
    )

    for wedge, label in zip(wedges, proportions.index):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(angle)
        pct = round(np.round(wedge.theta2 - wedge.theta1) / 360 * 100, 1)
        label_tr = clean_tick_label(label)
        # label_tr = str(label).replace("_", " ").title()
        axes[1].annotate(
            f"{label_tr}: {pct}%",
            xy=(x / 2, y / 2),
            xytext=(1.15 * x, 1.15 * y),
            arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle),
            horizontalalignment=horizontalalignment,
        )


def summary_df_features(source_df: pandas.DataFrame):
    source_df_no_cat = source_df.copy()

    scaler = StandardScaler()
    source_df_scaled = source_df_no_cat.select_dtypes(exclude=["object"])

    for i, variable in enumerate(source_df_no_cat.columns):
        col_vals = source_df_no_cat[variable]
        col_vals_non_nan = source_df_no_cat[variable].dropna()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axes[0].set_aspect(aspect="auto", adjustable="box")
        axes[1].set_aspect(aspect="auto", adjustable="box")

        nan_count = source_df_no_cat[variable].isna().sum()
        proportion_nan = round((nan_count / len(source_df_no_cat)), 2) * 100
        annotations = ""
        try:
            if stats_utils.is_non_numerical_or_discrete(col_vals):
                _summary_features_pie_chart(col_vals, source_df_no_cat, axes, variable)
            else:
                sns.kdeplot(x=variable, data=source_df_no_cat, ax=axes[0], fill=True)
                axes[0].set_title("Original Data Distribution (KDE)", fontsize=12)
                axes[0].set_ylabel("")
                axes[0].set_yticks([])

                # Statistical Annotations on KDE Plot
                mean_val = col_vals.mean()
                std_dev = col_vals.std()
                stat, p_val = normaltest(col_vals)
                normality = "Normal" if p_val > 0.05 else "Not normal"

                sns.boxenplot(
                    x=variable,
                    data=source_df_no_cat.dropna(),
                    ax=axes[1],
                    width=0.4,
                )
                axes[1].set_yticks([])

                Q3 = source_df_no_cat[variable].quantile(0.75)
                Q1 = source_df_no_cat[variable].quantile(0.25)
                IQR = Q3 - Q1
                upper_whisker = Q3 + 1.5 * IQR
                std_dev = source_df_no_cat[variable].std()

                upper_bound = upper_whisker + 0.5 * std_dev
                axes[1].set_xlim(0, min(upper_bound, source_df_no_cat[variable].max()))

                skewness = skew(col_vals_non_nan)
                excess_kurtosis = kurtosis(col_vals_non_nan)
                axes[1].text(
                    0.95,
                    0.85,
                    f"Skew: {skewness:.2f}\nKurt: {excess_kurtosis:.2f}",
                    ha="right",
                    va="center",
                    transform=axes[1].transAxes,
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"
                    ),
                )
                annotations = f"{normality}\nP-val: {round(p_val, 3)}\nMean: {mean_val:.2f}\nStd Dev: {std_dev:.2f}\n"
            if nan_count > 0:
                annotations += f"NaN%: {proportion_nan}"

            if len(annotations) > 0:
                axes[0].text(
                    0.95,
                    0.8,
                    annotations,
                    ha="right",
                    va="center",
                    transform=axes[0].transAxes,
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"
                    ),
                )

            for ax in axes:
                ax.set_title(ax.get_title(), fontsize=12)
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                ax.set_ylabel(ax.get_ylabel(), fontsize=10)

            title = variable.replace("_", " ").title()
            if title == "Bmi":
                title = "BMI"
            plt.suptitle(title, fontsize=16, y=1.02)

            plt.tight_layout()
            plt.show()

        except Exception as ex:
            plt.close(fig)
            raise ex


def render_corr_matrix_based_on_type(source_df: pd.DataFrame):
    def correlation_test(x, y):
        if x.dtype.name == "category":
            x = x.astype("object")
        if y.dtype.name == "category":
            y = y.astype("object")

        def chi_squared_test(x, y):
            contingency_table = pd.crosstab(x, y)
            chi2, p, _, _ = chi2_contingency(contingency_table)
            n = np.sum(contingency_table.values)
            k, r = contingency_table.shape
            cramers_v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))
            return cramers_v, p

        if x.dtype == "object" or y.dtype == "object":
            return chi_squared_test(x, y)
        elif x.dtype == "bool" and y.dtype in ["int64", "float64"]:
            return pointbiserialr(x, y)
        elif y.dtype == "bool" and x.dtype in ["int64", "float64"]:
            return pointbiserialr(y, x)
        elif x.dtype in ["int64", "float64"] and y.dtype in ["int64", "float64"]:
            return spearmanr(x, y)
        else:
            raise ValueError(
                f"Unsupported data types for correlation test: {x.dtype} and {y.dtype}"
            )

    corr = pd.DataFrame(index=source_df.columns, columns=source_df.columns)
    p_values = pd.DataFrame(index=source_df.columns, columns=source_df.columns)

    for col1 in source_df.columns:
        for col2 in source_df.columns:
            if col1 != col2:
                corr_value, p_value = correlation_test(source_df[col1], source_df[col2])
                corr.loc[col1, col2] = corr_value
                p_values.loc[col1, col2] = p_value
            else:
                # Handle the diagonal (correlation with self)
                corr.loc[col1, col2] = 1.0
                p_values.loc[col1, col2] = 0.0

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(26, 22))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    corr = round(corr.applymap(pd.to_numeric), 2)

    significant_mask = np.abs(corr) >= 0.1
    combined_mask = mask | ~significant_mask

    def format_annotation(corr_value, p_value):
        if p_value < 0.05:
            return f"{corr_value:.2f}*"
        elif abs(corr_value) > 0.35:
            return f"{corr_value:.2f}\np={p_value:.2f}"
        return ""

    vectorized_formatter = np.vectorize(format_annotation)

    sns.heatmap(
        corr,
        mask=combined_mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=vectorized_formatter(corr.to_numpy(), p_values.to_numpy()),
        fmt="",
    )

    plt.title("Correlation bet variable pairs")
    plt.annotate(
        "* p < 0.05\nonly columns where correlation is > 0.1 shown (",
        xy=(0.5, -0.175),
        xycoords="axes fraction",
        xytext=(0, -40),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    plt.show()


def draw_distribution_pie_charts(eda_df_ext: pd.DataFrame, split_var="gender", include_cols=None, group_small=True):
    if include_cols is None:
        include_cols = ["ever_married", "work_type", "Residence_type", "smoking_status", "bmi_binned_cats"]

    ii_empl_df = eda_df_ext[[split_var, *include_cols]]

    fig, axes = plt.subplots(len(include_cols), 2, figsize=(12, len(include_cols) * 5))

    for i, column in enumerate(include_cols):
        for j, target in enumerate(ii_empl_df[split_var].unique()):
            target_s = ii_empl_df[ii_empl_df[split_var] == target][column]
            if group_small:
                data = target_s.value_counts()
                data = _group_small_internal(data, absolute=True)

            else:
                data = target_s.value_counts()

            pie_labels = [
                f"{index}" for index, pct in zip(data.index, data * 100 / data.sum())
            ]
            axes[i, j].set_title(f"{column} for {target}", fontdict={"fontsize": 12})

            # proportions = col_vals.value_counts(normalize=True)

            explode = [0.02] * len(data)
            wedges, _ = axes[i, j].pie(
                data,
                labels=None,
                autopct=None,
                startangle=140,
                wedgeprops=dict(width=0.3),
                explode=explode,
            )
            # axes[i, j].pie(data, labels=pie_labels, autopct="%1.1f%%", startangle=90)

            #
            for wedge, label in zip(wedges, data.index):
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = np.cos(np.deg2rad(angle))
                y = np.sin(np.deg2rad(angle))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(angle)
                pct = round(np.round(wedge.theta2 - wedge.theta1) / 360 * 100, 1)
                label_tr = clean_tick_label(label)
                # label_tr = str(label).replace("_", " ").title()
                axes[i, j].annotate(
                    f"{label_tr}: {pct}%",
                    xy=(x / 2, y / 2),
                    xytext=(1.15 * x, 1.15 * y),
                    arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle),
                    horizontalalignment=horizontalalignment,
                )

    plt.suptitle(split_var.replace("_", " ").title(), fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()


def boxen_plot_by_cat(c, eda_df_ext, y_target):
    _df = eda_df_ext.copy()
    _df = _df[_df[y_target].notna()]
    grouped = _df.groupby(c)[y_target]
    groups = [group for name, group in grouped]
    stat, p_value = kruskal(*groups)
    test_explain = f"Kruskal-Wallis Test for {c} vs {y_target}: p-value = {p_value:.3f}"

    if p_value < 0.05:
        group_counts = _df.groupby(c).size()
        log_base = 2
        max_width = 0.8

        log_widths = np.log(group_counts + 1) / np.log(log_base)
        normalized_widths = log_widths / log_widths.max()
        scaled_widths = normalized_widths * max_width
        min_width = 0.05
        final_widths = scaled_widths.clip(min_width)

        plt.figure(figsize=(9, 6))
        for group in normalized_widths.index:
            sns.boxenplot(
                data=_df[_df[c] == group],
                x=c,
                y=y_target,
                color="b",
                width=final_widths[group],
            )
        plt.title(f"{' '.join(c.split('_')).title()}\n", fontdict={"fontsize": 18})
        plt.xticks(
            ticks=range(len(group_counts)),
            labels=[f"{group}\nn={count}" for group, count in group_counts.items()],
        )
        plt.annotate(
            test_explain,
            fontsize=12,
            xy=(0, -0.05),
            xycoords="axes fraction",
            xytext=(0, -40),
            textcoords="offset points",
            ha="left",
            va="top",
        )

        plt.xlabel("")
        plt.show()
    else:
        # if VERBOSE:
        #     print(
        #         f"{c} vs {y_target} No significant difference found (p-value = {p_value:.3f})"
        #     )
        return f"{c} vs {y_target} No significant difference found (p-value = {p_value:.3f})"


def boxen_plots_by_category(source_df: pd.DataFrame,
                            group_col: str,
                            target_col: str,
                            title: Optional[str] = None,
                            x_label: Optional[str] = None,
x_range: Tuple = None
                            ):
    group_names = source_df[group_col].unique()

    target_heigh = max(10, len(group_names) + 1)

    plt.figure(figsize=(10, target_heigh))  # Increase size

    group_names_sorted = (
        source_df.groupby(group_col)[target_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )


    if x_label:
        plt.xlabel(x_label)  # Change X-axis label
    ax = sns.boxenplot(
        k_depth="tukey",
        fill=False,
        linewidth=2.1,
        data=source_df,
        x=target_col,
        y=group_col,
        orient="y",
        # color="b",
        line_kws=dict(linewidth=1.5, color="black"),
        order=group_names_sorted,
        width_method="linear",
    )

    if x_range:
        plt.xlim(x_range[0], x_range[1])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: stats_utils.format_amount(x)))

    for i, league in enumerate(group_names_sorted):
        median = source_df[source_df[group_col] == league][
            target_col
        ].median()
        if i == 0:
            ax.text(
                median,
                i - 0.65,
                f"Median",
                ha="center",
                va="center",
                fontsize=12,
                # rotation=270,
            )

        offset = source_df[target_col].max() if x_range is None else x_range[1]
        ax.text(
            median + (offset * 0.015),
            i,
            f"{stats_utils.format_amount(median)}",
            ha="center",
            va="center",
            fontsize=12,
            rotation=270,
        )

    if x_label:
        plt.xlabel(x_label)
    plt.ylabel("\n")

    if title:
        plt.title(title, pad=25)
