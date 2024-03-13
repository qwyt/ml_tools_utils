import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import shared.ml_config_core as ml_config_core

CMResultsData = ml_config_core.CMResultsData
ModelTrainingResult = ml_config_core.ModelTrainingResult
CMResultsDataStats = ml_config_core.CMResultsDataStats


def calc_classification_metrics(predictions, probs_a, y_test):
    metrics = {}
    metrics["f1"] = f1_score(y_test, predictions, average="macro")
    metrics["accuracy"] = accuracy_score(y_test, predictions)
    metrics["precision"] = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(y_test, predictions, average="macro")

    if len(y_test.unique()) <= 2:
        # Bit bugges since now the preidiction list is binary
        metrics["recall_class_1"] = recall_score(y_test, predictions, pos_label=1)
        metrics["precision_class_1"] = precision_score(y_test, predictions, pos_label=1)
        metrics["f1_class_1"] = f1_score(y_test, predictions, pos_label=1)
        metrics["fbeta_25"] = fbeta_score(y_test, predictions, beta=2.5, pos_label=1)

    try:
        metrics["log_loss"] = log_loss(y_test, probs_a)
    except:
        metrics["log_loss"] = 2  # TODO: handle

    metrics = {key: round(metrics[key], 4) for key in metrics}
    return metrics


def evaluate_classifier_model(model, X_test, y_test, continuous=False):
    predictions = model.predict(X_test)
    predictions = pd.Series(predictions, index=X_test.index)

    try:
        probabilities = model.predict_proba(X_test)
        probabilities = pd.DataFrame(probabilities, index=X_test.index)
        use_prob = True
    except:
        probabilities = None
        use_prob = False
    probabilities_match_id = None

    rounded_predictions = predictions

    metrics = calc_classification_metrics(
        rounded_predictions, probabilities[[0, 1]] if use_prob else None, y_test
    )

    return metrics, predictions, probabilities, probabilities_match_id


def calc_regression_metrics(predictions, y_test):
    metrics = {}
    metrics["mae"] = mean_absolute_error(y_test, predictions)
    metrics["mse"] = mean_squared_error(y_test, predictions)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mape"] = mean_absolute_percentage_error(y_test, predictions)
    metrics["r2"] = r2_score(y_test, predictions)

    metrics = {key: round(value, 4) for key, value in metrics.items()}
    return metrics


def evaluate_regressor_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = pd.Series(predictions, index=X_test.index)

    metrics = calc_regression_metrics(predictions, y_test)

    return (
        metrics,
        predictions,
        None,
        None,
    )  # 'None' for probabilities and probabilities_match_id as they are not applicable in regression


# def compute_class_accuracies(predictions, y_test, classes):
#     predictions = np.rint(predictions).astype(int)
#     cm = confusion_matrix(y_test, predictions)
#     class_accuracies = {}
#
#     for idx, class_label in enumerate(classes):
#         accuracy = cm[idx, idx] / cm[idx].sum()
#         total_samples = cm[idx].sum()
#         class_accuracies[class_label] = {
#             "accuracy": accuracy,
#             "total_samples": total_samples,
#         }
#
#     return class_accuracies
#
#
# def _compute_class_accuracies(model, x_test, y_test):
#     predictions = model.predict(x_test)
#     return compute_class_accuracies(predictions, y_test, model.classes_)


# def get_pca_explained(df):
#     binary_columns = df.columns[df.nunique() == 2]
#     numerical_columns = df.select_dtypes(include=["number"]).columns
#     non_binary_numerical_columns = numerical_columns.difference(binary_columns)
#
#     numerical_transformer = Pipeline([("scaler", StandardScaler())])
#
#     passthrough_transformer = FunctionTransformer(lambda x: x)
#
#     # Preprocessing
#     preprocessor = ColumnTransformer(
#         [
#             ("num", numerical_transformer, non_binary_numerical_columns),
#             ("binary", passthrough_transformer, binary_columns),
#         ],
#         remainder="passthrough",
#     )
#
#     # Apply transformations
#     processed_df = pd.DataFrame(
#         preprocessor.fit_transform(df), columns=numerical_columns.append(binary_columns)
#     )
#
#     component_n = min(30, len(processed_df.columns) - 1)
#     _pca_11 = PCA(n_components=component_n)
#     _pca_11.fit(processed_df)
#
#     pca_transformed = _pca_11.transform(processed_df)
#     _pca_11_labels = [f"PC{i + 1}" for i in range(component_n)]
#
#     _explained_pc = pd.DataFrame(
#         {"var": _pca_11.explained_variance_ratio_, "PC": _pca_11_labels}
#     )
#     _explained_pc["cum_var"] = _explained_pc["var"].cumsum()


def get_pca_explained(df, component_n=None):
    binary_columns = df.columns[
        (df.nunique() == 2)
        & (
            df.apply(
                lambda x: sorted(x.unique()) == [0, 1]
                or sorted(x.unique()) == [False, True]
            )
        )
    ]
    numerical_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    non_binary_numerical_columns = numerical_columns.difference(binary_columns)

    numerical_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]
    )
    passthrough_transformer = FunctionTransformer(lambda x: x)

    # Preprocessing
    preprocessor = ColumnTransformer(
        [
            ("num", numerical_transformer, non_binary_numerical_columns),
            # ("cat", categorical_transformer, categorical_columns),
            # ("binary", passthrough_transformer, binary_columns),
        ],
        remainder="passthrough",
    )

    transformed_data = preprocessor.fit_transform(df)
    # transformed_col_names = (
    #     # preprocessor.named_transformers_["num"]
    #     # .get_feature_names_out(non_binary_numerical_columns)
    #     # .tolist()
    #     # + preprocessor.named_transformers_["cat"]
    #     # .get_feature_names_out(categorical_columns)
    #     # .tolist()
    #     # + binary_columns.tolist()
    # )

    # If using remainder='passthrough', add untransformed column names
    if preprocessor.remainder == "passthrough":
        remainder_columns = [
            col
            for col in df.columns
            if col not in non_binary_numerical_columns
            and col not in categorical_columns
            and col not in binary_columns
        ]
        # transformed_col_names.extend(remainder_columns)

    # Create processed DataFrame with new column names
    processed_df = pd.DataFrame(transformed_data, columns=df.columns)
    # processed_df = pd.DataFrame(transformed_data, columns=transformed_col_names)

    if component_n is None:
        component_n = min(30, processed_df.shape[1] - 1)
    _pca_11 = PCA(n_components=component_n)
    _pca_11.fit(processed_df)

    pca_transformed = _pca_11.transform(processed_df)
    _pca_11_labels = [f"PC{i + 1}" for i in range(component_n)]

    _explained_pc = pd.DataFrame(
        {"var": _pca_11.explained_variance_ratio_, "PC": _pca_11_labels}
    )
    _explained_pc["cum_var"] = _explained_pc["var"].cumsum()
    return _explained_pc, pca_transformed


def is_non_numerical_or_discrete(column):
    is_obj = column.dtype == "object"
    is_bool = column.dtype == "bool"
    is_cat = column.dtype == "category"
    is_discrete = column.dtype in ["int64", "int32"] and column.nunique() < 10

    return is_obj or is_bool or is_discrete or is_cat


def compute_class_accuracies(predictions, y_test, classes):
    predictions = np.rint(predictions).astype(int)
    cm = confusion_matrix(y_test, predictions)
    class_accuracies = {}

    for idx, class_label in enumerate(classes):
        accuracy = cm[idx, idx] / cm[idx].sum()
        total_samples = cm[idx].sum()
        class_accuracies[class_label] = {
            "accuracy": accuracy,
            "total_samples": total_samples,
        }

    return class_accuracies


def calculate_threshold_metrics(
    data: CMResultsData, threshold: float, positive_class_index=1
) -> CMResultsDataStats:
    if positive_class_index not in [0, 1]:
        raise ValueError("positive_class_index must be 0 or 1")

    positive_class_probabilities = data.probabilities.iloc[:, positive_class_index]

    new_predictions = (positive_class_probabilities >= threshold).astype(int)

    updated_metrics = calc_classification_metrics(
        new_predictions, positive_class_probabilities, data.y_test
    )
    classes = np.sort(data.y_test.unique())
    updated_class_accuracies = compute_class_accuracies(
        new_predictions, data.y_test, classes
    )

    return CMResultsDataStats(
        y_test=data.y_test,
        x_test=data.x_test,
        predictions=new_predictions,
        probabilities=data.probabilities,
        metrics=updated_metrics,
        class_accuracies=updated_class_accuracies,
    )


def filter_samples_above_threshold(model_training_result, threshold):
    # if class_pos is None:
    # high_prob_indices = model_training_result.probabilities.max(axis=1) > threshold
    # predictions = model_training_result.predictions

    high_prob_indices = model_training_result.probabilities.max(axis=1) > threshold
    # Filter the data
    filtered_y_test = model_training_result.y_test[high_prob_indices]
    filtered_predictions = model_training_result.predictions[high_prob_indices]
    filtered_x_test = model_training_result.x_test[high_prob_indices]
    filter_probs_a = model_training_result.probabilities[high_prob_indices]
    filtered_probs = model_training_result.probabilities.loc[high_prob_indices]

    updated_metrics = calc_classification_metrics(
        filtered_predictions, filter_probs_a, filtered_y_test
    )
    classes = np.sort(filtered_y_test.unique())  # Assuming class labels are in y_test
    updated_class_accuracies = compute_class_accuracies(
        filtered_predictions, filtered_y_test, classes
    )

    filtered_model_training_result = CMResultsDataStats(
        y_test=filtered_y_test,
        x_test=filtered_x_test,
        predictions=filtered_predictions,
        probabilities=filtered_probs,
        metrics=updated_metrics,
        class_accuracies=updated_class_accuracies,
    )

    # result : CMResultsData = CMResultsData(
    #     test_model: sklearn.pipeline.Pipeline
    #     y_test: pd.Series
    #     predictions: pd.Series
    #     probabilities: pd.DataFrame
    #
    #     class_accuracies: dict
    #
    # )
    #
    return filtered_model_training_result


def nan_summary(df):
    nan_counts = df.isna().sum()
    proportion_nan = round((df.isna().sum() / len(df)), 2) * 100
    summary_df = pd.DataFrame(
        {"Total NaN Values": nan_counts, "Proportion NaN (%)": proportion_nan}
    )
    summary_df = summary_df[summary_df["Total NaN Values"] > 0]
    return summary_df


def duplicate_summary(df):
    duplicates = df[df.duplicated(keep=False)]
    proportion_duplicates = len(duplicates) / len(df) * 100
    return proportion_duplicates, duplicates


def bin_and_label(column, num_bins=4):
    """
    Function to create bins based on quantiles and generate labels including value ranges
    Determines bin edges based on quantiles to handle skewness

    :param column:
    :param num_bins:
    :return:
    """
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = column.quantile(quantiles)

    unique_edges = bin_edges.unique()
    if len(unique_edges) < 2:
        unique_edges = np.linspace(column.min(), column.max(), num_bins + 1)

    labels = [
        f"{round(unique_edges[i], 2)} - {round(unique_edges[i + 1], 2)}"
        for i in range(len(unique_edges) - 1)
    ]

    binned_data = pd.cut(column, bins=unique_edges, labels=labels, include_lowest=True)
    return binned_data


def format_amount(amount: float) -> str:
    """
    Formats amount appending 'm' for millions or 'k' for thousands, rounding to two decimals.

    Parameters:
    - amount (float)

    Returns:
    - str
    """
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.2f}".rstrip("0").rstrip(".") + "m"
    elif amount >= 1_000:
        return f"{amount / 1_000:.2f}".rstrip("0").rstrip(".") + "k"
    else:
        return f"{amount:.2f}".rstrip("0").rstrip(".")


def extract_feature_names(pipeline, input_data):
    """
    Passes input_data through all transformers in the pipeline to extract feature names.
    Does not require 'get_feature_names_out' as long as transformers operate on pandas dfs instead of ndarrays

    :param pipeline: A scikit-learn Pipeline object.
    :param input_data: Dummy input data as a pandas DataFrame.
    :return: List of output feature names after all transformations.
    """
    transformed_data = input_data
    for name, transformer in pipeline.steps[
        :-1
    ]:  # Exclude the last step if it's a model
        # print(name)
        transformed_data = transformer.transform(transformed_data)

        # If the transformer reduces or modifies the feature space, adapt accordingly
        if hasattr(transformer, "get_feature_names_out"):
            # For transformers that support it, directly obtain the feature names
            feature_names = transformer.get_feature_names_out()
        else:
            # Otherwise, infer feature names (if possible, depending on the output)
            if isinstance(transformed_data, pd.DataFrame):
                feature_names = transformed_data.columns.tolist()
            else:
                # If the output is a NumPy array, generate placeholder names
                feature_names = [
                    f"feature_{i}" for i in range(transformed_data.shape[1])
                ]
    return feature_names


def get_model_feature_importances(model_config, transformed_data):
    # TODO: importances are missing
    # TODO: need to make importances work with with feature transformers since
    importances = model_config.test_data.feature_importances

    # TODO: implement feature importances setting inot the pipeline itself instead of having to do it manually here:
    model = model_config.test_data.test_model.named_steps["model"]
    feature_importances = model.feature_importances_

    feature_names = extract_feature_names(
        model_config.test_data.test_model, transformed_data.sample(10, random_state=42)
    )
    assert len(feature_names) == len(
        feature_importances
    ), "The length of feature names and importances does not match."
    # feature_importances = zip(feature_names, feature_importances)
    feature_importances = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    return feature_importances
