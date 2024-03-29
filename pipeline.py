from collections.abc import Iterable
from typing import Dict, Union, Optional, Any, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    fbeta_score,
    classification_report,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_error,
    log_loss,
)
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_predict,
    RandomizedSearchCV,
    StratifiedKFold,
    ShuffleSplit,
    KFold,
    cross_val_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from shared import stats_utils
from shared.definitions import TuningResult
from shared.ml_config_core import (
    ModelConfig,
    TestTrainData,
    CMResultsData,
    ModelTrainingResult,
    TuneType,
    ModelType,
    ModelPipelineConfig,
)

metrics_decls = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


class ModelProbabilitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        if self.model is None:
            raise ValueError("Base estimator is not set.")
        self.model.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self.model)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise ValueError(
                "The base estimator does not support probability predictions."
            )

    def set_params(self, **params):
        # Forward parameter setting to the underlying model
        valid_params = self.model.get_params(deep=True)
        for param, value in params.items():
            if param in valid_params:
                setattr(self.model, param, value)
        return self


class ThresholdTransformer(BaseEstimator, TransformerMixin):
    """
    Used for selecting a custom threshold on top of a parameter tuning with AUC or other threshold metrics
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        # No fitting necessary, so just return self
        return self

    def transform(self, X):
        # Assuming X is the output of predict_proba from the previous step
        # and contains probabilities for the positive class in the second column
        return (X[:, 1] >= self.threshold).astype(int)

    def predict(self, X):
        # Ensure this method uses the second column (positive class probabilities) for thresholding
        return (X[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return X


def __get_ensemble_classifier_factory(config, model_params: dict = None):
    if model_params is None:
        model_params = {}

    model_pipelines = []
    for i, m in enumerate(config.model):
        p = Pipeline([(f"clf{i}", m(**model_params))])
        # p = Pipeline([(m.__name__, m(**model_params))])
        model_pipelines.append((f"cl_{m.__name__}_{i}", p))

    return lambda: config.ensemble_classifier(
        estimators=model_pipelines,
    )


def __get_ensemble_classifier_pipeline(config, model_params: dict = None):
    index = len(config.model)

    voting_clf = Pipeline(
        [(f"clf{index}", __get_ensemble_classifier_factory(config, model_params)())]
    )
    return voting_clf


def get_config_param_grid(config: ModelConfig):
    uses_custom_threshold = any(
        "threshold__" in key for key in config.param_grid.keys()
    )

    if uses_custom_threshold:
        param_grid = {}
        for k, v in config.param_grid.items():
            param_grid[k.replace("model__", "model_proba__")] = config.param_grid[k]

    else:
        param_grid = config.param_grid

    return param_grid


def __get_model_params(
    config: ModelConfig,
    enable_hyperparameter_tuning=False,
    uses_custom_threshold=False,
    _best_params=None,
):
    best_params = {}

    if _best_params is None:
        target_params = config.best_params
    else:
        target_params = _best_params

    if target_params is None and enable_hyperparameter_tuning == False:
        raise Exception(
            f"{config.model_key} missing best_params and enable_hyperparameter_tuning = False"
        )
    elif target_params is None:
        target_params = {}

    if not enable_hyperparameter_tuning:
        target_params_2 = {}
        for k, v in target_params.items():
            if "model__" in k:
                target_params_2[k] = v
        target_params = target_params_2

    for k, v in target_params.items():
        if uses_custom_threshold and enable_hyperparameter_tuning:
            best_params[k.replace("model__", "model_proba__")] = target_params[k]
        else:
            best_params[k.replace("model__", "")] = target_params[k]

    if uses_custom_threshold:
        builtin_params = {}
        for k, v in config.builtin_params.items():
            if enable_hyperparameter_tuning:
                builtin_params[
                    k.replace("model__", "model_proba__")
                ] = config.builtin_params[k]
            else:
                builtin_params[k.replace("model__", "")] = config.builtin_params[k]
    else:
        builtin_params = config.builtin_params

    model_params = {
        **builtin_params,
        **(best_params if not enable_hyperparameter_tuning else {}),
    }
    return model_params


def get_pipeline(
    model_pipeline_config: ModelPipelineConfig,
    enable_transformer_hyperparameter_tuning=False,
    enable_model_hyperparameter_tuning=False,
    best_params=None,
):
    model_config = model_pipeline_config.model_config
    transformer_config = model_pipeline_config.transformer_config
    pipeline_steps = []

    # If necessary add preprocessing function
    if model_config["preprocessing"]:
        if isinstance(model_config["preprocessing"], Iterable):
            pipeline_steps.extend(model_config["preprocessing"])
        else:
            pipeline_steps.append(("preprocessing", model_config.preprocessing))

    # 2. Feature transformer/feature engineering
    if transformer_config is not None:
        for tr in transformer_config.transformers:
            if enable_transformer_hyperparameter_tuning:
                pipeline_steps.append(tr.create({}))
            else:
                # Each transformer will select appropriate params for it
                pipeline_steps.append(tr.create(best_params))

    # 3. Balancing
    if model_config.balancing_config is not None:
        print(
            f"Using balancing config: {model_config.balancing_config.__class__.__name__}"
        )
        balancing_step = model_config.balancing_config.get_pipeline()
        pipeline_steps.append(balancing_step)

    # TODO: support for custom thresholds (i.e. threshold tunning is disabled for now)
    uses_custom_threshold = False

    model_params = __get_model_params(
        model_config,
        enable_model_hyperparameter_tuning,
        uses_custom_threshold=uses_custom_threshold,
        _best_params=best_params,
    )

    # TODO: support for ensemble models disabled for now:
    # if isinstance(config.model, list):
    #     if config.ensemble_classifier is None:
    #         raise Exception(
    #             "Multiple model provided but no 'ensemble_classifier' specified!"
    #         )
    #     voting_clf = __get_ensemble_classifier_pipeline(config, model_params)
    #
    #     pipeline_steps.append(
    #         (
    #             "model",
    #             voting_clf,
    #         )
    #     )
    def remove_columns_with_prefix(X, prefix):
        return X.loc[:, ~X.columns.str.startswith(prefix)]

    # Drop target columns in case they were included
    drop_target_cols_transformer = FunctionTransformer(
        remove_columns_with_prefix, kw_args={"prefix": "target__"}
    )

    pipeline_steps.append(("remove_columns", drop_target_cols_transformer))

    if uses_custom_threshold and enable_model_hyperparameter_tuning:
        print("uses_custom_threshold")
        pipeline_steps.append(
            (
                "model_proba",
                ModelProbabilitiesTransformer(model=model_config.model(**model_params)),
            )
        )

    else:
        pipeline_steps.append(
            (
                "model",
                model_config.model(**model_params),
            )
        )

    if model_config.balancing_config is not None:
        pipeline = ImbPipeline(pipeline_steps)
    else:
        pipeline = Pipeline(pipeline_steps)

    return pipeline


def cap_inf(_df):
    for col in _df.columns:
        if "ratio" in col:
            max_val = _df[col].replace(np.inf, np.nan).max()
            _df[col].replace(np.inf, max_val, inplace=True)

            min_val = _df[col].replace(-np.inf, np.nan).min()
            _df[col].replace(-np.inf, min_val, inplace=True)


def clean_params(best_params_):
    return {k.replace("model_proba__", "model__"): v for k, v in best_params_.items()}


def _get_cv_for_config(model_type_a: ModelType, cv=None):
    if cv is None:
        if model_type_a.value == ModelType.Classifier.value:
            cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        elif model_type_a.value == ModelType.Regressor.value:
            cv = KFold(n_splits=5, random_state=42, shuffle=True)

    return cv


def _get_search_results_table(grid_search: BaseSearchCV):
    """
    Convert CV results to a pandas dataframe.
    :param grid_search:
    :return:
    """
    cv_results = grid_search.cv_results_

    results_df = pd.DataFrame(cv_results)

    param_columns = [col for col in results_df.columns if col.startswith("param_")]
    score_column = "mean_test_score"  # or choose the metric you're interested in
    relevant_columns = param_columns + [score_column]

    final_df = results_df[relevant_columns]

    final_df.columns = [col.replace("param_", "") for col in final_df.columns]
    return final_df


def _get_n_jobs(model_pipeline_config):
    """
    When training on the GPU do not attempt to run multiple jobs at once since we'll run out of memory
    :return:
    """
    n_jobs = -1
    if "task_type" in model_pipeline_config.model_config.builtin_params:
        if model_pipeline_config.model_config.builtin_params["task_type"] == "GPU":
            n_jobs = 1
    return n_jobs


def _tune_transformer_params(
    model_pipeline_config: ModelPipelineConfig,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cv=2,
):
    """
    Tunes  preprocessing transformer parameters using fixed model hyperparameters.
    This is a somewhat suboptimal compromise because doing nexted tunning (i.e. tunning model hyperparameters for every
    combination of transformer params would be computionally very expensive

    :param model_pipeline_config:
    :param features:
    :param labels:
    :param cv:
    :return:
    """
    tunning_target = get_tuning_target(model_pipeline_config.model_config)

    # TODO: hack, fix:
    # TODO: BEST PARAMS ARE NOT HARDCODED WHEN TUNING TRANSFORMERS, FIX

    best_params = model_pipeline_config.model_config.default_params

    pipeline = get_pipeline(
        model_pipeline_config=model_pipeline_config,
        enable_transformer_hyperparameter_tuning=True,
        best_params=best_params,
    )

    transformer_param_grid = (
        model_pipeline_config.transformer_config.get_feature_search_grid()
    )

    grid_search = GridSearchCV(
        pipeline,
        scoring=tunning_target,
        param_grid=transformer_param_grid,
        cv=cv,
        verbose=1,
        n_jobs=_get_n_jobs(model_pipeline_config),
    )
    grid_search.fit(features.copy(), labels)

    print(f"Best Transformer Parameters: {grid_search.best_params_}")

    all_cv_results = _get_search_results_table(grid_search)
    return grid_search, all_cv_results


def _update_dynamic_builtin_params(params, model_pipeline_config, features):
    # TODO: hack, fix (cat_features list needs to be specified manually for CatBoost, but is not used by XGBoost)
    if "CatBoost" in model_pipeline_config.model_config.model_key:
        params = {
            **params,
            "model__cat_features": list(
                features.select_dtypes(include=["category"]).columns
            ),
        }
    return params


def _tune_model_params(
    best_transformer_params, model_pipeline_config, features, labels, cv=5
) -> Tuple[RandomizedSearchCV, pd.DataFrame]:
    """
    New streamlined version of hyperparameter tuning for a given model using fixed preprocessing parameters. Note that
    this reuses the pipeline returned by '_tune_transformer_params'
    :param best_preprocessing_estimator:
    :param model_pipeline_config:
    :param X_train:
    :param y_train:
    :param cv:
    :return:
    """
    tunning_target = get_tuning_target(model_pipeline_config.model_config)

    # Ensure the model in the pipeline is not fixed and can be tuned
    tunning_target = get_tuning_target(model_pipeline_config.model_config)

    best_params = {**best_transformer_params}
    pipeline = get_pipeline(
        model_pipeline_config=model_pipeline_config,
        enable_model_hyperparameter_tuning=True,
        # TODO: add fixed model params to start with to each 'ModelPipelineConfig' in project config files
        best_params=best_params,
    )

    model_param_distributions = model_pipeline_config.model_config.param_grid

    # Perform Randomized Search to tune model hyperparameters
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=model_param_distributions,
        # n_iter=5,
        scoring=tunning_target,
        n_iter=model_pipeline_config.model_config.search_n_iter,
        cv=cv,
        verbose=1,
        n_jobs=_get_n_jobs(model_pipeline_config),
    )
    random_search.fit(features, labels)
    all_cv_results = _get_search_results_table(random_search)

    print(f"Best Model Parameters: {random_search.best_params_}")
    return random_search, all_cv_results


def _build_tuning_result(
    model_search: RandomizedSearchCV,
    pipeline_search: GridSearchCV,
    pipeline_config: ModelPipelineConfig,
    model_key: str,
    transformer_tun_all_civ_results: pd.DataFrame,
    hyper_param_all_cv_results: pd.DataFrame,
    features=pd.DataFrame,
    labels=pd.Series,
) -> TuningResult:
    """
    Build the tuning results for a given best estimator and tuning history, split from old 'run_tunning_for_config_OLD_2'
    TODO: only included results from '_tune_model_params' for now, also include data for '_tune_transformer_params'
    :return:
    """
    model_type_a = pipeline_config.model_config.get_type()
    tunning_target = pipeline_config.model_config.tunning_func_target
    tune_type = TuneType.Grid

    predictions = model_search.best_estimator_.predict(features)
    result_df = pd.DataFrame({"predictions": predictions, "labels": labels})

    if model_type_a.value == ModelType.Regressor.value:
        predictions = model_search.best_estimator_.predict(features)
        mse = mean_squared_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        report_dict = {"mean_squared_error": mse, "r2_score": r2}
    else:
        report_dict = classification_report(labels, predictions, output_dict=True)
    training_data = []

    for i in range(0, len(model_search.cv_results_["mean_fit_time"])):
        vals = {
            "number": i,
            "mean_test_score": float(model_search.cv_results_["mean_test_score"][i]),
            "rank_test_score": float(model_search.cv_results_["rank_test_score"][i]),
            "mean_fit_time": float(model_search.cv_results_["mean_fit_time"][i]),
            "params": model_search.cv_results_["params"][i],
        }
        training_data.append(vals)

    dataset_description = {
        "num_rows": len(features),
        "feature_types": {col: features[col].dtype.name for col in features.columns},
        "trials": training_data,
        "label_distribution": {},
    }

    tuning_result = TuningResult(
        model_key=model_key,
        model_pipeline_config=pipeline_config,
        model_params={**clean_params(model_search.best_params_)},
        pipeline_params={**clean_params(pipeline_search.best_params_)}
        if pipeline_search
        else {},
        dataset_description=dataset_description,
        all_scores={
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in model_search.cv_results_.items()
        },
        best_score=float(model_search.best_score_),
        result_report=report_dict,
        transformer_tun_all_civ_results=transformer_tun_all_civ_results,
        hyper_param_all_cv_results=hyper_param_all_cv_results,
    )
    return tuning_result


def _get_features_labels(df):
    target_cols = [col for col in df.columns if col.startswith("target__")]
    if len(target_cols) > 1:
        raise Exception(f"Multiple 'target__' columns found: {target_cols}")
    elif len(target_cols) == 1:
        labels = df[target_cols[0]]
    else:
        raise Exception("No 'target__' columns found.")

    return df, labels


def run_fixed_transformers(fixed_preprocessors: list, df: pd.DataFrame):
    if len(fixed_preprocessors) > 0:
        for t_def in fixed_preprocessors:
            t = t_def()
            df = t.transform(df)
            print(1)
    return df


def run_tunning_for_config(
    model_key: str,
    pipeline_config: ModelPipelineConfig,
    df: pd.DataFrame,
    cv=None,
) -> TuningResult:
    # Run any fixed preprocessors that are not used for tuning
    # if len(pipeline_config.transformer_config.fixed_preprocessors) > 0:
    #     for t_def in pipeline_config.transformer_config.fixed_preprocessors:
    #         t = t_def()
    #         df = t.transform(df)
    if pipeline_config.transformer_config.fixed_preprocessors:
        df = run_fixed_transformers(
            pipeline_config.transformer_config.fixed_preprocessors, df
        )
    # The target column is prefixed with target__, df should only have one
    features, labels = _get_features_labels(df)
    tuning_grid = pipeline_config.transformer_config.get_feature_search_grid()

    # Are there any combinations of transformers
    skip_feature_tuning = all(
        [True if len(v) == 1 else False for v in tuning_grid.values()]
    )

    # if len(tuning_grid) > 0:
    # if skip_feature_tuning:
    #     best_transformer_params = {k: v[0] for k, v in tuning_grid.items()}
    #     best_preproc_transformer_sarch = None
    #     print(f"Skip transformer tuning, only 1 pipeline: {best_transformer_params} ")
    if len(tuning_grid) > 0:
        (
            best_preproc_transformer_sarch,
            transformer_tun_all_civ_results,
        ) = _tune_transformer_params(pipeline_config, features, labels, cv=5)

        # TODO: export both this and hyperparams
        best_transformer_params = best_preproc_transformer_sarch.best_params_
    else:
        best_preproc_transformer_sarch = None
        transformer_tun_all_civ_results = None
        best_transformer_params = {}

    search_results, hyper_param_all_cv_results = _tune_model_params(
        best_transformer_params, pipeline_config, features=features, labels=labels, cv=5
    )

    tuning_results = _build_tuning_result(
        model_search=search_results,
        pipeline_search=best_preproc_transformer_sarch,
        pipeline_config=pipeline_config,
        model_key=model_key,
        features=features,
        labels=labels,
        transformer_tun_all_civ_results=transformer_tun_all_civ_results,
        hyper_param_all_cv_results=hyper_param_all_cv_results,
    )

    return tuning_results


def get_tuning_target(config: ModelConfig):
    model_type_a = config.get_type()
    if isinstance(config.tunning_func_target, _BaseScorer):
        tunning_target = config.tunning_func_target
    elif config.tunning_func_target is not None:
        tunning_target = make_scorer(config.tunning_func_target)
    else:
        if model_type_a.value == ModelType.Classifier.value:
            tunning_target = "f1_macro"
        elif model_type_a.value == ModelType.Regressor.value:
            tunning_target = "neg_mean_squared_error"
        else:
            raise Exception("Unknown model type")
    return tunning_target


def calculate_classification_metrics(
    labels: pd.Series,
    all_predictions: pd.Series,
    all_probabilities: Optional[pd.Series] = None,
) -> Dict:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": accuracy_score(labels, all_predictions),
        "precision_macro": precision_score(labels, all_predictions, average="macro"),
        "recall_macro": recall_score(labels, all_predictions, average="macro"),
        "f1_macro": f1_score(labels, all_predictions, average="macro"),
    }

    if len(labels.unique()) <= 2:
        metrics = {
            **metrics,
            "target_f1": f1_score(labels, all_predictions, pos_label=1),
            "target_recall": recall_score(labels, all_predictions, pos_label=1),
            "target_precision": precision_score(labels, all_predictions, pos_label=1),
            "fbeta_1.5": fbeta_score(labels, all_predictions, pos_label=1, beta=1.5),
            "fbeta_2.5": fbeta_score(labels, all_predictions, pos_label=1, beta=2.5),
            "fbeta_4.0": fbeta_score(labels, all_predictions, pos_label=1, beta=4.0),
        }

    if all_probabilities is not None:
        metrics["log_loss"] = log_loss(labels, all_probabilities)

    return metrics


def get_deterministic_train_test_split(features_all, labels_all):
    X_train, X_test, y_train, y_test = train_test_split(
        features_all, labels_all, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def run_pipeline_config(
    tuning_result: TuningResult,
    df: pd.DataFrame,
    cv: Optional[
        Any
    ] = None,  # Assume CVType is a type hint for cross-validation strategies
    VERBOSE: bool = True,
) -> ModelTrainingResult:
    """
    Execute the training pipeline for a given configuration, supporting both classifiers and regressors.

    Parameters:
    - config: Configuration for the model.
    - features: Input features for model training.
    - labels: Target labels for model training.
    - model_type: Specifies whether the model is a classifier or regressor.
    - export_prod: Flag to indicate whether to export the model for production.
    - export_test: Flag to indicate whether to export test results.
    - cv: Cross-validation splitting strategy.
    - VERBOSE: Flag to control verbosity of the output.

    Returns:
    - ModelTrainingResult: Object containing the results of the training process.
    """
    if hasattr(
        tuning_result.model_pipeline_config.transformer_config, "fixed_preprocessors"
    ):
        if tuning_result.model_pipeline_config.transformer_config.fixed_preprocessors:
            df = run_fixed_transformers(
                tuning_result.model_pipeline_config.transformer_config.fixed_preprocessors,
                df,
            )

    def calculate_regression_metrics(labels: pd.Series, predictions: pd.Series) -> Dict:
        """Calculate regression metrics."""
        metrics = {
            "mse": mean_squared_error(labels, predictions),
            "rmse": np.sqrt(
                mean_squared_error(labels, predictions)
            ),  # Root Mean Squared Error
            "mae": mean_absolute_error(labels, predictions),  # Mean Absolute Error
            "r2": r2_score(labels, predictions),  # R-squared
            "medae": median_absolute_error(
                labels, predictions
            ),  # Median Absolute Error
        }

        # MAPE calculation, handling cases where labels could be 0
        mape = (
            np.mean(
                np.abs((labels - predictions) / labels)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            * 100
        )
        metrics["mape"] = mape  # Mean Absolute Percentage Error

        return metrics

    model_pipeline_config = tuning_result.model_pipeline_config
    features_all, labels_all = _get_features_labels(df)

    X_train, X_test, y_train, y_test = get_deterministic_train_test_split(
        features_all, labels_all
    )
    # X_train, X_test, y_train, y_test = train_test_split(
    #     features_all, labels_all, test_size=0.2, random_stateint=42
    # )

    features = X_train
    labels = y_train

    best_params = tuning_result.get_best_params()

    model_type = model_pipeline_config.model_config.get_type()
    res = ModelTrainingResult()
    cv = _get_cv_for_config(
        model_type_a=model_pipeline_config.model_config.get_type(), cv=cv
    )

    if cv is not None:
        cv_pipeline = get_pipeline(model_pipeline_config, best_params=best_params)
        splits = list(cv.split(features, labels))

        all_predictions = cross_val_predict(
            cv_pipeline, features, labels, cv=splits, method="predict", n_jobs=-1
        )
        all_predictions = pd.Series(all_predictions, index=labels.index)

        if model_type.value == ModelType.Classifier.value:
            all_probabilities = cross_val_predict(
                cv_pipeline,
                features,
                labels,
                cv=splits,
                method="predict_proba",
                n_jobs=-1,
            )
            all_probabilities = pd.DataFrame(all_probabilities, index=labels.index)

            if "threshold__threshold" in best_params:
                threshold = best_params["threshold__threshold"]
                positive_class_probabilities = all_probabilities.iloc[:, 1]
                all_predictions = (positive_class_probabilities >= threshold).astype(
                    int
                )

            cv_metrics_results = calculate_classification_metrics(
                labels, all_predictions, all_probabilities
            )
            res.cm_data = CMResultsData(
                cv_pipeline,
                labels,
                features,
                all_predictions,
                all_probabilities,
                class_accuracies={},
            )

        elif model_type.value == ModelType.Regressor.value:
            cv_metrics_results = calculate_regression_metrics(labels, all_predictions)
            res.cm_data = CMResultsData(
                cv_pipeline,
                labels,
                features,
                all_predictions,
                class_accuracies={},
                probabilities=pd.DataFrame(),
            )
        else:
            raise Exception("Unknown model type")

        res.cv_metrics = cv_metrics_results
        res.cv_metrics["n_samples"] = len(features)

    test_pipeline = get_pipeline(model_pipeline_config, best_params=best_params)

    print(
        f"Training: {model_pipeline_config.model_config.model_key} with: {best_params}"
    )
    test_pipeline.fit(X_train, y_train)

    if "preprocessing" in test_pipeline.named_steps:
        preprocessor = test_pipeline.named_steps["preprocessing"]
        X_test_transformed = preprocessor.transform(X_test)
    else:
        X_test_transformed = X_test

    evaluation_method = (
        stats_utils.evaluate_classifier_model
        if model_type == ModelType.Classifier
        else stats_utils.evaluate_regressor_model
    )

    (
        metrics,
        predictions,
        probabilities,
        probabilities_match_id,
    ) = evaluation_method(test_pipeline, X_test, y_test)

    res.test_data = TestTrainData(
        test_model=test_pipeline,
        x_test_transformed=X_test_transformed,
        x_test=X_test,
        x_train=X_train,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics,
        class_accuracies=None,
    )

    return res


def build_cv_results_table(cv_results: Dict[str, ModelTrainingResult], VERBOSE=True):
    results_table = {}
    for key in cv_results:
        model_metrics = cv_results[key].cv_metrics
        if VERBOSE:
            print(f"\nResults for {key}")
            print(f'n-samples: {model_metrics["n_samples"]}')

        if "all_scores" in model_metrics:
            if VERBOSE:
                print(f'Best Score: {model_metrics["best_score"]:.3f}')
                print(f'Best Parameters: {model_metrics["best_params"]}')

            all_scores = model_metrics["all_scores"]
            param_scores = [
                (
                    round(all_scores["mean_test_score"][i], 3),
                    dict(
                        (param, all_scores["param_" + param][i]) for param in param_grid
                    ),
                )
                for i in range(len(all_scores["mean_test_score"]))
            ]
            if VERBOSE:
                print("All parameter sets and their scores:")
            for score, params in param_scores:
                print((score, params))

        else:
            results_table[key] = {}
            for metric in model_metrics.keys():
                results_table[key][metric] = round(np.mean(model_metrics[metric]), 3)
                if VERBOSE:
                    print(f"{metric}: {np.mean(model_metrics[metric]):.3f}")

        bins = {">0.5": {}, "0.05 - 0.5": {}}

        for bin_name, features in bins.items():
            sorted_features = dict(
                sorted(
                    features.items(), key=lambda item: round(item[1], 2), reverse=True
                )
            )
            if VERBOSE:
                print(f"{bin_name}: {sorted_features}")

    metrics_df = pd.DataFrame.from_dict(results_table, orient="index")

    try:
        metrics_df = metrics_df.sort_values(by=["f1_macro"], ascending=False)
    except:
        metrics_df = metrics_df.sort_values(by=["mse"], ascending=False)
    return metrics_df
