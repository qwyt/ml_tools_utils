import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, Union, Optional, Any

import numpy as np
import optuna
import pandas as pd
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    fbeta_score,
    classification_report, mean_squared_error, r2_score, median_absolute_error, mean_absolute_error, log_loss,
)
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_predict,
    RandomizedSearchCV, StratifiedKFold, ShuffleSplit, KFold, cross_val_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from shared import stats_utils
from shared.definitions import TuningResult, TuningResultBestParams
from shared.ml_config_core import (
    ModelConfig,
    BalancingConfig,
    TestTrainData,
    CMResultsData,
    ModelTrainingResult,
    TuneType,
    ModelType, ModelPipelineConfig,
)

metrics_decls = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


# TODO: move to ml_config_core
# class ModelProbabilitiesTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, model):
#         self.model = model
#
#     def fit(self, X, y=None):
#         self.model.fit(X, y)
#         return self
#
#     def transform(self, X):
#         # Check if the model has 'predict_proba' method
#         if hasattr(self.model, "predict_proba"):
#             return self.model.predict_proba(X)
#         else:
#             raise ValueError("The model does not support probability predictions.")
class ModelProbabilitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        # Ensure the model is properly initialized
        if self.model is None:
            raise ValueError("Base estimator is not set.")
        self.model.fit(X, y)
        return self

    def transform(self, X):
        # Check if fit has been called by verifying the presence of fitted attributes
        check_is_fitted(self.model)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise ValueError("The base estimator does not support probability predictions.")

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
    uses_custom_threshold = any("threshold__" in key for key in config.param_grid.keys())

    if uses_custom_threshold:
        param_grid = {}
        for k, v in config.param_grid.items():
            param_grid[k.replace("model__", "model_proba__")] = config.param_grid[k]

    else:
        param_grid = config.param_grid

    return param_grid


def __get_model_params(config: ModelConfig,
                       enable_hyperparameter_tuning=False,
                       uses_custom_threshold=False,
                       _best_params=None):
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
                builtin_params[k.replace("model__", "model_proba__")] = config.builtin_params[k]
            else:
                builtin_params[k.replace("model__", "")] = config.builtin_params[k]
    else:
        builtin_params = config.builtin_params

    model_params = {
        **builtin_params,
        **(best_params if not enable_hyperparameter_tuning else {}),
    }
    return model_params


def get_pipeline(model_pipeline_config: ModelPipelineConfig,

                 enable_transformer_hyperparameter_tuning=False,
                 enable_model_hyperparameter_tuning=False,
                 best_params=None
                 ):
    model_config = model_pipeline_config.model_config
    transformer_config = model_pipeline_config.transformer_config
    pipeline_steps = []

    # 1. If necessary add preprocessing function
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
        # if isinstance(config.balancing_config, BalancingConfig):
        print(f"Using balancing config: {model_config.balancing_config.__class__.__name__}")
        balancing_step = model_config.balancing_config.get_pipeline()
        pipeline_steps.append(balancing_step)

    # TODO: support for custom thresholds (i.e. threshold tunning is disabled for now)
    uses_custom_threshold = False

    # 4. Add model
    # TODO: refactor '__get_model_params' to use an approach similar to what BaseTransformer.create is doing instead
    # TODO: so split into two branches for tuning and not.
    model_params = __get_model_params(
        model_config,
        enable_model_hyperparameter_tuning,
        uses_custom_threshold=uses_custom_threshold,
        _best_params=best_params
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
    drop_target_cols_transformer = FunctionTransformer(remove_columns_with_prefix, kw_args={'prefix': 'target__'})

    pipeline_steps.append(('remove_columns', drop_target_cols_transformer))

    if uses_custom_threshold and enable_model_hyperparameter_tuning:
        print("uses_custom_threshold")
        # Add ModelProbabilitiesTransformer with the model as an argument
        pipeline_steps.append(
            (
                "model_proba",  # Output of this step will be probabilities
                ModelProbabilitiesTransformer(model=model_config.model(**model_params))
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

    # pipeline_steps = []
    #
    # uses_custom_threshold = any(
    #     "threshold__" in key for key in config.param_grid.keys()) and config.get_type().value != ModelType.Regressor
    #
    # uses_select_k_best = any(
    #     "feature_selection_k_best" in key for key in
    #     config.param_grid.keys()) and config.get_type().value != ModelType.Regressor
    #
    # if config["preprocessing"]:
    #     if isinstance(config["preprocessing"], Iterable):
    #         pipeline_steps.extend(config["preprocessing"])
    #     else:
    #         pipeline_steps.append(("preprocessing", config["preprocessing"]))
    #
    # if config.balancing_config is not None:
    #     # if isinstance(config.balancing_config, BalancingConfig):
    #     print(f"Using balancing config: {config.balancing_config.__class__.__name__}")
    #     balancing_step = config.balancing_config.get_pipeline()
    #     pipeline_steps.append(balancing_step)
    #
    # model_params = __get_model_params(
    #     config,
    #     enable_hyperparameter_tuning,
    #     uses_custom_threshold=uses_custom_threshold,
    #     _best_params=best_params
    # )
    #
    # if uses_select_k_best:
    #     pipeline_steps.append(('feature_selection_k_best', SelectKBest()))
    #
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
    # elif uses_custom_threshold and enable_hyperparameter_tuning:
    #     print("uses_custom_threshold")
    #     # Add ModelProbabilitiesTransformer with the model as an argument
    #     pipeline_steps.append(
    #         (
    #             "model_proba",  # Output of this step will be probabilities
    #             ModelProbabilitiesTransformer(model=config.model(**model_params))
    #         )
    #     )
    #
    # else:
    #     pipeline_steps.append(
    #         (
    #             "model",
    #             config.model(**model_params),
    #         )
    #     )
    #
    # # Keep as last step
    #
    # # TODO: must verify that binary scorer like f1 is used
    # if uses_custom_threshold and enable_hyperparameter_tuning:
    #     pipeline_steps.append(("threshold", ThresholdTransformer()))
    #
    # if config.balancing_config is not None:
    #     pipeline = ImbPipeline(pipeline_steps)
    # else:
    #     pipeline = Pipeline(pipeline_steps)
    # return pipeline


# def get_pipeline_OLD_2(config: ModelConfig, enable_hyperparameter_tuning=False, best_params=None):
#     # Used before 'ModelPipelineConfig' was implemented
#     pipeline_steps = []
#
#     uses_custom_threshold = any(
#         "threshold__" in key for key in config.param_grid.keys()) and config.get_type().value != ModelType.Regressor
#
#     uses_select_k_best = any(
#         "feature_selection_k_best" in key for key in
#         config.param_grid.keys()) and config.get_type().value != ModelType.Regressor
#
#     if config["preprocessing"]:
#         if isinstance(config["preprocessing"], Iterable):
#             pipeline_steps.extend(config["preprocessing"])
#         else:
#             pipeline_steps.append(("preprocessing", config["preprocessing"]))
#
#     if config.balancing_config is not None:
#         # if isinstance(config.balancing_config, BalancingConfig):
#         print(f"Using balancing config: {config.balancing_config.__class__.__name__}")
#         balancing_step = config.balancing_config.get_pipeline()
#         pipeline_steps.append(balancing_step)
#
#     model_params = __get_model_params(
#         config,
#         enable_hyperparameter_tuning,
#         uses_custom_threshold=uses_custom_threshold,
#         _best_params=best_params
#     )
#
#     if uses_select_k_best:
#         pipeline_steps.append(('feature_selection_k_best', SelectKBest()))
#
#     if isinstance(config.model, list):
#         if config.ensemble_classifier is None:
#             raise Exception(
#                 "Multiple model provided but no 'ensemble_classifier' specified!"
#             )
#         voting_clf = __get_ensemble_classifier_pipeline(config, model_params)
#
#         pipeline_steps.append(
#             (
#                 "model",
#                 voting_clf,
#             )
#         )
#     elif uses_custom_threshold and enable_hyperparameter_tuning:
#         print("uses_custom_threshold")
#         # Add ModelProbabilitiesTransformer with the model as an argument
#         pipeline_steps.append(
#             (
#                 "model_proba",  # Output of this step will be probabilities
#                 ModelProbabilitiesTransformer(model=config.model(**model_params))
#             )
#         )
#
#     else:
#         pipeline_steps.append(
#             (
#                 "model",
#                 config.model(**model_params),
#             )
#         )
#
#     # Keep as last step
#
#     # TODO: must verify that binary scorer like f1 is used
#     if uses_custom_threshold and enable_hyperparameter_tuning:
#         pipeline_steps.append(("threshold", ThresholdTransformer()))
#
#     if config.balancing_config is not None:
#         pipeline = ImbPipeline(pipeline_steps)
#     else:
#         pipeline = Pipeline(pipeline_steps)
#     return pipeline


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
    # Assuming 'grid_search' is your GridSearchCV object and it's already fitted
    cv_results = grid_search.cv_results_

    # Create a DataFrame from cv_results
    results_df = pd.DataFrame(cv_results)

    # Filter columns to include only parameter columns and the mean test score
    param_columns = [col for col in results_df.columns if col.startswith('param_')]
    score_column = 'mean_test_score'  # or choose the metric you're interested in
    relevant_columns = param_columns + [score_column]

    # Create a final DataFrame with just the relevant columns
    final_df = results_df[relevant_columns]

    # Rename columns for clarity if needed
    final_df.columns = [col.replace('param_', '') for col in final_df.columns]

    return final_df


def _tune_transformer_params(model_pipeline_config: ModelPipelineConfig,
                             features: pd.DataFrame,
                             labels: pd.DataFrame,
                             cv=2):
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
    # Construct the preprocessing pipeline
    # preprocessing_steps = [(f"trans_{trans.transformer_name}", trans.create({})) for trans in
    #                        model_pipeline_config.transformer_config.transformers]
    # fixed_model = model_pipeline_config.model_config.model()  # Instantiate the fixed model

    # Combine preprocessing steps with the fixed model
    # pipeline = Pipeline(steps=preprocessing_steps + [('model', fixed_model)])
    tunning_target = get_tuning_target(model_pipeline_config.model_config)

    # TODO: hack, fix:
    # TODO: BEST PARAMS ARE NOT HARDCODED WHEN TUNING TRANSFORMERS, FIX

    best_params = model_pipeline_config.model_config.default_params

    pipeline = get_pipeline(
        model_pipeline_config=model_pipeline_config,
        enable_transformer_hyperparameter_tuning=True,
        # TODO: add fixed model params to start with to each 'ModelPipelineConfig' in project config files
        best_params=best_params
    )

    # Define the grid for the transformer parameters
    transformer_param_grid = model_pipeline_config.transformer_config.get_feature_search_grid()

    # Perform Grid Search to tune transformer parameters
    grid_search = GridSearchCV(pipeline,
                               scoring=tunning_target,
                               param_grid=transformer_param_grid,
                               cv=cv,
                               verbose=1,
                               # n_iter=5,
                               # n_jobs=1
                               n_jobs=-1,
                               )
    # grid_search = RandomizedSearchCV(pipeline,
    #                                  scoring=tunning_target,
    #                                  param_distributions=transformer_param_grid,
    #                                  cv=cv,
    #                                  verbose=1,
    #                                  n_iter=5,
    #                                  # TODO: add debug option, module autoreload breaks if multiproc is used, so only use for prod, not when iterating
    #                                  # n_jobs=1
    #                                  n_jobs=-1,
    #                                  )
    grid_search.fit(features.copy(), labels)

    print(f"Best Transformer Parameters: {grid_search.best_params_}")

    all_cv_results = _get_search_results_table(grid_search)

    # TODO: VERIFY THAT DF IS ACTUALLY BEING TRANSFORMED INSIDE GRID SEARCH BECAUSE THE RESULTS FOR ALL PARAMS ARE ALWAYS THE SAME !

    # print(f"TODO: TODO: EXPORT THIS: tranformer params:\n{all_scores}")

    return grid_search, all_cv_results


def _tune_model_params(best_transformer_params, model_pipeline_config,
                       features,
                       labels,
                       cv=5) -> RandomizedSearchCV:
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
    # Set the best preprocessing parameters in the pipeline
    # pipeline = best_preprocessing_estimator
    # pipeline = best_preprocessing_estimator
    tunning_target = get_tuning_target(model_pipeline_config.model_config)

    # Ensure the model in the pipeline is not fixed and can be tuned
    # pipeline.named_steps['model'] = model_pipeline_config.model_config.model()
    tunning_target = get_tuning_target(model_pipeline_config.model_config)

    # TODO: hack, fix:
    # TODO: BEST PARAMS ARE NOT HARDCODED WHEN TUNING TRANSFORMERS, FIX

    best_params = {
        **best_transformer_params
    }

    pipeline = get_pipeline(
        model_pipeline_config=model_pipeline_config,
        enable_model_hyperparameter_tuning=True,
        # TODO: add fixed model params to start with to each 'ModelPipelineConfig' in project config files
        best_params=best_params
    )

    # Define the parameter distribution for the model hyperparameters
    model_param_distributions = model_pipeline_config.model_config.param_grid

    # Perform Randomized Search to tune model hyperparameters
    random_search = RandomizedSearchCV(pipeline,
                                       param_distributions=model_param_distributions,
                                       n_iter=5,
                                       scoring=tunning_target,
                                       # n_iter=model_pipeline_config.model_config.search_n_iter,
                                       cv=cv, verbose=1,
                                       n_jobs=-1)
    random_search.fit(features, labels)

    print(f"Best Model Parameters: {random_search.best_params_}")
    return random_search


# def _build_tuning_result(model_search: RandomizedSearchCV,
#                          pipeline_search: GridSearchCV,
#                          model_key: str,
#                          features: pd.DataFrame,
#                          labels: pd.Series,
#                          pipeline_config: ModelPipelineConfig,
#                          transformer_tune_results: pd.DataFrame
#                          ) -> TuningResult:
def _build_tuning_result(
        model_search: RandomizedSearchCV,
        pipeline_search: GridSearchCV,
        pipeline_config: ModelPipelineConfig,
        model_key: str,
        transformer_tun_all_civ_results: pd.DataFrame,
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

    if model_type_a.value == ModelType.Regressor.value:
        predictions = model_search.best_estimator_.predict(features)
        mse = mean_squared_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        report_dict = {'mean_squared_error': mse, 'r2_score': r2}
    else:
        report_dict = classification_report(
            labels, predictions, output_dict=True
        )
    training_data = []

    for i in range(0, len(model_search.cv_results_["mean_fit_time"])):
        vals = {
            "number": i,
            "mean_test_score": float(model_search.cv_results_["mean_test_score"][i]),
            "rank_test_score": float(model_search.cv_results_["rank_test_score"][i]),
            "mean_fit_time": float(model_search.cv_results_["mean_fit_time"][i]),
            "params": model_search.cv_results_["params"][i]
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
        pipeline_params={**clean_params(pipeline_search.best_params_)},
        dataset_description=dataset_description,
        all_scores={
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in model_search.cv_results_.items()
        },
        best_score=float(model_search.best_score_),
        result_report=report_dict,
        transformer_tun_all_civ_results=transformer_tun_all_civ_results
    )

    # best_params_info=TuningResultBestParams(
    #     model_key=model_key,
    #     dataset_description=dataset_description,
    #     search_type=tune_type.name,
    #     scoring_function=str(tunning_target),
    #     model_config_reference=str(pipeline_config.model_config),
    #     best_params=clean_params(grid_search.best_params_),
    # ),
    #
    # # model_key=model_key,
    # # dataset_description=dataset_description,
    # # search_type=tune_type.name,
    # # scoring_function=str(tunning_target),
    # best_score=float(grid_search.best_score_),
    # # best_params=clean_params(grid_search.best_params_),
    # all_scores={
    #     k: v.tolist() if isinstance(v, np.ndarray) else v
    #     for k, v in grid_search.cv_results_.items()
    # },
    # result_report=report_dict,
    # transformer_tune_results=transformer_tune_results,
    # )

    return tuning_result


def build_tuning_results(grid_search: RandomizedSearchCV,
                         model_key: str,
                         features: pd.DataFrame,
                         labels: pd.Series,
                         pipeline_config: ModelPipelineConfig,
                         transformer_tune_results: pd.DataFrame
                         ) -> TuningResult:
    print(1)
    pass


def _get_features_labels(df):
    target_cols = [col for col in df.columns if col.startswith('target__')]
    if len(target_cols) > 1:
        raise Exception(f"Multiple 'target__' columns found: {target_cols}")
    elif len(target_cols) == 1:
        labels = df[target_cols[0]]
    else:
        raise Exception("No 'target__' columns found.")

    return df, labels


def run_tunning_for_config(
        model_key: str,
        pipeline_config: ModelPipelineConfig,
        df: pd.DataFrame,
        cv=None,
) -> TuningResult:
    # The target column is prefixed with target__, df can only have one

    features, labels = _get_features_labels(df)

    best_preproc_transformer_sarch, transformer_tun_all_civ_results = _tune_transformer_params(pipeline_config,
                                                                                               features,
                                                                                               labels,
                                                                                               cv=3)
    # TODO: export both this and hyperparams
    best_transformer_params = best_preproc_transformer_sarch.best_params_
    search_results = _tune_model_params(best_transformer_params,
                                        pipeline_config,
                                        features=features,
                                        labels=labels,
                                        cv=3)

    tuning_results = _build_tuning_result(
        model_search=search_results,
        pipeline_search=best_preproc_transformer_sarch,
        pipeline_config=pipeline_config,
        model_key=model_key,
        features=features,
        labels=labels,
        transformer_tun_all_civ_results=transformer_tun_all_civ_results
    )

    return tuning_results
    # Step 1: Preprocessing Parameter Tuning with GridSearchCV
    # fixed_preproc_tune_pipeline = get


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


# def run_tunning_for_config_OLD_2(
#         model_key: str,
#         config: ModelConfig,
#         features: pd.DataFrame,
#         labels: pd.Series,
#         tune_type: TuneType = TuneType.Random,
#         cv=None,
# ) -> TuningResult:
#     model_type_a = config.get_type()
#
#     # if isinstance(config.tunning_func_target, _BaseScorer):
#     #     tunning_target = config.tunning_func_target
#     # elif config.tunning_func_target is not None:
#     #     tunning_target = make_scorer(config.tunning_func_target)
#     # else:
#     #     if model_type_a.value == ModelType.Classifier.value:
#     #         tunning_target = "f1_macro"
#     #     elif model_type_a.value == ModelType.Regressor.value:
#     #         tunning_target = "neg_mean_squared_error"
#     #     else:
#     #         raise Exception("Unknown model type")
#     tunning_target = get_tuning_target(config)
#     cv = _get_cv_for_config(model_type_a=model_type_a, cv=cv)
#     pipeline = get_pipeline(config, enable_hyperparameter_tuning=True)
#
#     if tune_type == TuneType.Grid:
#         grid_search = GridSearchCV(
#             pipeline, config["param_grid"], cv=cv, scoring=tunning_target, n_jobs=-1
#         )
#     elif tune_type == TuneType.Random:
#         if config["search_n_iter"] is None:
#             raise Exception(f"search_n_iter not defined for {config.model_key}")
#         else:
#             print(f"Using {RandomizedSearchCV} with n_iter={config['search_n_iter']}")
#         grid_search = RandomizedSearchCV(
#             pipeline,
#             get_config_param_grid(config),
#             n_iter=config["search_n_iter"],
#             cv=cv,
#             scoring=tunning_target,
#             n_jobs=-1,
#             random_state=42,
#         )
#     else:
#         raise NotImplementedError()
#
#     if model_type_a.value == ModelType.Classifier.value and not (labels.dtype == "int" and labels.isin([0, 1]).all()):
#         raise ValueError(
#             f"Invalid 'labels' series provided for binary classification (must be encoded as 0/1 and not True/False)\n{labels.value_counts()}"
#         )
#     elif model_type_a.value == ModelType.Regressor.value:
#         if not np.issubdtype(labels.dtype, np.number):
#             raise ValueError("Labels for regression should be numeric.")
#
#     grid_search.fit(features, labels)
#
#     predictions = grid_search.best_estimator_.predict(features)
#
#     if model_type_a.value == ModelType.Regressor.value:
#         predictions = grid_search.best_estimator_.predict(features)
#         mse = mean_squared_error(labels, predictions)
#         r2 = r2_score(labels, predictions)
#         report_dict = {'mean_squared_error': mse, 'r2_score': r2}
#     else:
#         report_dict = classification_report(
#             labels, predictions, output_dict=True
#         )
#     training_data = []
#
#     for i in range(0, len(grid_search.cv_results_["mean_fit_time"])):
#         vals = {
#             "number": i,
#             "mean_test_score": float(grid_search.cv_results_["mean_test_score"][i]),
#             "rank_test_score": float(grid_search.cv_results_["rank_test_score"][i]),
#             "mean_fit_time": float(grid_search.cv_results_["mean_fit_time"][i]),
#             "params": grid_search.cv_results_["params"][i]
#         }
#         training_data.append(vals)
#
#     dataset_description = {
#         "num_rows": len(features),
#         "feature_types": {col: features[col].dtype.name for col in features.columns},
#         "trials": training_data,
#         "label_distribution": {},
#     }
#
#     tuning_result = TuningResult(
#         best_params_info=TuningResultBestParams(
#             model_key=model_key,
#             dataset_description=dataset_description,
#             search_type=tune_type.name,
#             scoring_function=str(tunning_target),
#             model_config_reference=str(config),
#             best_params=clean_params(grid_search.best_params_),
#         ),
#
#         # model_key=model_key,
#         # dataset_description=dataset_description,
#         # search_type=tune_type.name,
#         # scoring_function=str(tunning_target),
#         best_score=float(grid_search.best_score_),
#         # best_params=clean_params(grid_search.best_params_),
#         all_scores={
#             k: v.tolist() if isinstance(v, np.ndarray) else v
#             for k, v in grid_search.cv_results_.items()
#         },
#         result_report=report_dict,
#     )
#
#     return tuning_result


# Your tuning function with Optuna
# def run_tuning_with_optuna(
#         model_key: str,
#         config: ModelConfig,
#         features: pd.DataFrame,
#         labels: pd.Series,
#         cv: Optional[int] = None
# ) -> TuningResult:
#     # Determine the scoring function
#     if config.tunning_func_target:
#         scoring = config.tunning_func_target  # make_scorer(config.tunning_func_target)
#     else:
#         scoring = 'accuracy'  # Default scoring function
#
#     # Define the objective function for Optuna
#     def objective(trial: optuna.trial.Trial) -> float:
#         # Dynamically suggest hyperparameters from the config grid
#         trial_params = {key: trial.suggest_categorical(key, values) for key, values in config.param_grid.items()}
#
#         # Incorporate any built-in params from config
#         params = {**config.builtin_params, **trial_params, "enable_categorical": True}
#
#         pipeline = get_pipeline(config, enable_hyperparameter_tuning=False, best_params=trial_params)
#
#         # Perform cross-validation
#         scores = cross_val_score(pipeline, features, labels, scoring=scoring, cv=cv, n_jobs=-1)
#         # all_predictions = cross_val_predict(pipeline, features, labels, cv=cv, method="predict", n_jobs=-1)
#         # cv_metrics_results = calculate_classification_metrics(labels, all_predictions)
#         # return cv_metrics_results["f1_macro"]
#
#         # scores = cross_val_score(model, features, labels, scoring=scoring, cv=cv, n_jobs=-1)
#         return scores.mean()
#
#     # Create and optimize the Optuna study
#     study = optuna.create_study(direction="maximize")
#     # study = optuna.create_study(direction=StudyDirection.MAXIMIZE)
#     study.optimize(objective, n_trials=config.search_n_iter, n_jobs=8)
#
#     # Best trial results
#     best_params = study.best_trial.params
#     best_score = study.best_trial.value
#
#     # Aggregate all scores for reporting
#     all_scores = {f"trial_{trial.number}": trial.value for trial in study.trials}
#
#     training_data = []
#     for trial in study.trials:
#         t: FrozenTrial = trial
#
#         training_data.append({
#             "number": t.number,
#             "value": t.value,
#             # "state": t.state,
#             "params": t.params
#         })
#
#     #
#     # # Generate report
#     # best_model = config.model(**{**config.builtin_params, **best_params})
#     # best_model.fit(features, labels)
#     # predictions = best_model.predict(features)
#     # report = classification_report(labels, predictions, output_dict=True)
#
#     # Prepare best parameters information
#     best_params_info = TuningResultBestParams(
#         model_key=model_key,
#         dataset_description={
#             "trials": training_data,
#             "best_score": best_score,
#             "desc": TuningResult._generate_dataset_description(features, labels)
#         },
#         search_type="Optuna",
#         scoring_function=str(scoring),
#         model_config_reference=str(config),  # Assuming 'config' can be converted to string for reference
#         best_params=best_params
#     )
#
#     # Construct TuningResult
#     return TuningResult(
#         best_score=best_score,
#         all_scores=all_scores,
#         result_report={"TODO": False},
#         # result_report={"classification_report": report},
#         best_params_info=best_params_info
#     )


def calculate_classification_metrics(labels: pd.Series, all_predictions: pd.Series,
                                     all_probabilities: Optional[pd.Series] = None) -> Dict:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": accuracy_score(labels, all_predictions),
        "precision_macro": precision_score(
            labels, all_predictions, average="macro"
        ),
        "recall_macro": recall_score(labels, all_predictions, average="macro"),
        "f1_macro": f1_score(labels, all_predictions, average="macro"),
    }

    if len(labels.unique()) <= 2:
        metrics = {**metrics,
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


def REMOVE_PRE_TUNING_CONFIG_run_pipeline_config(
        config: ModelConfig,
        features: pd.DataFrame,
        labels: pd.Series,
        export_prod: bool = False,
        export_test: bool = False,
        cv: Optional[Any] = None,  # Assume CVType is a type hint for cross-validation strategies
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

    def calculate_regression_metrics(labels: pd.Series, predictions: pd.Series) -> Dict:
        """Calculate regression metrics."""
        metrics = {
            "mse": mean_squared_error(labels, predictions),
            "rmse": np.sqrt(mean_squared_error(labels, predictions)),  # Root Mean Squared Error
            "mae": mean_absolute_error(labels, predictions),  # Mean Absolute Error
            "r2": r2_score(labels, predictions),  # R-squared
            "medae": median_absolute_error(labels, predictions),  # Median Absolute Error
        }

        # MAPE calculation, handling cases where labels could be 0
        mape = np.mean(np.abs((labels - predictions) / labels).replace([np.inf, -np.inf], np.nan).dropna()) * 100
        metrics["mape"] = mape  # Mean Absolute Percentage Error

        return metrics

    model_type = config.get_type()
    res = ModelTrainingResult()
    cv = _get_cv_for_config(model_type_a=config.get_type(), cv=cv)

    if cv is not None:
        cv_pipeline = get_pipeline(config)
        splits = list(cv.split(features, labels))

        all_predictions = cross_val_predict(cv_pipeline, features, labels, cv=splits, method="predict", n_jobs=-1)
        all_predictions = pd.Series(all_predictions, index=labels.index)

        if model_type.value == ModelType.Classifier.value:
            all_probabilities = cross_val_predict(cv_pipeline, features, labels, cv=splits,
                                                  method="predict_proba",
                                                  n_jobs=-1)
            all_probabilities = pd.DataFrame(all_probabilities, index=labels.index)

            if "threshold__threshold" in config.best_params:
                threshold = config.best_params["threshold__threshold"]
                positive_class_probabilities = all_probabilities.iloc[:, 1]
                all_predictions = (positive_class_probabilities >= threshold).astype(int)

            cv_metrics_results = calculate_classification_metrics(labels, all_predictions, all_probabilities)
            res.cm_data = CMResultsData(cv_pipeline, labels, features, all_predictions, all_probabilities,
                                        class_accuracies={})

        elif model_type.value == ModelType.Regressor.value:
            cv_metrics_results = calculate_regression_metrics(labels, all_predictions)
            res.cm_data = CMResultsData(cv_pipeline, labels, features, all_predictions, class_accuracies={},
                                        probabilities=pd.DataFrame())
        else:
            raise Exception("Unknown model type")

        res.cv_metrics = cv_metrics_results
        res.cv_metrics["n_samples"] = len(features)

    # if isinstance(config.model, list):
    #     # Additional steps for ensemble models, if applicable
    #     # This section can remain unchanged but should be adapted if regression models
    #     # can also be part of an ensemble in your use case

    if export_prod:
        prod_pipeline = get_pipeline(config)
        prod_pipeline.fit(features, labels)
        res.prod_model = prod_pipeline
        if hasattr(prod_pipeline.named_steps["model"], "feature_importances_"):
            try:
                # TODO: this is for one hot encoding or other feature transformation which increase the number of columns
                feature_names = features.columns
                categorical_columns = features.select_dtypes(
                    include=["object", "category"]
                ).columns

                # Get OneHotEncoder from the pipeline
                ohe = (
                    prod_pipeline.named_steps["preprocessing"]
                    .named_transformers_["cat"]
                    .named_steps["onehot"]
                )

                ohe_columns = ohe.get_feature_names_out(
                    input_features=categorical_columns
                )

                feature_mapping = {}
                for orig_feature in feature_names:
                    if orig_feature in categorical_columns:
                        feature_mapping[orig_feature] = [
                            col for col in ohe_columns if col.startswith(orig_feature)
                        ]
                    else:
                        feature_mapping[orig_feature] = [orig_feature]

                model_feature_importances = prod_pipeline.named_steps[
                    "model"
                ].feature_importances_

                num_non_ohe_cols = features.select_dtypes(
                    exclude=["object", "category"]
                ).shape[1]

                transformed_feature_indices = {}

                for i, feature in enumerate(features.columns):
                    if feature not in categorical_columns:
                        transformed_feature_indices[feature] = i

                for name, i in enumerate(ohe_columns, start=num_non_ohe_cols):
                    transformed_feature_indices[name] = i

                importances = {}
                for orig_feature, transformed_features in feature_mapping.items():
                    importances[orig_feature] = sum(
                        model_feature_importances[transformed_feature_indices[f]]
                        for f in transformed_features
                    )

                importances_list = [
                    (feature, importance) for feature, importance in importances.items()
                ]
                importances_df = pd.DataFrame(
                    importances_list, columns=["Feature", "Importance"]
                )

                importances_df = importances_df.sort_values(
                    by="Importance", ascending=False
                )

                res.feature_importances = importances_df

            except:
                model_md = prod_pipeline.named_steps["model"]
                feature_names = features.columns
                feature_importances = model_md.feature_importances_
                feature_importances = zip(feature_names, feature_importances)

                feature_importances = pd.DataFrame(
                    feature_importances, columns=["Feature", "Importance"]
                )
                feature_importances = feature_importances.sort_values(
                    by="Importance", ascending=False
                )
                res.feature_importances = feature_importances
        else:
            res.feature_importances = None

    if export_test:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.5
        )
        test_pipeline = get_pipeline(config)
        test_pipeline.fit(X_train, y_train)

        if "preprocessing" in test_pipeline.named_steps:
            preprocessor = test_pipeline.named_steps["preprocessing"]
            X_test_transformed = preprocessor.transform(X_test)
        else:
            X_test_transformed = X_test

        (
            metrics,
            predictions,
            probabilities,
            probabilities_match_id,
        ) = stats_utils.evaluate_classifier_model(test_pipeline, X_test, y_test)

        res.test_data = TestTrainData(
            test_model=test_pipeline,
            x_test_transformed=X_test_transformed,
            x_test=X_test,
            x_train=X_train,
            y_test=y_test,
            predictions=predictions,
            probabilities=probabilities,
            # probabilities_match_id: pd.DataFrame
            metrics=metrics,
            # class_accuracies=class_accuracies,
            class_accuracies=None,
        )

    return res


def run_pipeline_config(
        tuning_result: TuningResult,
        df: pd.DataFrame,
        cv: Optional[Any] = None,  # Assume CVType is a type hint for cross-validation strategies
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

    def calculate_regression_metrics(labels: pd.Series, predictions: pd.Series) -> Dict:
        """Calculate regression metrics."""
        metrics = {
            "mse": mean_squared_error(labels, predictions),
            "rmse": np.sqrt(mean_squared_error(labels, predictions)),  # Root Mean Squared Error
            "mae": mean_absolute_error(labels, predictions),  # Mean Absolute Error
            "r2": r2_score(labels, predictions),  # R-squared
            "medae": median_absolute_error(labels, predictions),  # Median Absolute Error
        }

        # MAPE calculation, handling cases where labels could be 0
        mape = np.mean(np.abs((labels - predictions) / labels).replace([np.inf, -np.inf], np.nan).dropna()) * 100
        metrics["mape"] = mape  # Mean Absolute Percentage Error

        return metrics

    model_pipeline_config = tuning_result.model_pipeline_config
    features, labels = _get_features_labels(df)
    best_params = tuning_result.get_best_params()

    model_type = model_pipeline_config.model_config.get_type()
    res = ModelTrainingResult()
    cv = _get_cv_for_config(model_type_a=model_pipeline_config.model_config.get_type(), cv=cv)

    if cv is not None:
        cv_pipeline = get_pipeline(model_pipeline_config, best_params=best_params)
        splits = list(cv.split(features, labels))

        all_predictions = cross_val_predict(cv_pipeline, features, labels, cv=splits, method="predict", n_jobs=-1)
        all_predictions = pd.Series(all_predictions, index=labels.index)

        if model_type.value == ModelType.Classifier.value:
            all_probabilities = cross_val_predict(cv_pipeline, features, labels, cv=splits,
                                                  method="predict_proba",
                                                  n_jobs=-1)
            all_probabilities = pd.DataFrame(all_probabilities, index=labels.index)

            if "threshold__threshold" in best_params:
                threshold = best_params["threshold__threshold"]
                positive_class_probabilities = all_probabilities.iloc[:, 1]
                all_predictions = (positive_class_probabilities >= threshold).astype(int)

            cv_metrics_results = calculate_classification_metrics(labels, all_predictions, all_probabilities)
            res.cm_data = CMResultsData(cv_pipeline, labels, features, all_predictions, all_probabilities,
                                        class_accuracies={})

        elif model_type.value == ModelType.Regressor.value:
            cv_metrics_results = calculate_regression_metrics(labels, all_predictions)
            res.cm_data = CMResultsData(cv_pipeline, labels, features, all_predictions, class_accuracies={},
                                        probabilities=pd.DataFrame())
        else:
            raise Exception("Unknown model type")

        res.cv_metrics = cv_metrics_results
        res.cv_metrics["n_samples"] = len(features)

    # if isinstance(config.model, list):
    #     # Additional steps for ensemble models, if applicable
    #     # This section can remain unchanged but should be adapted if regression models
    #     # can also be part of an ensemble in your use case

    # if export_prod:
    if False:
        prod_pipeline = get_pipeline(model_pipeline_config)
        prod_pipeline.fit(features, labels)
        res.prod_model = prod_pipeline
        if hasattr(prod_pipeline.named_steps["model"], "feature_importances_"):
            try:
                # TODO: this is for one hot encoding or other feature transformation which increase the number of columns
                feature_names = features.columns
                categorical_columns = features.select_dtypes(
                    include=["object", "category"]
                ).columns

                # Get OneHotEncoder from the pipeline
                ohe = (
                    prod_pipeline.named_steps["preprocessing"]
                    .named_transformers_["cat"]
                    .named_steps["onehot"]
                )

                ohe_columns = ohe.get_feature_names_out(
                    input_features=categorical_columns
                )

                feature_mapping = {}
                for orig_feature in feature_names:
                    if orig_feature in categorical_columns:
                        feature_mapping[orig_feature] = [
                            col for col in ohe_columns if col.startswith(orig_feature)
                        ]
                    else:
                        feature_mapping[orig_feature] = [orig_feature]

                model_feature_importances = prod_pipeline.named_steps[
                    "model"
                ].feature_importances_

                num_non_ohe_cols = features.select_dtypes(
                    exclude=["object", "category"]
                ).shape[1]

                transformed_feature_indices = {}

                for i, feature in enumerate(features.columns):
                    if feature not in categorical_columns:
                        transformed_feature_indices[feature] = i

                for name, i in enumerate(ohe_columns, start=num_non_ohe_cols):
                    transformed_feature_indices[name] = i

                importances = {}
                for orig_feature, transformed_features in feature_mapping.items():
                    importances[orig_feature] = sum(
                        model_feature_importances[transformed_feature_indices[f]]
                        for f in transformed_features
                    )

                importances_list = [
                    (feature, importance) for feature, importance in importances.items()
                ]
                importances_df = pd.DataFrame(
                    importances_list, columns=["Feature", "Importance"]
                )

                importances_df = importances_df.sort_values(
                    by="Importance", ascending=False
                )

                res.feature_importances = importances_df

            except:
                model_md = prod_pipeline.named_steps["model"]
                feature_names = features.columns
                feature_importances = model_md.feature_importances_
                feature_importances = zip(feature_names, feature_importances)

                feature_importances = pd.DataFrame(
                    feature_importances, columns=["Feature", "Importance"]
                )
                feature_importances = feature_importances.sort_values(
                    by="Importance", ascending=False
                )
                res.feature_importances = feature_importances
        else:
            res.feature_importances = None

    # if True:  # export_test:
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.5
    )
    test_pipeline = get_pipeline(model_pipeline_config, best_params=best_params)
    test_pipeline.fit(X_train, y_train)

    if "preprocessing" in test_pipeline.named_steps:
        preprocessor = test_pipeline.named_steps["preprocessing"]
        X_test_transformed = preprocessor.transform(X_test)
    else:
        X_test_transformed = X_test

    evaluation_method = stats_utils.evaluate_classifier_model if model_type == ModelType.Classifier else stats_utils.evaluate_regressor_model

    (
        metrics,
        predictions,
        probabilities,
        probabilities_match_id,
    ) = evaluation_method(test_pipeline, X_test, y_test)
    # ) = stats_utils.evaluate_classifier_model(test_pipeline, X_test, y_test)

    res.test_data = TestTrainData(
        test_model=test_pipeline,
        x_test_transformed=X_test_transformed,
        x_test=X_test,
        x_train=X_train,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        # probabilities_match_id: pd.DataFrame
        metrics=metrics,
        # class_accuracies=class_accuracies,
        class_accuracies=None,
    )

    return res


def run_pipeline_config_OLD(
        config: ModelConfig,
        features: pd.DataFrame,
        labels: pd.Series,
        export_prod=False,
        export_test=False,
        cv=None,
        VERBOSE=True,
) -> ModelTrainingResult:
    res = ModelTrainingResult()

    cv = _get_cv_for_config(cv)

    if cv is not None:
        cv_pipeline = get_pipeline(config)
        splits = list(cv.split(features, labels))

        all_predictions = cross_val_predict(
            cv_pipeline, features, labels, cv=splits, method="predict", n_jobs=-1
        )

        all_probabilities = cross_val_predict(
            cv_pipeline, features, labels, cv=splits, method="predict_proba", n_jobs=-1
        )

        all_probabilities = pd.DataFrame(all_probabilities, index=labels.index)

        if "threshold__threshold" in config.best_params:
            threshold = config.best_params["threshold__threshold"]
            positive_class_probabilities = all_probabilities.iloc[:, 1]  # Get probabilities for the positive class
            all_predictions = (positive_class_probabilities >= threshold).astype(int)  # Apply threshold
            all_predictions = pd.Series(all_predictions, index=labels.index)  # Convert to pd.Series
        else:
            all_predictions = pd.Series(all_predictions, index=labels.index)

        # Compute metrics manually using the predictions
        cv_metrics_results = {
            "accuracy": accuracy_score(labels, all_predictions),
            "precision_macro": precision_score(
                labels, all_predictions, average="macro"
            ),
            "recall_macro": recall_score(labels, all_predictions, average="macro"),
            "f1_macro": f1_score(labels, all_predictions, average="macro"),
            "target_f1": f1_score(labels, all_predictions, pos_label=1),
            "target_recall": recall_score(labels, all_predictions, pos_label=1),
            "target_precision": precision_score(labels, all_predictions, pos_label=1),
            "fbeta_1.5": fbeta_score(labels, all_predictions, pos_label=1, beta=1.5),
            "fbeta_2.5": fbeta_score(labels, all_predictions, pos_label=1, beta=2.5),
            "fbeta_4.0": fbeta_score(labels, all_predictions, pos_label=1, beta=4.0),
        }

        # Storing results in res.cv_metrics and res.cm_data
        res.cv_metrics = cv_metrics_results
        res.cv_metrics["n_samples"] = len(features)

        res.cm_data = CMResultsData(
            test_model=cv_pipeline,
            y_test=pd.Series(labels),
            x_test=features,
            predictions=pd.Series(all_predictions),
            probabilities=pd.DataFrame(all_probabilities),
            class_accuracies={},
        )

    # If using voting classifier we need to rerun with all of the models to get the classifier probs
    if isinstance(config.model, list):
        pipeline = get_pipeline(config)
        pipeline.fit(features, labels)

        preprocessor = pipeline.named_steps["preprocessing"]

        features_transformed = preprocessor.transform(features)

        model_params = __get_model_params(config, False)
        eclf = __get_ensemble_classifier_factory(config, model_params)

        ensemble_probas = [
            c().fit(features_transformed, labels).predict_proba(features_transformed)
            for c in [*config.model, eclf]
        ]

        res.ensemble_probas = ensemble_probas

    # TODO: disabled for now, need to fix:
    # TODO: - Cleanup
    # TODO: - integrate with CV code, esp metrics
    # TODO: - use a separate test sample passed to pipeline

    # if export_prod:
    #     prod_pipeline = get_pipeline(config)
    #     prod_pipeline.fit(features, labels)
    #     res.prod_model = prod_pipeline
    #     if hasattr(prod_pipeline.named_steps["model"], "feature_importances_"):
    #         try:
    #             # TODO: this is for one hot encoding or other feature transformation which increase the number of columns
    #             feature_names = features.columns
    #             categorical_columns = features.select_dtypes(
    #                 include=["object", "category"]
    #             ).columns
    #
    #             # Get OneHotEncoder from the pipeline
    #             ohe = (
    #                 prod_pipeline.named_steps["preprocessing"]
    #                 .named_transformers_["cat"]
    #                 .named_steps["onehot"]
    #             )
    #
    #             ohe_columns = ohe.get_feature_names_out(
    #                 input_features=categorical_columns
    #             )
    #
    #             feature_mapping = {}
    #             for orig_feature in feature_names:
    #                 if orig_feature in categorical_columns:
    #                     feature_mapping[orig_feature] = [
    #                         col for col in ohe_columns if col.startswith(orig_feature)
    #                     ]
    #                 else:
    #                     feature_mapping[orig_feature] = [orig_feature]
    #
    #             model_feature_importances = prod_pipeline.named_steps[
    #                 "model"
    #             ].feature_importances_
    #
    #             num_non_ohe_cols = features.select_dtypes(
    #                 exclude=["object", "category"]
    #             ).shape[1]
    #
    #             transformed_feature_indices = {}
    #
    #             for i, feature in enumerate(features.columns):
    #                 if feature not in categorical_columns:
    #                     transformed_feature_indices[feature] = i
    #
    #             for name, i in enumerate(ohe_columns, start=num_non_ohe_cols):
    #                 transformed_feature_indices[name] = i
    #
    #             importances = {}
    #             for orig_feature, transformed_features in feature_mapping.items():
    #                 importances[orig_feature] = sum(
    #                     model_feature_importances[transformed_feature_indices[f]]
    #                     for f in transformed_features
    #                 )
    #
    #             importances_list = [
    #                 (feature, importance) for feature, importance in importances.items()
    #             ]
    #             importances_df = pd.DataFrame(
    #                 importances_list, columns=["Feature", "Importance"]
    #             )
    #
    #             importances_df = importances_df.sort_values(
    #                 by="Importance", ascending=False
    #             )
    #
    #             res.feature_importances = importances_df
    #
    #         except:
    #             model_md = prod_pipeline.named_steps["model"]
    #             feature_names = features.columns
    #             feature_importances = model_md.feature_importances_
    #             feature_importances = zip(feature_names, feature_importances)
    #
    #             feature_importances = pd.DataFrame(
    #                 feature_importances, columns=["Feature", "Importance"]
    #             )
    #             feature_importances = feature_importances.sort_values(
    #                 by="Importance", ascending=False
    #             )
    #             res.feature_importances = feature_importances
    #     else:
    #         res.feature_importances = None
    #
    # if export_test:
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         features, labels, test_size=0.5
    #     )
    #     test_pipeline = get_pipeline(config)
    #     test_pipeline.fit(X_train, y_train)
    #
    #     if "preprocessing" in test_pipeline.named_steps:
    #         preprocessor = test_pipeline.named_steps["preprocessing"]
    #         X_test_transformed = preprocessor.transform(X_test)
    #     else:
    #         X_test_transformed = X_test
    #
    #     (
    #         metrics,
    #         predictions,
    #         probabilities,
    #         probabilities_match_id,
    #     ) = stats_utils.evaluate_model(test_pipeline, X_test, y_test)
    #
    #     res.test_data = TestTrainData(
    #         test_model=test_pipeline,
    #         x_test_transformed=X_test_transformed,
    #         x_test=X_test,
    #         x_train=X_train,
    #         y_test=y_test,
    #         predictions=predictions,
    #         probabilities=probabilities,
    #         # probabilities_match_id: pd.DataFrame
    #         metrics=metrics,
    #         # class_accuracies=class_accuracies,
    #         class_accuracies=None,
    #     )
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

    # if "f1_macro" in results_table:
    #     metrics_df = pd.DataFrame.from_dict(results_table, orient="index").sort_values(
    #         by=["f1_macro"], ascending=False
    #     )
    # else:
    #     metrics_df = pd.DataFrame.from_dict(results_table, orient="index").sort_values(
    #         by=["mse"], ascending=False
    #     )
    return metrics_df


def extract_feature_names(pipeline, input_data):
    """
    Passes input_data through all transformers in the pipeline to extract feature names.
    Does not require 'get_feature_names_out' as long as transformers operate on pandas dfs instead of ndarrays

    :param pipeline: A scikit-learn Pipeline object.
    :param input_data: Dummy input data as a pandas DataFrame.
    :return: List of output feature names after all transformations.
    """
    transformed_data = input_data
    for name, transformer in pipeline.steps[:-1]:  # Exclude the last step if it's a model
        # print(name)
        transformed_data = transformer.transform(transformed_data)

        # If the transformer reduces or modifies the feature space, adapt accordingly
        if hasattr(transformer, 'get_feature_names_out'):
            # For transformers that support it, directly obtain the feature names
            feature_names = transformer.get_feature_names_out()
        else:
            # Otherwise, infer feature names (if possible, depending on the output)
            if isinstance(transformed_data, pd.DataFrame):
                feature_names = transformed_data.columns.tolist()
            else:
                # If the output is a NumPy array, generate placeholder names
                feature_names = [f'feature_{i}' for i in range(transformed_data.shape[1])]
    return feature_names
