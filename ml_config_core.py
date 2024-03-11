import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List, Dict, Optional, Any, Callable, Type, Tuple

import numpy as np
import pandas as pd
import sklearn.pipeline
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    fbeta_score,
    f1_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    log_loss
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import shared.ml_config_preproc as ml_config_preproc
from dill import dump, load

THRESHOLDS = np.linspace(0.1, 0.9, 9)

EXPORT_MODEL_DIR = ".production_models"


# Define a custom scorer function that negates the log_loss since make_scorer assumes higher score is better
def neg_log_loss(y_true, y_pred):
    # log_loss expects probabilities, ensure y_pred is in the correct form if not already
    return -log_loss(y_true, y_pred)


# Update your tunning_func_target to use the neg_log_loss
tunning_func_target: Optional[Callable[[np.ndarray, np.ndarray], float]] = field(
    default_factory=lambda: make_scorer(neg_log_loss, needs_proba=True))


class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """

    TODO: currently only single option (defined in cls.Options enum) is supported. We need to allow a dynamic number of possible options
    """
    # prefix = "base"  # Default prefix, to be overridden by child classes
    transformer_name: Union[str, None] = None  # To be overridden by child classes

    @property
    def option(self):
        return self.Options(self._option)

    @option.setter
    def option(self, value):
        self._option = value

    @classmethod
    def select_params(cls, params):
        # Extract parameters relevant to the specific transformer based on prefix
        if cls.transformer_name is None:
            raise NotImplementedError()
        transformer_params = {}
        # {k.split(f"feat_trans_{cls.transformer_name}__")[1]: v for k, v in params.items() if
        #                       k.startswith(f"feat_trans_{cls.transformer_name}__")}
        for k, v in params.items():
            if k.startswith(f"feat_trans_{cls.transformer_name}__"):
                transformer_params[k.split(f"feat_trans_{cls.transformer_name}__")[1]] = v

        return transformer_params

    @classmethod
    def get_step_name(cls):
        if cls.transformer_name is None:
            raise NotImplementedError()

        step_name = "feat_trans_" + cls.transformer_name
        return step_name

    @classmethod
    def get_option_grid(cls):
        step_name = cls.get_step_name()
        prefix = f"{step_name}__"

        possible_options = cls.Options

        return {f"{prefix}option": [v.value for v in possible_options]}

    @classmethod
    def create(cls, all_params):
        step_name = cls.get_step_name()
        prefix = f"{step_name}__"

        relevant_params = {k.split(prefix)[1]: v for k, v in all_params.items() if k.startswith(prefix)}
        # Instantiate the transformer with the filtered parameters
        return (step_name, cls(**relevant_params))

    def fit(self, X, y=None):
        # Fit method doesn't need to do anything
        return self

    @abstractmethod
    def transform(self, X):
        # Abstract method to be implemented by each transformer
        pass

    def set_params(self, **params):
        for param, value in params.items():
            if param == 'option' and isinstance(value, int):
                value = self.Options(value)
            setattr(self, param, value)
        return self

    @abstractmethod
    def get_target_col_names(self):
        # TODO: instead of having to implement this for each do it dynamically by finding new columns that were not present in source df
        raise NotImplementedError()


class DummyTestTransformer(BaseTransformer):
    """
    Just drops all the features which have any predictive power and replace them with a random data
    """
    transformer_name = "dummytransformer"

    class Options(Enum):
        OFF = 0
        ON = 1

    def __init__(self, option=Options.ON):
        self.option = option

    def transform(self, df):
        if self.option == self.Options.ON:
            df['dummy_data'] = np.random.rand(len(df))
            df = df[['dummy_data']].copy()

        return df

    def get_target_col_names(self):
        return ['dummy_data']


def fbeta_threshold_scorer(y_true, y_proba, beta=0.5, threshold=0.5, pos_label=1):
    # Check if y_proba is 1-dimensional
    if y_proba.ndim == 1:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = (y_proba[:, pos_label] >= threshold).astype(int)

    return fbeta_score(y_true, y_pred, beta=beta, pos_label=pos_label)


@dataclass
class BalancingConfig(ABC):
    params: dict = field(default_factory=dict)

    def __init__(self):
        pass

    def get_params(self):
        return {**self.params, "random_state": 42}

    @abstractmethod
    def get_pipeline(self):
        return


@dataclass
class SmoteConfig(BalancingConfig):
    def get_pipeline(self):
        return ("smote", SMOTE(**self.get_params()))


@dataclass
class UnderSamplingConfig(BalancingConfig):
    def get_pipeline(self):
        return ("undersample", RandomUnderSampler(**self.get_params()))


@dataclass
class AdasynConfig(BalancingConfig):
    def get_pipeline(self):
        return ("adasyn", ADASYN(**self.get_params()))


@dataclass
class BorderlineSmoteConfig(BalancingConfig):
    def get_pipeline(self):
        return ("borderline_smote", BorderlineSMOTE(**self.get_params()))


@dataclass
class RandomOverSamplingConfig(BalancingConfig):
    def get_pipeline(self):
        return ("random_oversampling", RandomOverSampler(**self.get_params()))


class ModelType(Enum):
    Classifier = 1
    Regressor = 2


@dataclass
class ModelConfig:
    # TODO: use callable which returns BaseEstimator instead of instance
    model: Union[Callable[[], BaseEstimator], List[BaseEstimator]]
    supports_nan: bool
    param_grid: Dict[str, List[Any]]
    builtin_params: Dict[str, Any]

    search_n_iter: int
    balancing_config: Optional[BalancingConfig] = None
    preprocessing: Optional[Callable] = None
    tunning_func_target: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    best_params: Optional[Dict[str, Any]] = None

    ensemble_classifier: Optional[Any] = None  # TODO

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def model_key(self):
        return self.__class__.__name__

    def get_type(self):
        if issubclass(self.model, RegressorMixin):
            return ModelType.Regressor

        return ModelType.Classifier

    def optuna_objective(trial, X, y, cv):
        raise NotImplementedError()


@dataclass
class Ensemble_Log_KNN_SVM_SMOTE(ModelConfig):
    # model: Union[BaseEstimator, List[BaseEstimator]] = LogisticRegression
    model: Union[BaseEstimator, List[BaseEstimator]] = field(
        default_factory=lambda: [
            LogisticRegression,
            KNeighborsClassifier,
            lambda **kwargs: SVC(probability=True, **kwargs),
        ]
    )

    supports_nan: bool = False
    search_n_iter: int = field(default=150)
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {})
    preprocessing: Optional[Callable] = ml_config_preproc.preprocessing_for_logreg()
    balancing_config: Optional[Callable] = SmoteConfig()
    builtin_params: Dict[str, Any] = field(default_factory=lambda: {})
    best_params: Dict[str, Any] = field(default_factory=lambda: {})
    ensemble_classifier: Optional[Any] = lambda **kwargs: VotingClassifier(
        voting="soft", **kwargs
    )


@dataclass
class SVC_SMOTE(ModelConfig):
    # model: Union[BaseEstimator, List[BaseEstimator]] = LogisticRegression
    model: Union[BaseEstimator, List[BaseEstimator]] = lambda **kwargs: SVC(probability=True, **kwargs)

    supports_nan: bool = False
    search_n_iter: int = field(default=150)
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {})
    preprocessing: Optional[Callable] = ml_config_preproc.preprocessing_for_logreg()
    balancing_config: Optional[Callable] = SmoteConfig()
    builtin_params: Dict[str, Any] = field(default_factory=lambda: {})
    best_params: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class XGBoostBaseConfig(ModelConfig):
    search_n_iter: int = field(default=50)
    model: Union[BaseEstimator, List[BaseEstimator]] = XGBClassifier
    supports_nan: bool = True
    balancing_config: Optional[BalancingConfig] = None

    default_params: dict = field(
        default_factory=lambda: {
            "model__gamma": 0,
            "model__learning_rate": 0.4,
            "model__max_depth": 7,
            "model__min_child_weight": 3,
            "model__n_estimators": 250,
            "model__scale_pos_weight": 5,
        })

    preprocessing: Optional[Callable] = ml_config_preproc.preprocessing_for_xgboost(
        use_categorical_feature=True
    )

    param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            # "feature_selection_k_best__k": [5, 10, 15, 20, 25, 30, "all"],

            "model__learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1],
            "model__max_depth": [4, 5, 6, 7, 10, 12, None],
            "model__n_estimators": [50, 100, 150, 200, 250],
            "model__min_child_weight": [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3],
            "model__gamma": [0, 0.05, 0.1, 0.3, 0.4],
            "model__scale_pos_weight": [1, 2, 3, 5, 10],
        }
    )
    builtin_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_categorical": True,
            "tree_method": "hist",
            "device": "cuda"
        }
    )


@dataclass
class XGBoostRegressorBaseConfig(ModelConfig):
    search_n_iter: int = field(default=3)
    model: Union[BaseEstimator, List[BaseEstimator]] = XGBRegressor
    supports_nan: bool = True
    balancing_config: Optional[BalancingConfig] = None

    preprocessing: Optional[Callable] = ml_config_preproc.preprocessing_for_xgboost(
        use_categorical_feature=True
    )

    param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "model__learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1],
            "model__max_depth": [4, 5, 6, 7, 10, 12, None],
            "model__n_estimators": [50, 100, 150, 200, 250],
            "model__gamma": [0, 0.05, 0.1, 0.3, 0.4],
        }
    )
    builtin_params: Dict[str, Any] = field(
        default_factory=lambda: {"enable_categorical": True, "objective": "reg:squarederror"}
    )


# @dataclass
# class XGBoostRegressionDefault(XGBoostRegBaseConfig):
#     search_n_iter: int = field(default=100)


@dataclass
class CatBoostBaseConfig(ModelConfig):
    search_n_iter: int = field(default=80)
    balancing_config: Optional[BalancingConfig] = None
    model: Union[BaseEstimator, List[BaseEstimator]] = CatBoostClassifier

    supports_nan: bool = True
    preprocessing: Optional[Callable] = ml_config_preproc.preprocessing_for_xgboost(use_categorical_feature=True)

    default_params: dict = field(
        default_factory=lambda: {
            "model__scale_pos_weight": None,
            "model__depth": None,
            "model__learning_rate": None,
            "model__iterations": None,
            "model__l2_leaf_reg": None,
        })

    builtin_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "logging_level": "Silent",
            "leaf_estimation_iterations": 1,
            "boosting_type": "Plain",
            "thread_count": -1
            # "task_type": "GPU"
            # 'task_type': "GPU",
            # 'gpu_ram_part':0.1,
            # "leaf_estimation_method": "Exact",
            # 'objective': 'RMSE'
        }
    )

    param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "model__scale_pos_weight": [5, 10, 15, 19, 25],
            "model__depth": [6, 7, 8, 9, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3],
            "model__iterations": [100, 250, 500],
            "model__l2_leaf_reg": [2, 3, 5],
            # "model__border_count": [32, 64],
            # "model__class_weights": [{0: 1, 1: 19}],
            # "model__auto_class_weights": ['Balanced'],
            # "model__custom_loss": [['AUC', 'F1']],
            # "model__eval_metric": ['F1', 'AUC'],
            # "model__bagging_temperature": [0, 1],
            # "model__bootstrap_type": ['Bayesian', 'Bernoulli']
        }
    )
    # param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
    #     # "model__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
    #     # "model__depth": [4, 6, 8, 10, None],
    #     # "model__grow_policy": [None, "SymmetricTree"],#, "Lossguide", "Depthwise"],
    #     # "model__iterations": [25, 50, 75, 100, 150, 250, 400, 500],
    #     "model__scale_pos_weight": [2, 5, 7, 10, 12, 14, 15, 17, 18, 19, 20, 22, 25, 27, 30, 33, 35, 40, 45, 50],
    #     # "model__l2_leaf_reg": [None, 2, 3, 5, 7, 10, ],
    #     # "model__border_count": [None, 32, 64, 128, 255],
    #     # "model__bootstrap_type": [None, 'Bayesian', 'Bernoulli', 'MVS'],
    #     # "model__leaf_estimation_method": [None, 'Newton', 'Gradient'],
    #     # Approximate ratio of majority to minority class
    # })

    # cu_rf = pipeline.FeatureSetConfig(
    #     feature_set=None,
    #     synthetic_funcs=[],
    #     # model=cuRF,
    #     model=RandomForestClassifier,
    #     supports_nan=True,
    #     preprocessing=shared.ml_config_preproc.ml_config_preproc.preprocessing_for_xgboost(),  # Use the experimental feature
    #     best_params={},  # Specify SVM hyperparameters here if any
    #     param_grid={
    #         # Define grid for hyperparameter tuning if required
    #         # Example: "svm__C": [0.1, 1, 10, 100], "svm__kernel": ["linear", "rbf", "poly"]
    #     },
    # )


# Custom scorer function
def threshold_scorer(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    return f1_score(y_true, y_pred)


@dataclass
class RandomForestBaseConfig(ModelConfig):
    search_n_iter: int = field(default=100)
    model: Union[BaseEstimator, List[BaseEstimator]] = RandomForestClassifier

    supports_nan: bool = True
    preprocessing: Optional[Callable] = ml_config_preproc.preprocessing_for_xgboost()

    builtin_params: Dict[str, Any] = field(default_factory=lambda: {})

    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {})


#         lightgbm_config = pipeline.FeatureSetConfig(
#         feature_set=None,
#         synthetic_funcs=[],
#         model=LGBMClassifier,
#         supports_nan=True,
#         preprocessing=shared.ml_config_preproc.ml_config_preproc.preprocessing_for_xgboost(),
#         # Define appropriate preprocessing for CatBoost
#         # preprocessing=pipeline.preprocessing_for_lightgbm(),  # Define appropriate preprocessing for LightGBM
#         builtin_params={'verbosity': -1},  # Example of a built-in parameter
#         best_params={'model__learning_rate': 0.1, 'model__max_depth': -1, 'model__n_estimators': 200,
#                      'model__num_leaves': 124},
#         param_grid={
#             "model__learning_rate": [0.01, 0.05, 0.1],
#             "model__max_depth": [4, 6, 8, -1],
#             "model__n_estimators": [50, 100, 200],
#             "model__num_leaves": [31, 62, 124],
#             # Add other LightGBM-specific parameters here
#         },
#     )


@dataclass
class LGBMBaseConfig(ModelConfig):
    search_n_iter: int = field(default=20)
    model: Union[BaseEstimator, List[BaseEstimator]] = LGBMClassifier

    supports_nan: bool = True
    preprocessing: Optional[Callable] = ml_config_preproc.preprocessing_for_xgboost()

    builtin_params: Dict[str, Any] = field(default_factory=lambda: {"verbose": -1})

    param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [6, 8, -1],
            "model__n_estimators": [100],
            "model__num_leaves": [31, 62],
            "model__scale_pos_weight": [5, 10, 20, 30],
        }
    )


@dataclass
class LGBMForestBaseConfigTuneF1(LGBMBaseConfig):
    tunning_func_target = (make_scorer(f1_score, pos_label=1),)


@dataclass
class LGBMForestBaseConfigTuneFBeta_25(LGBMBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=2.5, pos_label=1)


# @dataclass CatBoostBaseConfigTuneF1 CatBoostBaseConfigTuneFBeta_15 XGBoostTuneF1 CatBoostBaseConfigTuneFBeta_40 CatBoostBaseConfigTuneFBeta_20 CatBoostBaseConfigTuneFBeta_25 XGBoostTuneCatFBeta_25 XGBoostTuneCatF1FBeta_175
@dataclass
class CatBoostBaseConfigTuneF1(CatBoostBaseConfig):
    tunning_func_target = (make_scorer(f1_score, pos_label=1),)


# preprocessing = pipeline.ml_config_preproc.preprocessing_for_xgboost(use_categorical_feature=True),  # Use the experimental feature
@dataclass
class CatBoostBaseConfigTuneFBeta_15(CatBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=1.5, pos_label=1)


@dataclass
class CatBoostBaseConfigTuneFBeta_20(CatBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=2.0, pos_label=1)


@dataclass
class CatBoostBaseConfigTuneFBeta_25(CatBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(
        fbeta_score, beta=2.5, pos_label=1
    )  # preprocessing = pipeline.ml_config_preproc.preprocessing_for_xgboost(use_categorical_feature=True),  # Use the experimental feature


@dataclass
class CatBoostBaseConfigTuneFBeta_40(CatBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(
        fbeta_score, beta=4, pos_label=1
    )  # preprocessing = pipeline.ml_config_preproc.preprocessing_for_xgboost(use_categorical_feature=True),  # Use the experimental feature


class CatBoostBaseConfigTuneFBeta_50(CatBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(
        fbeta_score, beta=5, pos_label=1
    )  # preprocessing = pipeline.ml_config_preproc.preprocessing_for_xgboost(use_categorical_feature=True),  # Use the experimental feature


@dataclass
class CatBoostBaseConfigTuneFBeta_325(CatBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(
        fbeta_score, beta=3.25, pos_label=1
    )  # preprocessing = pipeline.ml_config_preproc.preprocessing_for_xgboost(use_categorical_feature=True),  # Use the experimental feature


@dataclass
class CatBoostBaseConfigTuneRecall(CatBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(recall_score, pos_label=1)


@dataclass
class XGBoostTuneRecall(XGBoostBaseConfig):
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(recall_score, pos_label=1)


@dataclass
class XGBoostTuneF1(XGBoostBaseConfig):
    search_n_iter: int = field(default=50)
    tunning_func_target: Optional[Callable[[np.ndarray, np.ndarray], float]] = field(
        default_factory=lambda: make_scorer(f1_score, pos_label=1))


@dataclass
class XGBoostTuneLogLoss(XGBoostBaseConfig):
    search_n_iter: int = field(default=100)
    tunning_func_target: Optional[Callable[[np.ndarray, np.ndarray], float]] = field(
        default_factory=lambda: make_scorer(neg_log_loss, needs_proba=True))


@dataclass
class XGBoostOrdinalRegressor(XGBoostRegressorBaseConfig):
    search_n_iter: int = field(default=50)
    # tunning_func_target: Optional[Callable[[np.ndarray, np.ndarray], float]] = field(
    #     default_factory=lambda: make_scorer(neg_log_loss, needs_proba=True))


@dataclass
class XGBoostTunePRAUC(XGBoostBaseConfig):
    search_n_iter: int = field(default=50)
    tunning_func_target: Optional[Callable[[np.ndarray, np.ndarray], float]] = field(
        default_factory=lambda: make_scorer(average_precision_score, needs_proba=True, pos_label=1))


@dataclass
class XGBoostTuneCatFBeta_25(XGBoostBaseConfig):
    search_n_iter: int = field(default=250)
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=2.5, pos_label=1)


class XGBoostTuneCatFBeta_325(XGBoostBaseConfig):
    search_n_iter: int = field(default=250)
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=3.25, pos_label=1)


class XGBoostTuneCatFBeta_40(XGBoostBaseConfig):
    search_n_iter: int = field(default=250)
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=4.0, pos_label=1)


class XGBoostTuneCatFBeta_50(XGBoostBaseConfig):
    search_n_iter: int = field(default=250)
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=5.0, pos_label=1)


# TODO: reame to precision recall
@dataclass
class XGBoostTuneCatFBeta_25_TuneThreshold(XGBoostBaseConfig):
    search_n_iter: int = field(default=250)
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(average_precision_score, needs_proba=True)

    param_grid: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "model__learning_rate": [0.1, 0.4, 0.7, 0.9, 1],
            "model__max_depth": [4, 5, 6, 7, None],
            "model__n_estimators": [50, 100, 150, 200],
            "model__min_child_weight": [1, 3, 5, 7],
            "model__gamma": [0, 0.1, 0.3, 0.5],
            "model__scale_pos_weight": [1, 5, 10, 20],
        }
    )


@dataclass
class XGBoostTuneCatF1FBeta_175(XGBoostBaseConfig):
    search_n_iter: int = field(default=250)
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(fbeta_score, beta=175, pos_label=1)


@dataclass
class XGBoostCatF1(XGBoostBaseConfig):
    search_n_iter: int = field(default=25)
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(f1_score, pos_label=1)


@dataclass
class XGBoostCatF1UndersampleAuto(XGBoostBaseConfig):
    search_n_iter: int = field(default=250)
    balancing_config: Optional[Callable] = UnderSamplingConfig()
    tunning_func_target: Optional[
        Callable[[np.ndarray, np.ndarray], float]
    ] = make_scorer(f1_score, pos_label=1)


# @dataclass
# class FeatureSetConfig:
#     feature_set: Any
#     synthetic_funcs: List[Callable]
#
#     # TODO: use callable which returns BaseEstimator instead of instance
#     model: Union[BaseEstimator, List[BaseEstimator]]
#     supports_nan: bool
#     best_params: Dict[str, Any]
#     param_grid: Dict[str, List[Any]]
#     balancing_config: Optional[BalancingConfig] = None
#     preprocessing: Optional[Callable] = None
#     builtin_params: Dict[str, Any] = field(default_factory=dict)
#     tunning_func_target: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
#
#     ensemble_classifier: Optional[Any] = None  # TODO
#
#     def __getitem__(self, key):
#         return getattr(self, key)


@dataclass
class TestTrainData:
    test_model: sklearn.pipeline.Pipeline

    x_test_transformed: np.array
    y_test: pd.Series
    x_test: pd.DataFrame
    x_train: pd.DataFrame
    predictions: pd.DataFrame
    probabilities: pd.DataFrame

    metrics: dict
    class_accuracies: dict
    feature_importances: Optional[pd.DataFrame] = None


@dataclass
class CMResultsData:
    test_model: sklearn.pipeline.Pipeline
    y_test: pd.Series
    x_test: pd.DataFrame

    predictions: pd.Series
    probabilities: pd.DataFrame

    class_accuracies: dict


@dataclass
class PipelineTransformerConfig:
    transformers: List[Type[BaseTransformer]] = field(default_factory=list)

    def get_feature_search_grid(self) -> Dict[str, List]:
        feature_search_grid = []

        for cl in self.transformers:
            feature_search_grid.append(cl.get_option_grid())

        search_grid = {k: v for d in feature_search_grid for k, v in d.items()}
        return search_grid

    def __str__(self):
        search_grid = self.get_feature_search_grid()
        grid_iters = {key: len(value) for key, value in search_grid.items()}
        grid_iters = sum(grid_iters.values())
        return f"transformers: {len(self.transformers)}\n total options: {grid_iters}\n{self.transformers}\nsearch_grid:\n{search_grid}"


@dataclass
class ModelPipelineConfig:
    model_config: ModelConfig
    transformer_config: Optional[PipelineTransformerConfig] = None

    def to_yaml(self):
        pass


@dataclass
class ModelTrainingResult:
    # model: BaseModel
    cv_metrics: Optional[dict] = None
    prod_model: Optional[sklearn.pipeline.Pipeline] = None
    test_data: Optional[TestTrainData] = None
    feature_importances: Optional[pd.DataFrame] = None
    cm_data: Optional[CMResultsData] = None
    ensemble_probas: Optional[Any] = None

    @staticmethod
    def serialize_model(res: "ModelTrainingResult", model_key: str, target_folder=EXPORT_MODEL_DIR):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        target_path = f'{target_folder}/{model_key}.dill'
        with open(target_path, "wb") as targt_file:
            dump(res, targt_file)

    @staticmethod
    def load_serialize_model(model_key, target_folder=EXPORT_MODEL_DIR) -> "ModelTrainingResult":
        target_path = f'{target_folder}/{model_key}.dill'

        with open(target_path, "rb") as targt_file:
            return load(targt_file)


class TuneType(Enum):
    Random = 1
    Grid = 2


@dataclass
class CMResultsDataStats:
    y_test: pd.Series
    x_test: pd.DataFrame
    predictions: pd.Series
    probabilities: pd.Series
    # probabilities_match_id: pd.Series
    metrics: Dict
    class_accuracies: Dict


ModelConfigsCollection = Dict[str, ModelPipelineConfig]


def estimate_transformer_impact(data: pd.DataFrame):
    """
    Calculated the impact on performance of each transformer based on tuning results
    the input df must contain feature names as columns prefixed with 'feat_trans'
    :param data:
    :return:
    """

    data = data.dropna()
    # Define the feature columns (transformer options)
    feature_cols = [col for col in data.columns if col.startswith('feat_trans')]
    X = data[feature_cols]
    y = data['mean_test_score']

    # Define a pipeline with OneHotEncoder for categorical variables and LinearRegression model
    preprocessor = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(drop='first'), feature_cols)],
        remainder='passthrough')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Fit the model
    model.fit(X, y)

    # Get the feature names after one-hot encoding
    feature_names = model.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out()

    # Extract coefficients (impacts) from the regression model
    # impacts = model.named_steps['regressor'].coef_
    #
    # # Map coefficients to the corresponding transformer option
    # impact_dict = dict(zip(feature_names, impacts))

    # Round the impacts to 5 digits
    impacts = np.round(model.named_steps['regressor'].coef_, 4)

    # Create a DataFrame instead of a dictionary
    impact_df = pd.DataFrame({'Transformer_Option': feature_names,
                              'Impact': impacts,
                              })

    return impact_df.sort_values(by=["Impact"], ascending=False)
