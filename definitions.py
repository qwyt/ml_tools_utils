import os
from copy import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Union

from sklearn.metrics import classification_report
import pandas as pd
import yaml
import uuid
import sklearn

import shared.ml_config_core as ml_config_core

from joblib import dump, load

ModelConfig = ml_config_core.ModelConfig

TUNING_RESULTS_DIR = ".tuning_results"
# TUNING_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'tuning_results')

DatasetDescription = Dict[
    str, Union[int, Dict[str, Union[str, int]], Dict[Union[str, int], int]]
]


@dataclass
class TuningResultBestParams:
    model_key: str
    search_type: str
    scoring_function: str
    best_params: Dict[str, Union[int, float, str]]
    model_config_reference: str
    dataset_description: DatasetDescription


@dataclass
class TuningResult_OLD:
    # model_key: str
    # search_type: str
    # scoring_function: str
    best_score: float
    # best_params: Dict[str, Union[int, float, str]]
    all_scores: Dict[str, List[Union[float, int, str]]]
    result_report: Dict[str, Dict[str, Union[float, int, str]]]
    # model_config_reference: str
    # dataset_description: DatasetDescription

    best_params_info: TuningResultBestParams
    transformer_tune_results: pd.DataFrame

    unique_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    date_ran: datetime = field(default_factory=lambda: datetime.now())

    def to_yaml(self):
        if not os.path.exists(TUNING_RESULTS_DIR):
            os.makedirs(TUNING_RESULTS_DIR)

        file_path = os.path.join(TUNING_RESULTS_DIR, f"{self.best_params_info.model_key}.yaml")

        if os.path.isfile(file_path):
            os.remove(file_path)

        with open(file_path, "w") as file:
            yaml.dump(asdict(self.best_params_info), file)

    @staticmethod
    def from_yaml(unique_id: str) -> "TuningResult":
        file_path = os.path.join(TUNING_RESULTS_DIR, f"{unique_id}.yaml")
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            best_params = TuningResultBestParams(**data)
            return TuningResult(best_params_info=best_params, all_scores=None, best_score=0, result_report=None)

    # TODO: remove
    # @classmethod
    # def from_search_cv(
    #         cls,
    #         search_cv: sklearn.model_selection._search.BaseSearchCV,
    #         model_config_reference: str,
    #         features: pd.DataFrame,
    #         labels: pd.Series,
    # ) -> "TuningResult":
    #     report = classification_report(
    #         labels, search_cv.predict(features), output_dict=True
    #     )
    #     dataset_description = cls._generate_dataset_description(features, labels)
    #
    #     best_params = TuningResultBestParams(
    #         dataset_description=dataset_description,
    #         search_type=type(search_cv).__name__,
    #         scoring_function=str(search_cv.scoring),
    #         model_config_reference=model_config_reference,
    #         best_params=search_cv.best_params_,
    #         model_key="TODO"
    #     )
    #
    #     return cls(
    #         best_params_info=best_params,
    #         best_score=search_cv.best_score_,
    #         all_scores=search_cv.cv_results_,
    #         result_report=report,
    #     )

    @staticmethod
    def _generate_dataset_description(
            features: pd.DataFrame, labels: pd.Series
    ) -> DatasetDescription:
        feature_types = {col: str(features[col].dtype) for col in features.columns}
        # label_distribution = labels.value_counts().to_dict()
        label_distribution = {}
        return dict(
            num_rows=len(features),
            feature_types=feature_types,
            label_distribution=label_distribution,
        )

    @staticmethod
    def convert_to_dataframe(tuning_results: Dict[str, "TuningResult"]) -> pd.DataFrame:
        data_for_df = []
        for key, result in tuning_results.items():
            # Start with the existing data row structure
            data_row = {
                "model_key": key,
                "best_score": result.best_score,
                "best_params": result.best_params_info.best_params,
                "search_type": result.best_params_info.search_type,
                "model_config_reference": result.best_params_info.model_config_reference,
            }

            # Add classification metrics from result.result_report
            report = result.result_report  # Assume this is the structure you provided
            for label, metrics in report.items():
                if label in ['accuracy']:  # Handle overall accuracy
                    data_row[f'{label}'] = metrics
                else:
                    for metric, value in metrics.items():
                        data_row[f'{label}_{metric}'] = value

            data_for_df.append(data_row)

        df = pd.DataFrame(data_for_df)
        df.set_index("model_key", inplace=True)
        return df


@dataclass
class TuningResult:
    model_key: str
    model_pipeline_config: ml_config_core.ModelPipelineConfig
    # model_pipeline_config = pipeline_config,
    model_params: dict
    pipeline_params: dict
    dataset_description: dict
    all_scores: dict
    result_report: dict
    best_score: float
    transformer_tun_all_civ_results: pd.DataFrame
    def get_best_params(self):
        return {**self.pipeline_params, **self.model_params}

    @staticmethod
    def _generate_dataset_description(
            features: pd.DataFrame, labels: pd.Series
    ) -> DatasetDescription:
        feature_types = {col: str(features[col].dtype) for col in features.columns}
        # label_distribution = labels.value_counts().to_dict()
        label_distribution = {}
        return dict(
            num_rows=len(features),
            feature_types=feature_types,
            label_distribution=label_distribution,
        )

    @staticmethod
    def serialize_tuning_result(res, target_folder=TUNING_RESULTS_DIR):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        dump(res, f'{target_folder}/{res.model_key}.dill')

    @staticmethod
    def load_serialized_tuning_result(model_key, target_folder=TUNING_RESULTS_DIR):
        return load(f'{target_folder}/{model_key}.dill')

    @staticmethod
    def convert_to_dataframe(tuning_results: Dict[str, "TuningResult"]) -> pd.DataFrame:
        data_for_df = []
        for key, result in tuning_results.items():
            # Start with the existing data row structure
            data_row = {
                "model_key": key,
                "best_score": result.best_score,
                "best_params": result.get_best_params(),
            }

            # Add classification metrics from result.result_report
            report = result.result_report  # Assume this is the structure you provided
            for label, metrics in report.items():
                if label in ['accuracy']:  # Handle overall accuracy
                    data_row[f'{label}'] = metrics
                elif not isinstance(metrics, dict):
                    data_row[f'{label}'] = metrics
                else:
                    for metric, value in metrics.items():
                        data_row[f'{label}_{metric}'] = value

            data_for_df.append(data_row)

        df = pd.DataFrame(data_for_df)
        df.set_index("model_key", inplace=True)
        return df


class TuningResultsAPI:
    @staticmethod
    def load_all_results() -> Dict[str, TuningResult]:
        results = {}
        for file_name in os.listdir(TUNING_RESULTS_DIR):
            if file_name.endswith(".yaml"):
                unique_id = file_name.split(".")[0]
                result = TuningResult.from_yaml(unique_id)
                results[unique_id] = result
        return results

    @staticmethod
    def get_model_configs_with_hyperparams(
            model_configs: Dict[str, ml_config_core.ModelPipelineConfig],
            skip_missing=False
    ) -> Dict[str, ml_config_core.ModelPipelineConfig]:
        tunning_results = TuningResultsAPI.load_all_results()

        results: Dict[str, ml_config_core.ModelPipelineConfig] = {}

        for model_key in model_configs.keys():
            results[model_key] = copy(model_configs[model_key])

            if not model_key in tunning_results:
                if model_configs[model_key].model_config.best_params is not None:
                    best_params = model_configs[model_key].model_config.best_params
                elif skip_missing:
                    continue
                else:
                    raise Exception(
                        f"Tunning not performed for {model_key}, run 'run_tunning_for_config' before fitting"
                    )
            else:
                best_params = tunning_results[model_key].best_params_info.best_params

            results[model_key].model_config.best_params = best_params

        return results
