import os
from copy import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Union
import pandas as pd
import shared.ml_config_core as ml_config_core
from joblib import dump, load

ModelConfig = ml_config_core.ModelConfig

TUNING_RESULTS_DIR = ".tuning_results"

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
class TuningResult:
    model_key: str
    model_pipeline_config: ml_config_core.ModelPipelineConfig
    model_params: dict
    pipeline_params: dict
    dataset_description: dict
    all_scores: dict
    result_report: dict
    best_score: float
    transformer_tun_all_civ_results: pd.DataFrame
    hyper_param_all_cv_results: pd.DataFrame

    def get_best_params(self):
        return {**self.pipeline_params, **self.model_params}

    @staticmethod
    def _generate_dataset_description(
        features: pd.DataFrame, labels: pd.Series
    ) -> DatasetDescription:
        feature_types = {col: str(features[col].dtype) for col in features.columns}
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

        dump(res, f"{target_folder}/{res.model_key}.dill")

    @staticmethod
    def load_serialized_tuning_result(model_key, target_folder=TUNING_RESULTS_DIR):
        return load(f"{target_folder}/{model_key}.dill")

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
        model_configs: Dict[str, ml_config_core.ModelPipelineConfig], skip_missing=False
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
