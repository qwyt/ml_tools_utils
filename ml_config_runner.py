import time
from typing import Callable, Dict

import pandas as pd

from shared import pipeline, definitions

from shared.definitions import TuningResult
from shared.ml_config_core import (
    ModelConfigsCollection,
    ModelPipelineConfig,
    ModelTrainingResult,
    ModelTrainingResultMetadata,
)


# def run_tuning_for_config(
#     model_name: str, pipeline_config: ModelPipelineConfig, df: pd.DataFrame
# ) -> TuningResult:
#     start_time = time.time()
#
#     print(
#         f"Tunning:"
#         f" - transformers: {pipeline_config.transformer_config}"
#         f""
#         f"\n\n - model: {model_name} n_iters={pipeline_config.model_config.search_n_iter} with:\n {pipeline_config.model_config.param_grid}"
#     )
#
#     tunning_result = pipeline.run_tunning_for_config(
#         model_key=model_name, pipeline_config=pipeline_config, df=df
#     )
#
#     end_time = time.time()
#     elapsed_time = round(end_time - start_time, 1)
#     print(f"{model_name} Fit, total time:{elapsed_time}\n")
#
#     return tunning_result
def run_tuning_for_config(
    model_name: str, pipeline_config: ModelPipelineConfig, df: pd.DataFrame
) -> TuningResult:
    start_time = time.time()

    print(
        f"Tunning:"
        f" - transformers: {pipeline_config.transformer_config}"
        f""
        f"\n\n - model: {model_name} n_iters={pipeline_config.model_config.search_n_iter} with:\n {pipeline_config.model_config.param_grid}"
    )

    tunning_result = pipeline.run_bayesian_tuning_for_config(
        model_key=model_name, pipeline_config=pipeline_config, df=df
    )

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 1)
    print(f"{model_name} Fit, total time:{elapsed_time}\n")
    return tunning_result


def run_tuning_for_configs_collection(
    model_configs: ModelConfigsCollection, load_df: Callable[..., pd.DataFrame]
) -> Dict[str, TuningResult]:
    results = {}
    for model_key, model_config in model_configs.items():
        df = model_config.load_data(loader_function=load_df)

        tune_results = run_tuning_for_config(
            model_name=model_key, pipeline_config=model_config, df=df
        )

        TuningResult.serialize_tuning_result(tune_results)
        results[model_key] = tune_results
    return results


def build_production_model_for_tuning_result(
    tuning_result: TuningResult,
    load_df: Callable[..., pd.DataFrame],
    random_state: int = 420,
) -> ModelTrainingResult:
    start_time = time.time()

    df = tuning_result.model_pipeline_config.load_data(loader_function=load_df)

    result = pipeline.run_pipeline_config(tuning_result, df, random_state=random_state)

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 1)

    print(f"{tuning_result.model_key}: {elapsed_time} seconds")

    result.meta_data = ModelTrainingResultMetadata(
        elapsed_time=elapsed_time,
    )

    return result


def TODO_refactor_run_tuning_for_configs_collection(
    model_configs: ModelConfigsCollection,
    features: pd.DataFrame,
    labels: pd.Series,
    use_optuna: bool = False,
    only_tune_transformers=False,
):
    res = {}
    transformer_tune_results = {}
    include_models = [k for k in model_configs.keys() if model_configs[k][1]]

    for model_name in include_models:
        pipeline_config: ModelPipelineConfig = model_configs[model_name][0]

        start_time = time.time()

        print(
            f"Tunning:"
            f" - transformers: {pipeline_config.transformer_config}"
            f""
            f"\n\n - model: {model_name} n_iters={pipeline_config.model_config.search_n_iter} with:\n {pipeline_config.model_config.param_grid}"
        )

        sample_indices = features.index
        model_config = pipeline_config.model_config

        tunning_result = pipeline.run_tunning_for_config(
            model_key=model_name,
            pipeline_config=pipeline_config,
            features=features.loc[sample_indices],
            labels=labels.loc[sample_indices],
        )
        tunning_result.to_yaml()
        res[model_name] = tunning_result

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 1)

        transformer_tune_results[model_name] = tunning_result.transformer_tune_results
        print(f"Total time:{elapsed_time}\n")

    if len(include_models) > 0:
        tunning_result_res_df = definitions.TuningResult.convert_to_dataframe(res)
        return tunning_result_res_df, transformer_tune_results
    return None


def TODO_REMOVE_run_cv_configs(
    get_config: Callable[[], ModelConfigsCollection], features, labels
):
    model_task_infos = get_config()
    model_configs = {key: value[0] for key, value in model_task_infos.items()}

    model_configs_with_params = (
        definitions.TuningResultsAPI.get_model_configs_with_hyperparams(
            model_configs, skip_missing=True
        )
    )

    cv_results = {}

    for model_name, cfg in model_configs_with_params.items():
        start_time = time.time()

        sample_indices = features.sample(100000, random_state=42).index

        result = pipeline.run_pipeline_config(
            config=cfg,
            export_prod=True,
            features=features.loc[sample_indices],
            labels=labels.loc[sample_indices],
            export_test=True,
        )
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 1)

        print(f"{model_name}: {elapsed_time} seconds")

        cv_results[model_name] = result

    cv_results_df = pipeline.build_cv_results_table(cv_results, VERBOSE=False)
    return cv_results_df, cv_results
