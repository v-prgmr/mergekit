import tempfile

import optuna
import yaml
from lm_eval import evaluator

import mergekit.config as config
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, add_merge_options

multiline_yaml_template = """
models:
 - model: meta-llama/Llama-2-7b-hf
   #no parameters necessary for base model
 - model: meta-llama/Llama-2-7b-chat-hf
   parameters:
     density: 0.5
     weight: 0.5
 - model: epfl-llm/meditron-7b
   parameters:
     density: 0.5
     weight: 0.5
 - model: arcee-ai/no_zero_1_step_1000
   parameters:
     density: 0.5
     weight: 0.5

merge_method: ties
base_model: meta-llama/Llama-2-7b-hf
parameters:
  normalize: false
  int8_mask: true
dtype: float16

"""


def objective(trial):
    merge_config: config.MergeConfiguration = config.MergeConfiguration.model_validate(
        yaml.safe_load(multiline_yaml_template)
    )
    for i in merge_config.models:
        if i.parameters is None:
            continue
        for parameter_key, value in i.parameters.items():
            model_name = i.model.model.path
            key = f"{model_name}/{parameter_key}"
            value = trial.suggest_float(key, 0.0, 1.0)
            i.parameters[parameter_key] = value

    print(merge_config.to_yaml())

    with tempfile.TemporaryDirectory() as temp_dir:
        run_merge(
            merge_config,
            temp_dir,
            options=MergeOptions(),
        )

        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={temp_dir}",
            batch_size=1,
            tasks=["medqa_4options"],
            device="cuda",
        )

        print(results["results"])

        return results["results"]["medqa_4options"]["acc,none"]


if __name__ == "__main__":
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=2)
    best_params = study.best_params
    print(best_params)
