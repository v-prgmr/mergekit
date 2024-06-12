# Phi-3 MoE generation
![phi3xtral.jpg](output_phi3_moe%2Fphi3xtral.jpg)
The [mergekit](https://github.com/arcee-ai/mergekit) in its original flavour does not support [microsoft/phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) (at the time of writing this article) because of mismatch in layers names. This fork was done to make the mergekit work with [microsoft/phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) based SLMs to create a "Phi2-Mixture of Experts" model. Follow the instructions below to create your own Mixture of Experts from multiple individual phi-3 experts. Please checkout the "phi3xtral" branch to start with.

## Instructions for creating Phi-3 MoE
Checkout the `phi3xtral` branch of this repository.
## Merging experts into MoE
- Craete a merge configuration like [config_moe_phi3.yaml](output_phi3_moe/config_moe_phi3.yml) where you either pass in the absolute path of the expert model's directory or to the expert's huggingface repository.
- Run [phi3_moe.py](mergekit/scripts/phi3_moe.py) by passing the following arguments to it 
  - merge configuration: [config_moe_phi3.yaml](output_phi3_moe/config_moe_phi3.yml) 
  - path to output folder: for example you can use this folder [output_phi3_moe](output_phi3_moe) (as it has the configuration files needed for inferencing the MoE model)
  - load-in-4bit 
  - trust-remote-code
  - therefore the run command looks like: `python /mergekit/scripts/phi3_moe.py /output_phi3_moe/config_moe_phi3.yaml output_phi3_moe --load-in-4bit --trust-remote-code --allow-crimes --out-shard-size 1B --lazy-unpickle --copy-tokenizer --i-understand-this-is-not-useful-without-training`
- This should now create the Mixture of Experts model from your individual experts as per the merge configuration inside the output directory.
- **Note**: If you are using your own custom finetuned phi-3, that was fine tuned using techniques like Qlora, merge the adapter weights back to the base model before using it as one of the experts in the mergekit.
  - You can read about merging the adapters to base model here: https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge
## Inference
- You will find the customized configuration and modelling file within the `output_phi3_moe` folder.
  - You will need `configuration_phi_3.py` and `m̀odelling_phi_3.py` for inference of your Phi2MoE.
  - Create your inference script as you would normally do using the huggingface transformer library and load the MoE you just craeted above.
  - The model will use the customized configuration files present inside the output folder as per the `config.json` that gets created from the previous step using `phi3_moe.py`
- Enjoy your Phi2MoE!

# mergekit

`mergekit` is a toolkit for merging pre-trained language models. `mergekit` uses an out-of-core approach to perform unreasonably elaborate merges in resource-constrained situations. Merges can be run entirely on CPU or accelerated with as little as 8 GB of VRAM. Many merging algorithms are supported, with more coming as they catch my attention.

Features:

- Supports Llama, Mistral, GPT-NeoX, StableLM, and more
- Many [merge methods](#merge-methods)
- GPU or CPU execution
- Lazy loading of tensors for low memory use
- Interpolated gradients for parameter values (inspired by Gryphe's [BlockMerge_Gradient](https://github.com/Gryphe/BlockMerge_Gradient) script)
- Piecewise assembly of language models from layers ("Frankenmerging")

## Installation

```sh
git clone https://github.com/cg123/mergekit.git
cd mergekit

pip install -e .  # install the package and make scripts available
```

If the above fails with the error of:

```
ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode:
(A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
```

You may need to upgrade pip to > 21.3 with the command `python3 -m pip install --upgrade pip`

## Usage

The script `mergekit-yaml` is the main entry point for `mergekit`. It takes a YAML configuration file and an output path, like so:

```sh
mergekit-yaml path/to/your/config.yml ./output-model-directory [--cuda] [--lazy-unpickle] [--allow-crimes] [... other options]
```

For more information on the arguments accepted by `mergekit-yaml` run the command `mergekit-yaml --help`.

## Merge Configuration

Merge configurations are YAML documents specifying the operations to perform in order to produce your merged model.
Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. See [Merge Methods](#merge-methods) for a list.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type used for the merging operation.
- `tokenizer_source`: Determines how to construct a tokenizer for the merged model.

### Parameter Specification

Parameters are flexible and can be set with varying precedence. They can be specified conditionally using tensor name filters, which allows finer control such as differentiating between attention heads and fully connected layers.

Parameters can be specified as:

- **Scalars**: Single floating-point values.
- **Gradients**: List of floating-point values, specifying an interpolated gradient.

The parameters can be set at different levels, with decreasing precedence as follows:

1. `slices.*.sources.parameters` - applying to a specific input slice
2. `slices.*.parameters` - applying to a specific output slice
3. `models.*.parameters` or `input_model_parameters` - applying to any tensors coming from specific input models
4. `parameters` - catchall

### Tokenizer Source

The `tokenizer_source` field of a configuration file determines what tokenizer is used by the merged model. This also effects how embeddings and language model heads are merged.

This functionality is still experimental and may break. Please file an issue if you encounter any issues with it.

Valid values:

- `base`: use the tokenizer from the base model
- `union`: construct a tokenizer with all tokens from all models
- `model:<model_path>`: use the tokenizer from a specific model

If set, mergekit will find a mapping between each model's vocabulary and the output tokenizer. This allows models with different vocabularies or added tokens to be meaningfully merged.

`tokenizer_source` is compatible with all merge methods, but when used `lm_head`/`embed_tokens` will be merged linearly. For two-model merges, the `embed_slerp` parameter can be set to `true` to use SLERP instead.

If the `tokenizer_source` field is not set, mergekit will fall back to its legacy default behavior. The tokenizer for the base model (or first model in the merge, if no base model is specified) will be copied to the output directory. The parameter matrices for `lm_head`/`embed_tokens` will be truncated to the smallest size present in the merge. In _most_ cases this corresponds to using the tokenizer for the base model.

### Examples

Several examples of merge configurations are available in [`examples/`](examples/).

## Merge Methods

A quick overview of the currently supported merge methods:

| Method                                                                                       | `merge_method` value | Multi-Model | Uses base model |
| -------------------------------------------------------------------------------------------- | -------------------- | ----------- | --------------- |
| Linear ([Model Soups](https://arxiv.org/abs/2203.05482))                                     | `linear`             | ✅          | ❌              |
| SLERP                                                                                        | `slerp`              | ❌          | ✅              |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                          | `task_arithmetic`    | ✅          | ✅              |
| [TIES](https://arxiv.org/abs/2306.01708)                                                     | `ties`               | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [TIES](https://arxiv.org/abs/2306.01708)            | `dare_ties`          | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [Task Arithmetic](https://arxiv.org/abs/2212.04089) | `dare_linear`        | ✅          | ✅              |
| Passthrough                                                                                  | `passthrough`        | ❌          | ❌              |

### Linear

The classic merge method - a simple weighted average.

Parameters:

- `weight` - relative (or absolute if `normalize=False`) weighting of a given tensor
- `normalize` - if true, the weights of all models contributing to a tensor will be normalized. Default behavior.

### SLERP

Spherically interpolate the parameters of two models. One must be set as `base_model`.

Parameters:

- `t` - interpolation factor. At `t=0` will return `base_model`, at `t=1` will return the other one.

### [Task Arithmetic](https://arxiv.org/abs/2212.04089)

Computes "task vectors" for each model by subtracting a base model. Merges the task vectors linearly and adds back the base. Works great for models that were fine tuned from a common ancestor. Also a super useful mental framework for several of the more involved merge methods.

Parameters: same as [Linear](#linear)

### [TIES](https://arxiv.org/abs/2306.01708)

Builds on the task arithmetic framework. Resolves interference between models by sparsifying the task vectors and applying a sign consensus algorithm. Allows you to merge a larger number of models and retain more of their strengths.

Parameters: same as [Linear](#linear), plus:

- `density` - fraction of weights in differences from the base model to retain

### [DARE](https://arxiv.org/abs/2311.03099)

In the same vein as TIES, sparsifies task vectors to reduce interference. Differs in that DARE uses random pruning with a novel rescaling to better match performance of the original models. DARE can be used either with the sign consensus algorithm of TIES (`dare_ties`) or without (`dare_linear`).

Parameters: same as [TIES](#ties) for `dare_ties`, or [Linear](#linear) for `dare_linear`

### Passthrough

`passthrough` is a no-op that simply passes input tensors through unmodified. It is meant to be used for layer-stacking type merges where you have only one input model. Useful for frankenmerging.
