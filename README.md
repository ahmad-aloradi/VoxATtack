# Robust Speaker Recognition Against Adversarial Attacks and Spoofing

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/gorodnitskiy/yet-another-lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![Publication](https://img.shields.io/badge/Paper-WASPAA%202025-green)](https://waspaa.com/)<br>

## Description

This is the official implementation for the paper: "VoxATack a MultiModal Attack on Voice Anonymization Systems".

The framework is based on [this template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template), which is based on
[PyTorch Lightning](https://github.com/Lightning-AI/lightning) and [Hydra](https://github.com/facebookresearch/hydra). 


## Results
Our results can be found [IN THIS LINK](https://faubox.rrze.uni-erlangen.de/getlink/fi9RnHmqs8AUdfp5tiNsQX/proposed_results%20Table2)


## Get started

```shell
# clone template
git clone https://github.com/ahmad-aloradi/VoxATtack.git
cd VoxATtack

# install requirements
pip install -r requirements.txt
```

## Project structure
The structure is directly inherited from the used [template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template). It is structured as follows:


```
├── configs                     <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── debug                   <- Debugging configs
│   ├── experiment              <- Experiment configs
│   ├── extras                  <- Extra utilities configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── hydra                   <- Hydra settings configs
│   ├── local                   <- Local configs
│   ├── logger                  <- Logger configs
│   ├── module                  <- Module configs
│   ├── paths                   <- Project paths configs
│   ├── trainer                 <- Trainer configs
│   │
│   ├── eval.yaml               <- Main config for evaluation
│   └── train.yaml              <- Main config for training
│
├── data                        <- Project data
├── logs                        <- Logs generated by hydra, lightning loggers, etc.
├── notebooks                   <- Jupyter notebooks.
├── scripts                     <- Shell scripts
│
├── src                         <- Source code
│   ├── callbacks               <- Additional callbacks
│   ├── datamodules             <- Lightning datamodules
│   ├── modules                 <- Lightning modules
│   ├── utils                   <- Utility scripts
│   │
│   ├── eval.py                 <- Run evaluation
│   └── train.py                <- Run training
│
├── tests                       <- Tests of any kind
│
├── .dockerignore               <- List of files ignored by docker
├── .gitattributes              <- List of additional attributes to pathnames
├── .gitignore                  <- List of files ignored by git
├── .pre-commit-config.yaml     <- Configuration of pre-commit hooks for code formatting
├── Dockerfile                  <- Dockerfile
├── Makefile                    <- Makefile with commands like `make train` or `make test`
├── pyproject.toml              <- Configuration options for testing and linting
├── requirements.txt            <- File for installing python dependencies
├── setup.py                    <- File for installing project as a package
└── README.md
```


## Dataset and Data Preparation

### Downloading Dataset
We are using the `VoicePrivacy2025` and `Librispeech` datasets. Please download them before hand to use the repo. You need to contact the [VPAC 2025 challenge organizers](https://www.voiceprivacychallenge.org/attacker/) to obtain the dataset. You may optionally download `LibriSpeech` if you need to train or evaluate on the original speech. If you choose not to use it, please remove it from the data configs in `configs/datamodule/datasets/vpc.yaml`.

Once downloaded, we recommend you create a symlink to the data folder (if did not save it there)
```shell
ln -s YOUR_DATA_PATH data/.
```
If your data resides elsewhere, you may override the data by 
```shell
python src/train.py paths.data_dir='PATH_TO_YOUR_DATA'
```

### Specfying a Dataset
By default, all models are used in the training/. This is euivalent to running:
```bash
python src/train datamodule.models=${datamodule.available_models}
```
You can train/evaluate against a specific anonymization model (e.g. B3) by overriding the key via command line:
```bash
python src/train datamodule.models={B3: ${datamodule.available_models.B3}}
```

You can train/evaluate against multiple anonymization model (e.g. B3 & LibriSpeech) by overriding the key via command line:
```bash
python src/train datamodule.models={librispeech: ${datamodule.available_models.librispeech}, B3:${datamodule.available_models.B3}}'
```

### Generate Metadata
- Execute `scripts/datasets/prep_vpc.sh` to generate metadata for the anonymization models `B3`, `B4`, `B5`, `T8-5`, `T10-2`, and `T25-1`. Upon successful execution, this script creates `metadata` folders stored in `vpc2025_official/ANON_MODEL/data/metadata`.

- **Optional**: To generate equivalent metadata for the original, non-anonymized LibriSpeech dataset (e.g., for ASV splits), run `src/datamodules/components/vpc25/01_OPT_convert_b3_to_librispeech.py`. This script generates metadata based on the `B3` metadata from the previous step.

- When extracting the `T25-1` model's data, a folder name contains a typo. **Please correct this manually**.

## Usage

### Training

The framework uses Hydra for configuration management. You can train models using different experiment configurations:

#### Basic Training
```bash
# Train with default configuration
python src/train.py

# Train with custom parameters
python src/train.py trainer.max_epochs=50 datamodule.loaders.train.batch_size=32
```

#### Available VPC Experiments (Recommended)

1. **VoxATtack (multimodal attack)**

   ```shell
   python src/train.py experiment=vpc/voxattack_base
   ```
   - Uses multimodal approach with audio and text features
   - Includes ensemble, fusion, audio, and text loss components

2. **VoxATtack With Data Augmentation (SpeechAugment)**
   ```bash
   python src/train.py experiment=vpc/voxattack_aug
   ```
   - Includes noise addition, reverberation, frequency dropping, chunk dropping, and speed perturbation
   - Automatically downloads and prepares noise and RIR datasets

3. **ECAPA_ours (Audio-Only Model)**
   ```bash
   python src/train.py experiment=vpc/ecapa_ours_base
   ```
   - Audio-only approach using robust audio model
   - Using a single AAM loss term

4. **ECAPA_ours with Data Augmentation (SpeechAugment)**
   ```bash
   python src/train.py experiment=vpc/ecapa_ours_aug
   ```
   - Audio-only approach using robust audio model
   - Simplified loss function without multimodal components

---

### Evaluation

#### Basic evaluation
```bash
# Evaluate a trained model
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```


#### Detailed Evaluation (Recommended)

- After training completion, your experiment directory will contain the following **Key Output Directories:**
   - `valid_artifacts/`: Validation evaluation results
   - `test_artifacts_TimeStamp/`: Test evaluation results  
   - `checkpoints/`: Saved model checkpoints
   - `tensorboard/`: Training logs and metrics

   You may customize the output directories by overriding the default log directory and/or run using:
   ```bash
   # Custom logs directory
   python src/train.py paths.log_dir=RESULTS_DIR

   # Custom run directory  
   python src/train.py hydra.run.dir=RESULTS_DIR/train/runs/RUN_NAME
   ```

   **The Default paths are:**
   - Log directory: `logs/`
   - Run directory: `${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}`


- Use `notebooks/test_results_analysis.ipynb` for comprehensive evaluation analysis. Configure the following variables:

   ```python
   # Required configuration
   MODELS_PATH = "PATH_TO_YOUR_MODELS"  # Update to your experiments directory

   # Evaluation mode
   EVAL_MODE = "EVAL_ALL"  # Options: "SINGLE" or "EVAL_ALL"

   # Single experiment evaluation (when EVAL_MODE = "SINGLE")
   SINGLE_EXPERIMENT = 'voxattack_base-B3-max_dur10-bs32'

   # Batch evaluation (when EVAL_MODE = "EVAL_ALL") 
   EXPERIMENT_PATTERN = '*'  # Regex pattern to select experiments

   # Data split selection
   EVAL_TEST = True  # True for test data, False for validation data
   ```

---
### Data Augmentation Configuration

#### Classical Audio Signals Augmentation
When using the augmentation experiment (`*_aug`), the model will automatically:

1. Download noise dataset from Dropbox if not present
2. Download RIR (Room Impulse Response) dataset if not present
3. Prepare CSV annotations for both datasets
4. Apply some or all of the following augmentations during training:
   - **Noise Addition**: Randomly adds background noise with SNR levels between 0-15 dB
   - **Reverberation**: Applies room impulse responses to simulate different acoustic environments
   - **Frequency Dropping**: Randomly removes frequency bands from the spectrum  (enabled by default ✅)
   - **Chunk Dropping**: Randomly removes temporal segments from the audio (enabled by default ✅)
   - **Speed Perturbation**: Applies speed variations of 90% and 110% to the original speech (enabled by default ✅)

#### Augmentation with Anonymized Speech 
The configuration for all datasets is defined in `configs/datamodule/datasets/vpc.yaml`. Each anonymization dataset (and the original) contains several keys for specifying the locations of train/dev/test/enrollment data. By default, we provide an example showing how LibriSpeech's training data can be used to augment the dataset.

```yaml
   LibriSpeech:
      data_dir: ${datamodule.root_dir}/librispeech
      metadata: ${datamodule.available_models.LibriSpeech.data_dir}/vpc_metadata
      train: ${datamodule.available_models.LibriSpeech.metadata}/train.csv
      # # Uncomment when evaluating the clean model only!
      # dev: ${datamodule.available_models.LibriSpeech.metadata}/dev.csv
      # test: ${datamodule.available_models.LibriSpeech.metadata}/test.csv
      # dev_enrolls: ${datamodule.available_models.LibriSpeech.metadata}/dev_enrolls.csv
      # test_enrolls: ${datamodule.available_models.LibriSpeech.metadata}/test_enrolls.csv
      # dev_trials: ${datamodule.available_models.LibriSpeech.metadata}/dev_trials.csv
      # test_trials: ${datamodule.available_models.LibriSpeech.metadata}/test_trials.csv
```

### Logging

The framework supports multiple logging backends:

```bash
# Use WandB (TensorBoard used by default)
python src/train.py logger=wandb

# Use multiple loggers --> define which loggers configs/logger/many_loggers.yaml
python src/train.py logger=many_loggers
```

### GPU/CPU Training

```bash
# Train on GPU (default)
python src/train.py trainer=gpu

# Multi-GPU training
python src/train.py trainer=ddp
```

## Remarks

### Known Issues
1. **VoicePrivacy2025 dataset**: When extracting the `T25-1` model's data, a folder name contains a typo. Please correct this manually.
2. **LibriSpeech dataset**: Line 60 in `SPEAKERS.TXT` previously caused parsing issues when loading as CSV with `sep='|'`. This is now handled automatically.

### Troubleshooting
Use debug configurations if needed

```bash
# Fast development run with limited data
python src/train.py debug=default

# Overfit on small batch for debugging
python src/train.py debug=overfit
```

### Validation Loop
Currently, the validation data does not perform score normalization when computing the scores. This means that:
1. The best model is saved based on the raw scores, not the normalized scores. 
2. The scores saved in `valid_best_scores.csv` and `valid_best_metrics.json` (under `valid_artifacts`) are not based on the normalized scores.

For the purpose of evaluation, `notebooks/test_results_analysis.ipynb` generates the raw and normalized scores for each model on both dev and test. 


## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{aloradi2025voxattack,
   title={VoxATack: a MultiModal Attack on Voice Anonymization Systems},
   author={Aloradi, Ahmad and Gaznepoglu, Ünal Ege and Habets, Emanuël A.P. and Tenbrinck, Daniel},
   booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
   year={2025},
   address={Lake Tahoe, CA, USA}
}
```


