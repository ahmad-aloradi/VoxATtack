# VoxATtack: a Multimodal Attack on Voice Anonymization Systems

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/gorodnitskiy/yet-another-lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![Publication](https://img.shields.io/badge/Paper-WASPAA%202025-green)](https://waspaa.com/)<br>

## Description

This is the official implementation for the paper: "VoxATtack: a Multimodal Attack on Voice Anonymization Systems".

The framework is based on [this template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template), which is based on
[PyTorch Lightning](https://github.com/Lightning-AI/lightning) and [Hydra](https://github.com/facebookresearch/hydra). 


## Table of Contents

- [Get Started](#get-started)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset and Data Preparation](#dataset-and-data-preparation)
- [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Data augmentation](#data-augmentation)
- [Remarks](#remarks)
- [Citation](#citation)


## Get Started

```shell
# clone template
git clone https://github.com/ahmad-aloradi/VoxATtack.git
cd VoxATtack

# create conda environment
conda create -n voxattack python=3.11 -y
conda activate voxattack

# install requirements
pip install -r requirements.txt
```


## Results
Our results can be found [here](https://faubox.rrze.uni-erlangen.de/getlink/fi5V82Lijua16fSGwE2Wok/waspaa2025). 

### Notes
   - Currently, we only add results on the test set
   - The results show more details than what has been shown in the paper. E.g., DET curves and per-speaker evaluation (only per-gender evaluation was shown)
   - We use the normalized scores for all models except `T8-5` 

### Main results
Our demonstrate that utilizing text improves the attacks performance against voice anonymization systems. The following are Table 1 and Figure 3 in the paper, which summarize the key findings from our experimental evaluation:

1. Comparison between the proposed VoxATtack and first VoicePrivacy Attacker challenge winners against all anonymization systems. Our model outperforms the winners on all systems but `T12-5` and `T25-1`.
<p align="center">
   <img src="assets/Table1.png" alt="Table 1: Performance of several attacker systems" width="400"/>
</p>
<p>
   <em>Table 1: Performance of several attacker systems in EER_avg [%]. `ECAPA_baseline` refers to the official baseline, while `ECAPA_ours` is our audio-only system. Systems A.5 and A.20 together achieve the top scores against every anonymization system in the VPAC data.</em>
</p>

2. Using SpecAugment and/or anonymized data from other anonymization systems can notably improve the attacks, allowing VoxATtack to surpass the VPAC winner on the two remaining models.
<p align="center">
   <img src="assets/Figure3.png" alt="Figure 3: Performance comparison across different anonymization systems" width="400"/>
</p>
<p>
   <em>Figure 3: Effect of different augmentations on VoxATtack against T10-2, T12-5, and T25-1 reported in EER_avg [%].</em>
</p>

## Project Structure
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

### Downloading datasets
We use the `VoicePrivacy2025` and `LibriSpeech` datasets. Please download them beforehand to use the repository. Please contact the [VPAC 2025 challenge organizers](https://www.voiceprivacychallenge.org/attacker/) to obtain the attacker challenge dataset. You may optionally download `LibriSpeech` if you need to train or evaluate on the original speech. If you choose not to use it, please remove it from the data configs in `configs/datamodule/datasets/vpc.yaml`.

Once downloaded, we use the `data` folder to point to the dataset. You may, e.g., create a symlink to the data folder:
```shell
ln -s YOUR_DATA_PATH data/.
```
If you choose for your data to reside elsewhere, you may override the data path by:
```shell
python src/train.py paths.data_dir='PATH_TO_YOUR_DATA'
```

### Specifying a dataset
By default, all models are used in training/evaluation. This is equivalent to running:
```bash
python src/train.py datamodule.models=${datamodule.available_models}
```
You can train/evaluate against a specific anonymization model (e.g., B3) by overriding the key via command line:
```bash
python src/train.py datamodule.models={B3: ${datamodule.available_models.B3}}
```

You can train/evaluate against multiple anonymization models (e.g., B3 & LibriSpeech) by overriding the key via command line:
```bash
python src/train.py datamodule.models={librispeech: ${datamodule.available_models.librispeech}, B3:${datamodule.available_models.B3}}
```

### Generate metadata
- Execute `scripts/datasets/prep_vpc.sh` to generate metadata for the anonymization models `B3`, `B4`, `B5`, `T8-5`, `T10-2`, and `T25-1`. Upon successful execution, this script creates `metadata` folders stored in `vpc2025_official/ANON_MODEL/data/metadata`.

- **Optional**: To generate equivalent metadata for the original, non-anonymized LibriSpeech dataset (e.g., for ASV splits), run `src/datamodules/components/vpc25/01_OPT_convert_b3_to_librispeech.py`. This script generates metadata based on the `B3` metadata from the previous step.

- **Note**: When extracting the `T25-1` model's data, a folder name contains a typo. Please correct this manually.

## Usage

### Training

The framework uses Hydra for configuration management. You can train models using different experiment configurations:

#### Basic training
```bash
# Train with default configuration
python src/train.py

# Train with custom parameters
python src/train.py trainer.max_epochs=50 datamodule.loaders.train.batch_size=32
```

#### Available experiments (recommended)

1. **VoxATtack (multimodal attack)**

   ```shell
   python src/train.py experiment=vpc/voxattack_base
   ```
   - Uses multimodal approach with audio and text features
   - Includes ensemble, fusion, audio, and text loss components

2. **VoxATtack with data augmentation (SpeechAugment)**
   ```bash
   python src/train.py experiment=vpc/voxattack_aug
   ```
   - Includes noise addition, reverberation, frequency dropping, chunk dropping, and speed perturbation
   - Automatically downloads and prepares noise and RIR datasets

3. **ECAPA_ours (audio-only model)**
   ```bash
   python src/train.py experiment=vpc/ecapa_ours_base
   ```
   - Audio-only approach using robust audio model
   - Using a single AAM loss term

4. **ECAPA_ours with data augmentation (SpeechAugment)**
   ```bash
   python src/train.py experiment=vpc/ecapa_ours_aug
   ```
   - Audio-only approach using robust audio model
   - Simplified loss function without multimodal components

#### Logging

The framework supports multiple logging backends:

```bash
# Use WandB (TensorBoard used by default)
python src/train.py logger=wandb

# Use multiple loggers --> define which loggers configs/logger/many_loggers.yaml
python src/train.py logger=many_loggers
```

#### GPU/CPU training

```bash
# Train on GPU (default)
python src/train.py trainer=gpu

# Multi-GPU training
python src/train.py trainer=ddp
```


---

### Evaluation

#### Basic evaluation
```bash
# Evaluate a trained model
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```


#### Detailed Evaluation (recommended)

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
   MODELS_PATH = "PATH_TO_YOUR_MODELS"  # Update to your experiments directory
   EVAL_MODE = "EVAL_ALL"  # Options: "SINGLE" or "EVAL_ALL"
   EVAL_TEST = True  # True for test data, False for validation data

   # Single experiment eval (when EVAL_MODE = "SINGLE")
   SINGLE_EXPERIMENT = 'voxattack_base-B3-max_dur10-bs32'
   # Batch eval (when EVAL_MODE = "EVAL_ALL") 
   EXPERIMENT_REGEX = r'.*'  # Regex pattern to select experiments
   ```

---
### Data augmentation

#### Classical audio signals augmentation
When using the augmentation experiment (`*_aug`), the model will automatically:

1. Download noise dataset if not present
2. Download RIR (Room Impulse Response) dataset if not present
3. Prepare CSV annotations for both datasets
4. Apply some or all of the following augmentations during training:
   - **Noise Addition**: Randomly adds background noise with SNR levels between 0-15 dB.
   - **Reverberation**: Appliebs room impulse responses to simulate different acoustic environments based on real and simulated RIRs.
   - **Frequency Dropping**: Randomly removes frequency bands from the spectrum  (enabled by default ✅)
   - **Chunk Dropping**: Randomly removes temporal segments from the audio (enabled by default ✅)
   - **Speed Perturbation**: Applies speed variations of 90% and 110% to the original speech (enabled by default ✅)

For more information, see the different augemtations, please refer to [speechbrain's augmentations](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.augment.html)

#### Augmentation with anonymized speech 
The configuration for all datasets is defined in `configs/datamodule/datasets/vpc.yaml`. Each anonymization dataset (and the original) contains several keys for specifying the locations of train/dev/test/enrollment data. You may choose your dataset as previously described in the [Specifying a Dataset](#specifying-a-dataset) section.
If you comment out all corresponding paths except the ones to `train` (along with `data_dir` & `metadata`), this will only include the training data of that specific dataset. E.g., the following snippet shows LibriSpeech's training split used as an augmentation:
```yaml
   LibriSpeech:
      data_dir: ${datamodule.root_dir}/librispeech
      metadata: ${datamodule.available_models.LibriSpeech.data_dir}/vpc_metadata
      train: ${datamodule.available_models.LibriSpeech.metadata}/train.csv
      # # Uncomment when evaluating the clean model only!
      # dev: ${datamodule.available_models.LibriSpeech.metadata}/dev.csv
      # test: ${datamodule.available_models.LibriSpeech.metadata}/test.csv
      #...
```


## Remarks

1. **Known issues**
   - **VoicePrivacy2025 dataset**: When extracting the `T25-1` model's data, a folder name contains a typo. Please correct this manually.
   - **LibriSpeech dataset**: Line 60 in `SPEAKERS.TXT` previously caused parsing issues when loading as CSV with `sep='|'`. This is now handled automatically.


2. **Troubleshooting**

   Use debug configurations if needed
   ```bash
   # Fast development run with limited data
   python src/train.py debug=default

   # Overfit on small batch for debugging
   python src/train.py debug=overfit
   ```

3. **Experiment's directory**

   - The validation data does not perform AS-norm when computing the scores. This means that:
      1. The best model is saved based on the raw scores, not the normalized scores. 
      2. The scores saved in `valid_best_scores.csv` and `valid_best_metrics.json` (under `valid_artifacts`) are not normalized scores.

   - For the purpose of evaluation, `notebooks/test_results_analysis.ipynb` generates the raw and normalized scores for each model on both dev and test. 

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{aloradi2025voxattack,
   title={VoxATtack: a MultiModal Attack on Voice Anonymization Systems},
   author={Aloradi, Ahmad and Gaznepoglu, Ünal Ege and Habets, Emanuël A.P. and Tenbrinck, Daniel},
   booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
   year={2025},
   address={Lake Tahoe, CA, USA}
}
```
