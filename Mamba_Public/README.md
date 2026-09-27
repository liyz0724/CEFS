# CEFS · Building Damage Assessment

Baseline implementation and CEFS pseudocode for building damage assessment with bi-temporal remote sensing imagery.

This repository provides a ChangeMamba-based training and evaluation pipeline for xBD. The current training entry point runs the **baseline**. The CEFS file contains an **interface-level pseudocode outline**; it is independent of baseline execution and does not reproduce the paper's method or results.

## Contents

| Component | Included |
| --- | --- |
| ChangeMamba-based model and CUDA kernel sources | Yes |
| xBD data loading and baseline training | Yes |
| Checkpoint loading, training resume and inference | Yes |
| Building localization and damage assessment metrics | Yes |
| Optional evaluation grouped by disaster event | Yes |
| CEFS | Pseudocode outline |

Dataset files and pretrained weights are obtained separately. This is a maintained project snapshot derived from ChangeMamba, not an assertion of byte-for-byte equivalence with an upstream release.

## Setup

Use Linux with an NVIDIA GPU, a CUDA-compatible PyTorch installation, and a matching CUDA toolkit with `nvcc` for compiling the selective-scan extensions. Start from a Python 3.10 environment compatible with the existing training stack.

Install PyTorch and torchvision for your CUDA environment first, then run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./kernels/selective_scan
```

The dependency list is not a locked environment. Use the same PyTorch/CUDA/compiler combination as your working training environment when possible. The complete model requires its CUDA/Triton dependencies; CPU-only execution is not a supported full-model setup here.

## Data preparation

Use preprocessed xBD images and masks arranged under `images/` and `masks/` in each split directory. This release consumes prepared PNG data; it does not convert raw annotation polygons into masks.

| Path | Contents |
| --- | --- |
| `data/xBD/train/images/` | Training image pairs |
| `data/xBD/train/masks/` | Training localization and damage masks |
| `data/xBD/val/images/` | Validation image pairs |
| `data/xBD/val/masks/` | Validation masks |
| `data/xBD/test/images/` | Held-out test image pairs |
| `data/xBD/test/masks/` | Held-out test masks |
| `data/xBD/train_set.txt` | Training sample names, one per line |
| `data/xBD/val_set.txt` | Validation sample names, one per line |
| `data/xBD/test_set.txt` | Test sample names, one per line |

The loader uses different naming conventions for training patches and evaluation images:

| Split | Example list entry | Corresponding image pair |
| --- | --- | --- |
| Training | `event_00000001_0_0` | `event_00000001_pre_disaster_0_0.png`, `event_00000001_post_disaster_0_0.png` |
| Validation / test | `event_00000001` | `event_00000001_pre_disaster.png`, `event_00000001_post_disaster.png` |

Masks use the same filenames as their corresponding images. The current loader reads the first channel of each mask, so provide three-channel PNG masks whose first channel stores integer class IDs, not a color visualization. Localization labels use `0` for background and `1` for buildings. Damage labels use `0` for background and `1–4` for no damage, minor damage, major damage and destroyed buildings. The training loader maps damage-background pixels to ignore label `255`.

Keep training, validation and held-out test sets separate. The training CLI retains the historical argument names `--test_dataset_path` and `--test_data_list_path` for its validation loader; pass the **validation split** to those arguments during training.

## Baseline training

Place an encoder checkpoint compatible with the selected YAML configuration in `pretrained_weight/`. Replace the example checkpoint filename below with your actual file.

```bash
python changedetection/script/train_MambaBDA.py \
  --dataset xBD \
  --cfg changedetection/configs/vssm1/vssm_small_224.yaml \
  --encoder_pretrained_path pretrained_weight/encoder_small.pth \
  --train_dataset_path data/xBD/train \
  --train_data_list_path data/xBD/train_set.txt \
  --test_dataset_path data/xBD/val \
  --test_data_list_path data/xBD/val_set.txt \
  --batch_size 4 \
  --crop_size 256 \
  --max_iters 50000 \
  --seed 0 \
  --model_type baseline_small \
  --model_param_path saved_models
```

These settings are usage examples, not the paper's reproduction configuration. Adjust the batch size to the available GPU memory. Training evaluates every 750 iterations and saves `best_model.pth` and `latest.pth` under the generated experiment directory. A run shorter than its first evaluation interval does not produce these evaluation checkpoints.

For full training resume, append the following argument to the same training command:

```bash
--resume_training_path saved_models/xBD/YOUR_RUN/latest.pth
```

For weight-only initialization, use `--model_checkpoint_path` instead. Do not combine weight-only initialization and training resume.

## Inference and evaluation

Use a full baseline checkpoint and the same architecture configuration used for training:

```bash
python changedetection/script/infer_MambaBDA.py \
  --dataset xBD \
  --cfg changedetection/configs/vssm1/vssm_small_224.yaml \
  --model_checkpoint_path saved_models/xBD/YOUR_RUN/best_model.pth \
  --test_dataset_path data/xBD/test \
  --test_data_list_path data/xBD/test_set.txt \
  --crop_size 256 \
  --model_type baseline_small \
  --result_saved_path results
```

Add `--report_by_event` to obtain metrics grouped by disaster event. The full-model checkpoint must match the model architecture; an encoder-only checkpoint is insufficient for evaluation.

| Metric | Meaning |
| --- | --- |
| `loc_F1` | Building localization F1 |
| `clf_F1` | Harmonic mean of F1 across the four building damage classes |
| `oa_F1` | `0.3 × loc_F1 + 0.7 × clf_F1` |

## CEFS pseudocode

The outline is located at [`changedetection/utils_func/cefs.py`](changedetection/utils_func/cefs.py).

```text
PROCEDURE CEFS_TRAINING_OBJECTIVE(batch, model, configuration):
    objective <- METHOD_SPECIFIC_PROCEDURE(batch, model, configuration)
    RETURN objective
```

`METHOD_SPECIFIC_PROCEDURE` denotes an abstract operation, not an implementation recipe. The outline intentionally leaves out internal operations, mathematical definitions and configuration values. It does not execute or alter baseline training.

## Roadmap

- [x] Baseline training and evaluation pipeline
- [x] CEFS interface-level pseudocode
- [ ] Complete CEFS implementation after paper acceptance
- [ ] Paper-specific configurations and reproduction instructions

## Tests

In an environment with the required dependencies installed:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests cover utility behavior, metrics and checkpoint handling. They do not establish full GPU training reproducibility or validate CEFS.

## Acknowledgments and license

This project builds on ChangeMamba and its VMamba/Mamba-related components. Credit belongs to the original authors of those components. Existing source notices and the supplied [LICENSE](LICENSE) are retained. This repository does not claim the baseline architecture or third-party kernels as the contribution of CEFS.

When using third-party code, pretrained weights or datasets, consult their respective terms and cite the corresponding original work. Paper-specific bibliographic information will be added with the full release.
