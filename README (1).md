# CEFS

**Temporally Coupled Cross-Event Frequency–Style Regularization for Building Damage Assessment**

CEFS studies cross-event generalization for building damage assessment from bi-temporal remote sensing imagery. It is a training-time regularization framework that leaves the inference architecture unchanged. This repository provides the project code built on ChangeMamba, with MambaBDA-Tiny as the primary backbone.

The experiments use xBD for training and event-disjoint evaluation, and EBD for zero-shot cross-dataset evaluation.

## Repository Structure

| Path | Description |
| --- | --- |
| `changedetection/models/` | Backbone and decoder implementations |
| `changedetection/datasets/` | Data loading and preprocessing |
| `changedetection/configs/` | Model configurations |
| `changedetection/script/` | Training and inference entry points |
| `changedetection/tasks/` | Training and evaluation routines |
| `changedetection/utils_func/` | Loss functions and utilities |
| `tools/prepare_xbd_event_split.py` | xBD event-based data preparation |
| `kernels/selective_scan/` | Selective-scan CUDA extensions |
| `tests/` | Utility, metric, and checkpoint tests |

## Installation

Use a Linux environment with an NVIDIA GPU. Install PyTorch and torchvision for your CUDA environment, and ensure that the matching CUDA toolkit and `nvcc` are available for building the extensions.

Run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./kernels/selective_scan
```

## Dataset Preparation

### Datasets and Protocols

| Dataset | Role | Protocol |
| --- | --- | --- |
| xBD | Training and in-dataset evaluation | Event-disjoint split: 8,202 training, 912 validation, and 1,920 test pairs |
| EBD | External evaluation | Zero-shot evaluation on 18,215 pairs from 12 events absent from xBD |

The xBD test events are Nepal flooding, Guatemala volcano, Sunda tsunami, Santa Rosa wildfire, Hurricane Matthew, and Tuscaloosa tornado. Checkpoints are selected using xBD validation overall F1. EBD is used for evaluation without fine-tuning or parameter updates.

### xBD Data Format

Prepare the xBD images and segmentation masks as PNG files. Each split contains an `images/` directory and a `masks/` directory. Sample lists contain one sample name per line, without a file extension.

| Split | Data directory | Sample list |
| --- | --- | --- |
| Training | `data/xBD/train/` | `data/xBD/train_set.txt` |
| Validation | `data/xBD/val/` | `data/xBD/val_set.txt` |
| Test | `data/xBD/test/` | `data/xBD/test_set.txt` |

The data loader expects the following naming conventions:

| Split | Example list entry | Image filenames |
| --- | --- | --- |
| Training patches | `event_00000001_0_0` | `event_00000001_pre_disaster_0_0.png` and `event_00000001_post_disaster_0_0.png` |
| Validation / test | `event_00000001` | `event_00000001_pre_disaster.png` and `event_00000001_post_disaster.png` |

Masks use the same filenames as the corresponding images. Provide three-channel masks with integer class IDs in the first channel. Localization masks use `0` for background and `1` for buildings. Damage masks use the following labels:

| Label | Class |
| --- | --- |
| 0 | Background |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |

During training, damage-background pixels are mapped to ignore label `255`.

### Event-based Split

Use `tools/prepare_xbd_event_split.py` to reorganize a prepared xBD dataset into an event-based training, validation, and test split. The script operates on existing PNG images and masks; raw polygon annotations must be converted to masks beforehand.

The input dataset root must contain all three source subsets and their sample lists:

| Source subset | Image directory | Mask directory | Sample list |
| --- | --- | --- | --- |
| Train | `train/images/` | `train/masks/` | `train_set.txt` |
| Test | `test/images/` | `test/masks/` | `test_set.txt` |
| Hold | `hold/images/` | `hold/masks/` | `hold_set.txt` |

Source training files use the `_pre_disaster_0_0.png` and `_post_disaster_0_0.png` suffixes. Source test and hold files use `_pre_disaster.png` and `_post_disaster.png`. This script expects this specific prepared naming format.

```bash
python -m pip install scikit-learn

python tools/prepare_xbd_event_split.py \
  --dataset-root data/XBD_ChangeMamba \
  --output-dir data/xBD_event \
  --seed 321 \
  --val-ratio 0.1 \
  --link-mode symlink
```

The script assigns samples from 13 configured source events to training and validation, with a stratified 90/10 split by default. Six configured events are reserved for testing. Training and validation share source events; test events are disjoint from both. The event lists are defined in `TRAIN_EVENTS` and `TEST_EVENTS` inside the script.

`symlink` creates symbolic links and requires the original files to remain accessible. Use `--link-mode copy` for a standalone copy, or `--link-mode hardlink` when source and destination are on the same filesystem. Duplicate sample identifiers and unknown event names are rejected.

The generated output contains:

| Output path | Contents |
| --- | --- |
| `data/xBD_event/data/train/` | Training images and masks |
| `data/xBD_event/data/val/` | Validation images and masks |
| `data/xBD_event/data/test/` | Test images and masks |
| `data/xBD_event/train_set.txt` | Training sample list |
| `data/xBD_event/val_set.txt` | Validation sample list |
| `data/xBD_event/test_set.txt` | Test sample list |
| `data/xBD_event/event_split_manifest.csv` | Sample assignments and source subsets |
| `data/xBD_event/event_split_summary.json` | Counts, event lists, and overlap checks |
| `data/xBD_event/event_split_audit.md` | Human-readable split report |

Review the generated summary before training. Retain the generated lists and the splitting environment for reproducibility; the script has a dependency-free fallback that can produce different assignments from scikit-learn even with the same seed. Use a new output directory for a new split.

To train on this split, replace the four data arguments in the training command with:

```bash
  --train_dataset_path data/xBD_event/data/train \
  --train_data_list_path data/xBD_event/train_set.txt \
  --test_dataset_path data/xBD_event/data/val \
  --test_data_list_path data/xBD_event/val_set.txt
```

For final evaluation, use `data/xBD_event/data/test` and `data/xBD_event/test_set.txt` in the inference command.

### EBD Data Preparation

For evaluation with the existing loader, export EBD to the same paired-image and mask format described above. The directory names below describe the prepared evaluation layout, rather than the raw EBD download structure.

| Prepared path | Contents |
| --- | --- |
| `data/EBD/test/images/` | Registered pre-event and post-event RGB PNG images |
| `data/EBD/test/masks/` | Localization and damage masks aligned with the image pairs |
| `data/EBD/test_set.txt` | Sample identifiers, one per line |

Prepare the evaluation data as follows:

1. Match the pre-event image, post-event image, localization mask, and damage mask for each sample. Keep the two timestamps and their spatial alignment consistent.
2. Export the masks using the label convention in the xBD section: binary localization and damage IDs `0–4`. Use three-channel mask PNGs with class IDs in the first channel. Resolve annotation-label mappings from the EBD annotations before export; do not use RGB visualization colors as class IDs.
3. Assign each pair an identifier such as `hurricane-ian_00000001`. Use an event prefix followed by a numeric identifier of at least six digits so event-wise reporting can recover the event name.
4. Save the pair as `<identifier>_pre_disaster.png` and `<identifier>_post_disaster.png` under `images/`. Save the corresponding masks with the same filenames under `masks/`.
5. Write the identifiers, without extensions or timestamp suffixes, to `test_set.txt`. Check for missing files, duplicate identifiers, mismatched dimensions, and invalid label values.
6. Apply the same input preprocessing to every model being compared. Keep EBD outside training and checkpoint selection.

The manuscript evaluates Turkey earthquake, Hurricanes Delta, Dorian, Ian, Ida, Irma and Laura, Mount Semeru eruption, Pakistan flooding, St. Vincent volcano, Texas tornadoes, and Tonga volcano. For the full evaluation set, check that the prepared list contains 18,215 pairs across these 12 events.

## Training

The following example trains the ChangeMamba baseline. Place a compatible encoder checkpoint in `pretrained_weight/` and replace `encoder_tiny.pth` with its filename.

```bash
python changedetection/script/train_MambaBDA.py \
  --dataset xBD \
  --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml \
  --encoder_pretrained_path pretrained_weight/encoder_tiny.pth \
  --train_dataset_path data/xBD/train \
  --train_data_list_path data/xBD/train_set.txt \
  --test_dataset_path data/xBD/val \
  --test_data_list_path data/xBD/val_set.txt \
  --batch_size 16 \
  --crop_size 256 \
  --max_iters 50000 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --seed 0 \
  --model_type baseline_tiny \
  --model_param_path saved_models
```

The baseline example uses the training schedule described in the manuscript: 50,000 iterations, 256 × 256 crops, batch size 16, and AdamW with learning rate `1e-4` and weight decay `5e-3`. Repeat experiments with seeds `0`, `1`, and `2`. Adjust batch size if required by GPU memory. The training arguments `--test_dataset_path` and `--test_data_list_path` specify the validation data used for checkpoint selection. Use a separate held-out split for final evaluation.

Checkpoints are saved under `saved_models/xBD/<run>/`. Evaluation occurs every 750 iterations; `best_model.pth` stores the best evaluated model and `latest.pth` stores the latest evaluation checkpoint.

### Resume Training

To resume training, append the following option to the training command:

```bash
--resume_training_path saved_models/xBD/YOUR_RUN/latest.pth
```

Use `--model_checkpoint_path` for model-weight initialization without restoring the optimizer state. These two options are mutually exclusive.

## Evaluation

Load a full-model checkpoint with the same architecture configuration used for training:

```bash
python changedetection/script/infer_MambaBDA.py \
  --dataset xBD \
  --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml \
  --model_checkpoint_path saved_models/xBD/YOUR_RUN/best_model.pth \
  --test_dataset_path data/xBD/test \
  --test_data_list_path data/xBD/test_set.txt \
  --crop_size 256 \
  --model_type baseline_tiny \
  --result_saved_path results
```

Add `--report_by_event` to report results separately for each disaster event.

| Metric | Description |
| --- | --- |
| `loc_F1` | Building localization F1 |
| `clf_F1` | Harmonic mean of F1 across the four damage classes |
| `oa_F1` | Overall score: `0.3 × loc_F1 + 0.7 × clf_F1` |

### Zero-Shot Evaluation on EBD

After preparing EBD in the layout above, evaluate an xBD-trained checkpoint without updating its parameters:

```bash
python changedetection/script/infer_MambaBDA.py \
  --dataset xBD \
  --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml \
  --model_checkpoint_path saved_models/xBD/YOUR_RUN/best_model.pth \
  --test_dataset_path data/EBD/test \
  --test_data_list_path data/EBD/test_set.txt \
  --crop_size 256 \
  --model_type xbd_to_ebd \
  --result_saved_path results/EBD \
  --report_by_event
```

`--dataset xBD` selects the loader for the prepared image-and-mask format; the input paths select EBD. The printed summary pools confusion matrices across all samples, while event-wise reports aggregate samples by event. Event-macro overall F1 is the unweighted mean of the 12 event-level overall F1 values and should be calculated from those reports. Report three-seed means using the corresponding xBD-selected checkpoints.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Acknowledgments

This project builds on ChangeMamba and its VMamba/Mamba components. We thank the original authors for their work. Please cite the corresponding original papers when using these components, as well as the xBD and EBD dataset papers when using the data.

EBD reference: Z. Wang, C. Wu, F. Zhang, and J. Xia, “Constructing an Extensible Building Damage Dataset via Semi-Supervised Fine-Tuning Across 12 Natural Disasters,” *Journal of Remote Sensing*, vol. 5, article 0733, 2025. DOI: `10.34133/remotesensing.0733`.

## License

See [LICENSE](LICENSE). Existing third-party copyright and license notices are retained. Datasets and pretrained weights are subject to their respective terms.
