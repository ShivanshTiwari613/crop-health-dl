# crop-health-dl

Transfer-learning CNNs for **image-based plant-health classification**: ResNet50, InceptionV3 and EfficientNetB0 heads trained on leaf images, with class-imbalance handling and a two-phase (head → fine-tune) schedule.

Open re-implementation of the pipeline described in
**Mishra, Tiwari, Yadav & Kumar, "Google Cloud Implementation of Deep Learning Models for Crop Health Classification", IEEE ETCOM 2025** ([DOI 10.1109/ETCOM66606.2025.11436792](https://doi.org/10.1109/ETCOM66606.2025.11436792)).
The paper's drone/IoT imagery and cloud jobs are not public, so this repo trains the same architectures on the public **PlantVillage** dataset and reports numbers only from runs committed under `results/`.

## Problem

Given a photograph of a leaf, predict its health state. Two tasks:

| task | classes | what it tests |
|---|---|---|
| `coarse` | healthy / diseased | the paper's framing (disease vs not) |
| `fine` | 38 crop–disease classes | whether the features separate *which* disease |

## Data

[PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) via `tensorflow_datasets`: 54,303 RGB leaf images, 14 crops, 38 classes (26 diseases + healthy). Split deterministically 80 / 10 / 10 (train / val / test) by `tfds` slicing. Raw data is never committed; `make data` downloads ~0.8 GB.

Class balance is strongly skewed (e.g. thousands of tomato images vs a few hundred for some classes), so training uses `sklearn` *balanced* class weights (configurable).

## Method

- ImageNet-pretrained backbone (`include_top=False`) → GlobalAveragePooling → Dropout(0.3) → softmax.
- Augmentation (flips, 90° rotations, brightness, contrast) is applied to raw pixels in the training pipeline, before the backbone's own preprocessing.
- Phase 1: backbone frozen, train the head with Adam 1e-3. Phase 2: unfreeze the top 30 backbone layers (BatchNorm stays frozen) and continue at 1e-5. Early stopping on validation accuracy, best weights restored.
- Everything is one YAML in `configs/`.

## Results

Test split, 5,430 images, single seed (42), one RTX 4090. Epochs are what ran after early stopping; `train_seconds` is training wall time only.

| run_name | task | img_size | epochs | accuracy | f1_macro | f1_weighted | train_seconds |
|---|---|---|---|---|---|---|---|
| efficientnetb0_coarse | coarse | 224 | 10 | 0.9945 | 0.9931 | 0.9945 | 203 |
| efficientnetb0_fine | fine | 224 | 16 | 0.984 | 0.9825 | 0.984 | 339 |
| inceptionv3_coarse | coarse | 299 | 10 | 0.9871 | 0.9839 | 0.9871 | 333 |
| inceptionv3_fine | fine | 299 | 15 | 0.9608 | 0.9473 | 0.9611 | 433 |
| resnet50_coarse | coarse | 224 | 10 | 0.9987 | 0.9984 | 0.9987 | 255 |
| resnet50_fine | fine | 224 | 16 | 0.991 | 0.9892 | 0.991 | 401 |

- All three backbones separate healthy from diseased leaves almost perfectly on PlantVillage, so the coarse task says little about architecture choice.
- The 38-class task does separate them: ResNet50 at 224 px is best (99.1%), InceptionV3 at 299 px is worst (96.1%) despite the larger input, with confusions concentrated in the two corn foliar diseases (see `results/inceptionv3_fine/classification_report.txt`).
- Balanced class weights were on for every run. The rarest class, potato healthy (152 images, 10 in the test split), scores F1 1.00 with ResNet50 and EfficientNet but only 0.76 with InceptionV3, so class weighting alone did not rescue the weakest backbone.

Per-run confusion matrices, per-class reports and training curves are under `results/<run>/`.

## How to run

```bash
make setup                                   # uv venv + deps (Python 3.11)
make data                                    # download PlantVillage (~0.8 GB)
make train CONFIG=configs/resnet50_coarse.yaml
make table                                   # results/RESULTS.md
```

Other configs: `inceptionv3_{coarse,fine}`, `efficientnetb0_{coarse,fine}`, `resnet50_fine`, `smoke` (2 batches, sanity only).

```bash
make test    # dataset-free smoke tests: label mapping, each backbone builds and trains one batch
make lint    # ruff + black
```

Apple Silicon: uncomment `tensorflow-metal` in `requirements.txt` for GPU acceleration.

To train on a rented GPU instead, `podenv/` drives a RunPod pod end to end (`python3 podenv/pod.py run all`); see `podenv/README.md`.

## Limitations

- **Not the paper's data.** The paper used drone/IoT field imagery with three classes (healthy / stressed / diseased). PlantVillage is lab-style single-leaf photos and has no "stressed" (abiotic) class, so only the healthy/diseased boundary is comparable. Field-vs-lab domain shift is the main reason PlantVillage accuracies are optimistic.
- **Leakage risk in PlantVillage.** Images of the same leaf/plant can appear across splits; published results on this dataset are known to be inflated. Numbers here are for architecture comparison, not deployment claims.
- **No GCP path.** The paper's Cloud Storage + AI Platform deployment is out of scope; the code is plain TF/Keras and runs locally.
- Single seed per run; no confidence intervals yet.

## Repository layout

```
data/download.py     tfds download only
src/data.py          tfds pipeline, fine/coarse labels, augmentation, class weights
src/models.py        backbones, head, staged unfreezing
src/train.py         python -m src.train --config configs/X.yaml
src/evaluate.py      metrics.json, confusion matrix, per-class report
scripts/results_table.py   results/RESULTS.md
configs/*.yaml       one file per run
results/<run>/       config, history, metrics, plots (weights git-ignored)
tests/               pytest smoke tests
```

## Citation

```bibtex
@inproceedings{mishra2025cropcloud,
  title     = {Google Cloud Implementation of Deep Learning Models for Crop Health Classification},
  author    = {Mishra, Tushar and Tiwari, Shivansh and Yadav, Sarthak and Kumar, Vinod},
  booktitle = {2025 IEEE International Conference on Emerging Trends in Computing and Communication (ETCOM)},
  year      = {2025},
  pages     = {1--6},
  publisher = {IEEE},
  doi       = {10.1109/ETCOM66606.2025.11436792}
}
```

MIT licence.
