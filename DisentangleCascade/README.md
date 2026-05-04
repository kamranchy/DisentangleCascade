# DisentangleCascade

**Disentangled Representation Learning with Uncertainty-Driven ROI Zoom for Skin Lesion Classification**  
ACM-BCB 2026

---

## Overview

DisentangleCascade addresses three core challenges in automated skin lesion classification:

| Challenge | Module |
|---|---|
| Entangled feature representations | Multi-head Independence-Guided Channel Attention (MICA) + HSIC regularization |
| Ambiguous lesion boundaries | Uncertainty-Driven Adaptive ROI Zoom (UDA-Zoom) via MC Dropout |
| Severe class imbalance | Weighted cross-entropy + adaptive attribute fusion |

---

## Project structure

```
DisentangleCascade/
├── main.py              # Entry point — data → train → test
├── requirements.txt
└── src/
    ├── model.py         # Full architecture (DisentangleCascade, MICA, UDA-Zoom, HSIC)
    ├── dataset.py       # ISICDataset, augmentations, stratified splits
    └── train.py         # Training loop, weighted CE loss, macro-averaged evaluation
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

Edit the paths at the top of `main.py`:

```python
DATASET_PATH = '/path/to/isic-2019/'
CSV_PATH     = DATASET_PATH + 'ISIC_2019_Training_GroundTruth.csv'
```

Then run:

```bash
python main.py
```

---

## Key implementation notes

- **Metrics** — all evaluation metrics (F1, Recall, Precision) are **macro-averaged** across all 8 classes, consistent with the paper's Tables 1, 6, and 7.
- **Loss** — weighted cross-entropy with `w_c = N_total / (C × N_c)` to counter the up to 58× class imbalance in ISIC 2019.
- **HSIC bandwidth** — `sigma` is a learnable parameter (initialized at 5.0) rather than fixed, allowing adaptive kernel optimization.
- **UDA-Zoom** — MC Dropout runs on the projected 7×7×256 feature map, not the full image, keeping inference overhead low.

---

## Results on ISIC 2019 (8 classes)

| Method | Macro F1 | Accuracy | Macro Recall | Macro Precision |
|---|---|---|---|---|
| DisentangleCascade (ours) | 0.8401 | 88.53% | 0.8192 | 0.8641 |

---

## Citation

```bibtex
@inproceedings{disentanglecascade2026,
  title     = {DisentangleCascade: Disentangled Representation Learning with Uncertainty-Driven ROI Zoom for Skin Lesion Classification},
  booktitle = {Proceedings of the 17th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM-BCB '26)},
  year      = {2026}
}
```
