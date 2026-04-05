# Insider Threat Detection Using Unsupervised Behavioural Analysis

 **Dataset:** CERT Insider Threat Dataset r4.2

---

## What This Project Does

Most insider threat detection research assumes you already know who the bad actors are — you train on labeled data, the model learns what malicious looks like, and off you go. That works fine in a research setting, but in the real world, you almost never have those labels. Confirmed insiders only get identified *after* an investigation, which is too late for early warning.

This project takes a different approach: **learn what normal looks like for each individual user, then flag the days that don't fit that pattern** — no labels required.

We convert each user's daily activity into a 96×7 image (96 fifteen-minute time slots × 7 activity types), build a personal behavioural baseline per user, and score days by how much they deviate from that baseline. Five unsupervised models are trained and evaluated against three supervised baselines on the CERT r4.2 dataset.

The headline result: **One-Class SVM achieves ROC-AUC 0.8100 without any labels**, sitting only 0.09 behind the best supervised model (CNN Classifier, 0.9033).

---

## Project Structure

```
insider_threat_analysis.ipynb   ← Full pipeline, one notebook
README.md                       ← This file
```

The entire pipeline lives in a single notebook. It is structured as sequential cells that run top to bottom — no separate scripts or modules needed.

---

## Dataset

This project uses the **CERT Insider Threat Dataset r4.2**, released by Carnegie Mellon University's Software Engineering Institute. It is **not included in this repository** — you need to download it separately.

**Download:** https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

Once downloaded, place the following files in a folder and update the `data_folder` path at the top of the notebook:

```
r4.2/
├── logon.csv
├── email.csv
├── device.csv
├── file.csv
├── http.csv          ← rename http_erased.csv to this, or update the path in the notebook, we have erased few rows in this due to spce constraints
└── answers/
    └── insiders.csv
```

The dataset contains 5.3 million log entries for 1,000 virtual users over 17 months, with 70 confirmed insiders across three scenarios. Malicious activity accounts for just 0.41% of all samples — which is exactly what makes this problem hard.

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras xgboost tqdm scipy
```

Tested with:
- Python 3.9+
- TensorFlow 2.12+
- scikit-learn 1.3+
- XGBoost 1.7+

A GPU is strongly recommended for the deep learning models. The Conv Autoencoder and VAE each take roughly 30–45 minutes to train on CPU; on a mid-range GPU this drops to under 10 minutes.

---

## How to Run

1. Clone the repo and open the notebook
2. Update `data_folder` in **Cell 4** to point to wherever you saved the CERT r4.2 files
3. Run all cells top to bottom

That's it. Each section is clearly labelled and the cells are designed to run sequentially without needing to re-run earlier ones.

---

## Pipeline Walkthrough

Here is what each section of the notebook does:

### 1. Load All Datasets (Cell 4)
Reads the five raw CSV files — logon, email, device, file, and http — into separate DataFrames. Outputs the shape of each to confirm they loaded correctly.

```
logon:  (854859, 5)
email:  (2629979, 11)
device: (405380, 5)
file:   (445581, 6)
http:   (1048575, 5)
```

### 2. Merge and Clean Activity Logs (Cell 6)
Combines all five logs into a single unified DataFrame with three columns: `user`, `date`, `activity`. Activity labels are standardised — logon events keep their Logon/Logoff labels, everything else is collapsed to one label per type (device, email, file, http). Timestamps are converted into 15-minute time bins (0–95), where bin 0 = midnight and bin 95 = 11:45pm.

### 3. Build 96×7 Activity Matrices (Cell 8)
For every (user, day) pair, a 96×7 matrix is constructed. Rows are time bins, columns are the 7 activity types: login, logout, email, file, device, http, unknown. Each cell holds the count of that activity in that time slot. A `log1p` transform is applied immediately to compress extreme outlier counts — a user with 500 USB events in one hour would otherwise dominate normalisation.

This produces the main data structure: a tensor of shape **(330452, 96, 7)** — one 96×7 image per (user, day) pair.

### 4. MinMax Normalisation (Cell 10)
Each sample is independently scaled to [0, 1] using its own min and max values. This is the normalisation used by the **supervised models**. Because it is per-sample, test data does not leak any information from training data.

Output shape: **(330452, 96, 7, 1)** — the extra channel dimension is added for compatibility with Conv2D layers.

### 5. Per-User Z-Score Normalisation (Cell 12)
This is the key normalisation for the **unsupervised models**. Rather than scaling globally, each user's own mean and standard deviation (computed across all their days) is used to score each day. The result tells you not "is this user active?" but "is this user doing something unusual *for them*?".

Z-scores are clipped to [−5, +5], then shifted globally as `(z + 5) / 10`. This maps:
- z = 0 (perfectly normal day) → 0.5
- z = +4 (very unusual day) → 0.9
- z = −3 (very inactive day) → 0.2

**Why a global shift and not per-sample rescaling?** Per-sample rescaling would make an anomalous day with z = [−2, +4] look identical to a quiet day with z = [−0.1, +0.3] after rescaling both to [0, 1]. The global shift preserves the magnitude of the deviation, which is the actual signal.

### 6. Labels (Cell 14)
Labels are loaded from `answers/insiders.csv`. Each insider's start–end window is expanded day by day, and every (user, day) pair within that window is labelled malicious (1). Everything else is benign (0).

```
Total samples  : 330452
Malicious      : 1364  (0.41%)
Benign         : 329088
```

### 7. Train/Test Split (Cell 16)
An 80/20 stratified split with `random_state=42`. Both normalisation variants (MinMax and z-score) share identical train/test indices so all models are evaluated on exactly the same test data. The unsupervised training set contains **benign days only** — labels are not used at any point during unsupervised training.

```
Train : 264361  (1091 malicious)
Test  : 66091   (273 malicious)
Unsupervised training (benign-only): 263270 samples
```

---

## Models

### Unsupervised Models (trained on benign days only, no labels used)

**Convolutional Autoencoder (Cell 18)**  
Two Conv2D + BatchNorm + MaxPooling encoder blocks compress the 96×7×1 input into an 8-filter bottleneck. The decoder mirrors this with Conv2DTranspose + UpSampling layers. Anomaly score = per-pixel mean squared reconstruction error. Days that the model struggles to reconstruct are flagged as unusual.

**Convolutional VAE with KL Annealing (Cell 19)**  
The autoencoder is extended with a probabilistic latent space (dim = 32) using the reparametrisation trick. Without intervention, training collapsed within two epochs — the encoder learned to ignore the latent space entirely (posterior collapse). This was fixed with a KL annealing schedule:
- Epochs 1–10: KL weight = 0 (learn reconstruction only)
- Epochs 11–30: KL weight linearly ramps from 0 → 1
- Epochs 31+: KL weight = 1 (full VAE objective)

This keeps the latent space meaningful throughout training.

**LSTM-VAE with GRU Encoders (Cell 20)**  
Same VAE architecture but treating the 96×7 matrix as a temporal sequence rather than a spatial image. The input is first downsampled from 96×7 to 48×7 (summing adjacent 15-minute bins into 30-minute bins) to reduce training time. GRU layers replace LSTM units (~30% fewer parameters, comparable performance). Architecture: GRU(64) → GRU(32) encoder; RepeatVector(48) → GRU(32) → GRU(64) → TimeDistributed(Dense) decoder. Same KL annealing schedule as the Conv VAE.

*Note: A 96×7 version was also run once (ROC-AUC 0.8086) but was too slow for iterative experimentation. Results for both variants are reported.*

**Isolation Forest (Cell 21)**  
200-tree ensemble trained on the flattened 672-dimensional z-score vectors. Anomaly score = negated `score_samples` output (so higher = more anomalous). No labels used.

**One-Class SVM (Cell 22)**  
RBF-kernel SVM with ν=0.05 and `gamma='scale'`. Trained on a random 10,000-sample subset of the benign training data (the full dataset is too slow for a kernel method). Despite this, it achieves the best unsupervised AUC.

---

### Supervised Baselines (trained with labels, for comparison only)

**Random Forest (Cell 24)**  
300 trees with `class_weight='balanced'` to handle the extreme class imbalance.

**XGBoost (Cell 25)**  
300 estimators with `scale_pos_weight ≈ 800` (ratio of benign to malicious samples), learning rate 0.05, max_depth 6.

**CNN Classifier (Cell 26)**  
Conv(32) → BatchNorm → MaxPool → Conv(64) → BatchNorm → GlobalAvgPool → Dense(64) → Dropout(0.4) → Sigmoid. Class weights passed to `fit()`. Treats each user-day as a small greyscale image.

---

## Results

| Model | Track | ROC-AUC | Avg Precision | F1 |
|---|---|---|---|---|
| Conv Autoencoder | Unsupervised | 0.7532 | 0.0103 | — |
| VAE | Unsupervised | 0.8048 | 0.0129 | — |
| LSTM VAE (48×7) | Unsupervised | 0.7877 | 0.0104 | — |
| LSTM VAE (96×7) | Unsupervised | 0.8086 | 0.0108 | — |
| Isolation Forest | Unsupervised | 0.7294 | 0.0074 | — |
| **One-Class SVM** | **Unsupervised** | **0.8100** | **0.0115** | **—** |
| Random Forest | Supervised | 0.8922 | 0.0437 | 0.0000 |
| XGBoost | Supervised | 0.8777 | 0.0556 | 0.0726 |
| **CNN Classifier** | **Supervised** | **0.9033** | **0.0622** | **0.0243** |

**Key takeaways:**
- One-Class SVM is the best unsupervised model despite being a classical algorithm trained on only 10,000 samples. In z-score space, normal behaviour clusters tightly, which the RBF kernel is naturally good at bounding.
- The Conv VAE (0.8048) and full-resolution LSTM-VAE (0.8086) perform nearly identically, suggesting the 96×7 spatial layout already captures enough temporal structure that sequential modelling via GRU adds nothing.
- The downsampled LSTM-VAE (48×7, 0.7877) performs worse, confirming the performance gap is about temporal resolution, not the architecture.
- Random Forest achieves F1 = 0.0000 despite AUC 0.8922 — at its optimal threshold, it predicts no positives at all, a consequence of how extreme the class imbalance is.
- The supervised-to-unsupervised gap is **0.0933 AUC** — a concrete, rarely-reported figure for the cost of going label-free.

---

## Known Limitations

- The CERT r4.2 dataset is synthetic. Real insider behaviour is messier, less predictable, and almost certainly harder to detect.
- The unsupervised models have no notion of false positive rate at a deployment threshold. At AUC 0.8100, the practical precision at any usable recall level is still very low — flagging dozens of benign users daily would be operationally unacceptable.
- The One-Class SVM is trained on a 10,000-sample subset due to kernel SVM scaling constraints. A full-dataset training might improve results further.
- Per-user z-score baselines are static — computed once at the start. In a real deployment, baselines would need to update continuously as users' normal patterns evolve.

---

## Acknowledgements

CERT Insider Threat Dataset: Lindauer, B. (2020). Insider Threat Test Dataset. Carnegie Mellon University. https://doi.org/10.1184/R1/12841247.v1
