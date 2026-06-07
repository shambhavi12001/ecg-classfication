
# ECG Heartbeat Classification

Efficient ECG heartbeat classification for wearables, built on top of the
transferable residual-CNN representation of Kachuee et al. (2018). The project
reproduces the five-class (AAMI EC57) arrhythmia classifier on MIT–BIH and then
compares four efficiency levers — **input-length reduction**, a
**MobileNet-style depthwise-separable architecture**, **post-training
quantization**, and **magnitude pruning** — reporting parameters, on-disk size,
MACs, latency, and accuracy for each.

## Repository structure

```
.
├── ecg_efficiency.ipynb     
└── README.md
```

## The notebook: `ecg_efficiency.ipynb`

Runs end-to-end and produces every figure/table in the report:

1. **Data loading** — reads the pre-segmented MIT–BIH CSVs (187-sample beats).
   If the CSVs are absent it falls back to clearly-labelled synthetic data so
   the notebook still runs; use the real data for reported numbers.
2. **Experiment A — input-length sweep** — trains the baseline residual CNN at
   input lengths {187, 140, 96, 64, 48, 32} and records params, MACs, on-disk
   size, latency, and accuracy.
3. **Experiment B — better levers** — a depthwise-separable `MobileNet-1D`
   architecture, TFLite dynamic + int8 quantization (with quantized-model
   accuracy check), and magnitude pruning measured by gzip size.
4. **Pareto comparison** — accuracy-vs-model-size plot across all variants.

## Data

Download the **ECG Heartbeat Categorization Dataset** (processed MIT–BIH and
PTB) from Kaggle and place `mitbih_train.csv` and `mitbih_test.csv` in the
folder pointed to by `DATA_DIR` in the notebook's config cell. Source:
PhysioNet MIT–BIH Arrhythmia Database (Moody & Mark, 2001; Goldberger et al.,
2000).

## Requirements

```bash
pip install tensorflow numpy pandas matplotlib
```

Trains in minutes on a GPU (developed on an NVIDIA RTX 2080 Ti). For final
numbers, set `EPOCHS≈30` and `SUBSAMPLE_TRAIN=None` in the config cell.

## How to run

```bash
jupyter notebook ecg_efficiency.ipynb   # run all cells top to bottom
```

To regenerate the report figures, save the notebook's example-beats figure as
`report/figures/example_beats.png` and the accuracy-vs-length plot as
`report/figures/accuracy_vs_length.png`.

## Results (real MIT–BIH)

| Model | Params | Size | Accuracy |
|---|---|---|---|
| Residual CNN (FP32, L=187) | 58,053 | 800.4 KB | 0.9863 |
| Residual CNN (int8)        | 58,053 | 85.9 KB  | ≈0.985 |
| MobileNet-1D (FP32)        | 7,493  | —        | 0.9758 |
| Residual CNN (L=32)        | 52,933 | 740.4 KB | 0.9774 |


