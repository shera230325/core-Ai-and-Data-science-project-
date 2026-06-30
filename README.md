# Deep Learning-Based Soil Nutrient Profiling

CNN-based computer vision pipeline for classifying soil nutrient deficiencies from field photographs, built for fully offline use on basic Android phones.

🏆 **1st Place — National Innovation & Technology Competition (600+ teams), 2024–25**

## Overview

This project was built as part of a 5-member team (I served as team lead) to address a real cost barrier: farmers in our region pay roughly ₹500 for a single lab soil test, which is unaffordable for many. We built a CNN that classifies soil nutrient deficiency from a photograph alone, exported to run completely offline on a basic Android device — no internet connection required, since that's the reality in many of the villages this was built for.

The original training run (2024) was performed on a dataset of 5,000+ field images, personally collected and labelled by our team, achieving **91.4% classification accuracy** on a held-out test set.

## A note on this repository

The original training environment and full dataset were lost. This repository contains the architecture and pipeline rebuilt from memory, matching the original design, and retrained on a smaller available sample of the original images while the full dataset is recovered. The code here is fully functional — `train.py` actually trains, `predict.py` actually predicts — unlike a placeholder template. Results in this repo reflect the current (smaller-sample) retraining run, not the original 91.4% competition result, until the full dataset is restored and retraining is complete.

## Architecture

3 convolutional blocks (32 → 64 → 128 filters), built from scratch with no pretrained backbone, chosen to keep the model small enough for offline mobile deployment.

![Architecture](images/architecture.png)

## Project structure

```
├── data_loader.py     # Loads and augments images from class folders
├── model.py            # CNN architecture
├── train.py             # Training loop, saves model + accuracy plot
├── predict.py          # Run inference on a single image
├── dataset/             # One folder per class (not included, see below)
├── models/              # Saved trained model
├── results/             # Accuracy plots
└── images/               # Architecture diagram, sample predictions
```

## Classes

- Nitrogen deficient
- Phosphorus deficient
- Potassium deficient
- Healthy

## How to run

```bash
pip install -r requirements.txt

# Organize your images as:
# dataset/nitrogen_deficient/, dataset/phosphorus_deficient/,
# dataset/potassium_deficient/, dataset/healthy/

python train.py
python predict.py path/to/your/image.jpg
```

## Technologies

Python, TensorFlow, OpenCV, NumPy, Matplotlib

## Dataset

Field images of soil collected and labelled manually across our district. Not publicly released due to ongoing use; a small sample is included for demonstration as the original full dataset is recovered.

## Key design decision: offline-first

Most teams at the competition optimised purely for accuracy. We optimised for the person who would actually use the model — a farmer with no internet connection who cannot afford to be wrong about his soil. The model was exported and tested for inference on a basic Android device with no network dependency.

## Future improvements

- Restore and retrain on the full 5,000+ image dataset
- TFLite conversion benchmarking for low-end devices
- Grad-CAM visualisation for interpretability
- Expand class granularity (e.g. severity levels per deficiency)