# Pomeranian Activity Classification

This project detects and classifies Pomeranian dog activities (Eat, Play, Sleep) from home surveillance video clips using a deep learning model trained on single image frames.

## Overview

The goal is to improve pet monitoring by automating behavior recognition using a frame-based classifier. The model takes a single frame from a 5-second video and classifies it into one of three activities. This approach is simple, low-compute, and suitable for use on offline or embedded devices.

## Dataset

- 5-second clips of home surveillance footage
- 3 behavior classes: Eat, Play, Sleep
- One representative frame extracted per clip
- Final dataset split (train/val): 80/20 stratified
- Image preprocessing includes resizing to 224×224 and normalization with ImageNet stats

Final class distribution in training set:

## Baseline

A simple pixel-difference motion baseline was used to detect motion between frames and guess activity:
- High difference → Play
- Low difference → Sleep
- Eat is not distinguishable via this method

Baseline accuracy: **37.01%**

## Main Model

A ResNet18 pretrained on ImageNet was fine-tuned on extracted frames:
- Final FC layer modified to 3 classes
- Optimized with Adam (lr=1e-4), trained for 10 epochs
- WeightedRandomSampler used to balance training batches

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score (macro)


