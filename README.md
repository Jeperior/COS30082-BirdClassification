# COS30082 Assignment 1 — Bird Species Classification

## Overview

This project is developed for **COS30082 Assignment 1: Bird Species Classification**.

The task is to develop a machine learning model for **multi-class bird species classification** using the **Caltech-UCSD Birds 200 (CUB-200)** dataset.

The dataset contains images from **200 bird species** and 4,829 training images.

---

# Project Structure

```text
COS30082-BirdClassification/
│
├── app/
│   └── # Gradio application
│
├── data/
│   ├── Train/
│   ├── Test/
│   ├── train.txt
│   └── test.txt
│
├── models/
│   └── # Saved trained models
│
├── notebooks/
│   └── # Optional experiments
│
├── results/
│   └── # Training and evaluation results
│
├── src/
│   ├── dataset.py
│   ├── test_dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── .gitignore
├── README.md
└── requirements.txt