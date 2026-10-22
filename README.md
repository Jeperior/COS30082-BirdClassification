# COS30082 Assignment 1 — Bird Species Classification

## 1. Project Overview

This project was developed for **COS30082 Assignment 1: Bird Species Classification**.

The objective is to develop a deep learning model capable of classifying bird images into 200 species using the Caltech-UCSD Birds-200-2011 (CUB-200) dataset.

The project uses transfer learning with a pretrained **ConvNeXt-Tiny** model. Several training experiments were conducted to investigate how fine-tuning, data augmentation, and regularisation affect classification performance.

The final model was also deployed through a Streamlit web application, allowing users to upload bird images and receive classification predictions.

## 2. Dataset

The project uses the CUB-200 bird dataset, which contains images belonging to 200 bird species.

- **Number of classes:** 200
- **Training images:** 4,829
- **Task:** Multi-class image classification

The dataset directory contains the training and test images, together with their annotation files.

## 3. Model Architecture

The project uses **ConvNeXt-Tiny pretrained on ImageNet-1K**.

The original classification head was replaced with a new linear layer containing 200 output units, corresponding to the 200 bird species.

Transfer learning was used to leverage pretrained visual features. Different experiments then investigated training the classification head and fine-tuning selected feature-extraction layers.

The training process uses Cross-Entropy Loss and the AdamW optimiser.

## 4. Experiments

The following experiments were conducted:

1. **Baseline:** Trained the new classification head while freezing the pretrained feature extractor.
2. **Fine-tuning:** Unfroze selected feature layers to adapt the pretrained model to bird classification.
3. **Data augmentation:** Investigated training with additional image transformations.
4. **Regularisation:** Evaluated the effect of additional regularisation techniques.
5. **Label smoothing:** Investigated label smoothing during training.
6. **Final full-data training:** Fine-tuned the selected augmented checkpoint using the full training dataset.

The experiments were evaluated to compare classification performance and determine the final model configuration.

## 5. Results

The final full-data model achieved the following results on the separate test set:

| Evaluation Metric | Result |
|---|---:|
| Top-1 Accuracy | 80.23% |
| Average Per-Class Accuracy | 79.81% |

Top-1 accuracy measures the proportion of test images for which the highest-ranked prediction matches the correct class.

Average per-class accuracy measures the classification accuracy for each species and then averages those class-level accuracies.

The results indicate that the final model can distinguish between many of the bird species in the dataset, although further improvements may be possible.

## 6. Project Structure

```text
COS30082-BirdClassification/
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── data/
│   ├── Train/
│   ├── Test/
│   ├── train.txt
│   └── test.txt
│
├── models/
│   ├── best_model.pth
│   ├── finetuned_model.pth
│   ├── finetuned_augmented_model.pth
│   ├── finetuned_regularized_model.pth
│   ├── finetuned_label_smoothing_model.pth
│   └── final_full_data_model.pth
│
├── models_fp16/
│   └── Compressed model checkpoints
│
├── results/
│   ├── misclassified_images/
│   ├── model_comparison.csv
│   └── Evaluation results and visualisations
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── transforms.py
│   ├── train.py
│   ├── train_finetune.py
│   ├── train_augmented.py
│   ├── train_regularized.py
│   ├── train_label_smoothing.py
│   ├── train_final.py
│   ├── evaluate.py
│   ├── evaluate_all_models.py
│   ├── test_dataset.py
│   ├── test_dataloader.py
│   ├── test_model.py
│   ├── test_fp16.py
│   ├── inspect_mistakes.py
│   └── compress_models.py
│
├── streamlit_app.py
├── create_class_mapping.py
├── class_names.json
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: This tree summarises the main project files. Optional experiment files and generated results may vary.*

## 7. Installation and Usage

### Requirements

- Python
- pip
- Required Python packages listed in `requirements.txt`

### Install dependencies

Open a terminal in the project root directory and run:

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

From the project root directory, run:

```bash
streamlit run streamlit_app.py
```

The application should open in a browser. Users can upload a bird image and view the model's predicted species.

### Training scripts

The training scripts are located in the `src/` directory.

The baseline training script can be run from the project root using:

```bash
python -m src.train
```

The final full-data training script can be run using:

```bash
python -m src.train_final
```

Training requires the dataset and the appropriate pretrained or previously saved checkpoints. Final full-data training starts from the augmented fine-tuned checkpoint.

**Note:** Training the final model again is not necessary to use the already trained checkpoint in the application.

## 8. Deployment

The Streamlit application is available at:

https://cos30082-birdclassification-4affowupxonnvxwrxlcmcf.streamlit.app/

Users can upload a bird image to obtain a predicted species and, where supported by the interface, view additional prediction information.

## 9. Conclusion

This project demonstrates the application of transfer learning to bird species classification. Through several experiments involving fine-tuning, augmentation, and regularisation, the final model achieved 80.23% Top-1 accuracy and 79.81% average per-class accuracy on the test set.

The deployed application demonstrates how a trained deep learning model can be integrated into a practical image classification interface.