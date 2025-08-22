````markdown
# AI-Powered Pneumonia Detection from Chest X-Rays

This project is a **Streamlit-based AI application** that detects pneumonia from chest X-ray images using a **DenseNet121 deep learning model**. It also provides **Grad-CAM visualizations** to highlight areas the model focuses on for prediction.

---

## Overview

The AI model classifies chest X-rays as:

- **PNEUMONIA**  
- **NORMAL**

Grad-CAM overlays highlight regions of interest on the X-ray, helping interpret model predictions.

---

## Features

- Upload or paste X-ray images (`.dcm`, `.jpg`, `.jpeg`, `.png`)  
- Real-time predictions with confidence scores  
- Grad-CAM heatmaps to visualize model focus  
- Session-based multi-image analysis  
- Download the trained model for offline use  

---

## Installation

Install required Python packages:

```bash
pip install streamlit tensorflow pydicom pillow opencv-python albumentations matplotlib streamlit-paste-button
````

Run the app:

```bash
streamlit run app.py
```

---

## Dataset

* Uses **RSNA Pneumonia Detection Challenge dataset**
* Prepare dataset:

```python
!kaggle competitions download -c rsna-pneumonia-detection-challenge
```

* Extract dataset and split into **training** and **validation** sets.

---

## Data Preparation

* Resize all images to 224×224
* Normalize pixel values to `[0, 255]`
* Convert single-channel images to 3-channel RGB for DenseNet121
* Use **Albumentations** for augmentations: flips, rotations, brightness/contrast adjustments, elastic/grid distortions, and CLAHE

---

## Model Architecture

* **Base:** DenseNet121 (pretrained on ImageNet, include\_top=False)
* **Top layers:** GlobalAveragePooling2D → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
* **Loss:** Binary Cross-Entropy
* **Optimizer:** Adam
* **Class weights** applied for imbalance between Pneumonia and Normal

---

## Training Procedure

**Stage 1: Train classifier head**

* Freeze DenseNet121 backbone
* Train top layers for 5 epochs

**Stage 2: Fine-tune full model**

* Unfreeze entire DenseNet121
* Train with `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`
* Achieved validation accuracy around 79%

---

## Grad-CAM Visualization

* Extract activations from last convolutional layer (`conv5_block16_2_conv`)
* Compute gradients w\.r.t predicted class
* Generate heatmap and overlay on original X-ray
* Helps interpret AI predictions by highlighting key regions

---

## Using the Application

1. Launch Streamlit app
2. Upload or paste X-ray images
3. View predictions and confidence scores
4. Grad-CAM overlay shows AI focus areas
5. Remove images from session as needed

---

## Testing

* Tested on 3 Pneumonia and 2 Normal validation samples
* Grad-CAM heatmaps visualize AI attention
* Confidence scores reported for each prediction

---

## Download Trained Model

After training, the best model (`best_pneumonia_model.h5`) can be downloaded:

```python
from google.colab import files
files.download('best_pneumonia_model.h5')
```

---

## Limitations

* Model trained only on RSNA dataset, may not generalize to other populations
* Cannot detect other lung conditions (e.g., TB, COVID, lung cancer)
* Confidence can be lower on edge cases
* Not clinically validated, for research and educational use only

---

## Disclaimer

> Experimental AI tool. **Not a substitute for professional medical diagnosis.** Always consult a qualified radiologist.

---

## References

* [DenseNet121 Paper](https://arxiv.org/abs/1608.06993)
* [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
* [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)



