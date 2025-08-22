# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection

   pip install -r requirements.txt

   Place the trained model best_pneumonia_model.h5 in the project root.
   streamlit run app.py
   Open http://localhost:8501 in your browser.

Markdown

# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection
Create a virtual environment and install dependencies

Bash

pip install -r requirements.txt
Place the trained model best_pneumonia_model.h5 in the project root.

Run the Streamlit app

Bash

streamlit run app.py
Open http://localhost:8501 in your browser.

Model Architecture
Base Model: DenseNet121 pretrained on ImageNet.

Custom Head: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Training Strategy:

Stage 1: Train classifier head (base frozen)

Stage 2: Fine-tune full model

Augmentations: Horizontal flips, rotation, brightness/contrast adjustments, CLAHE, elastic/grid/optical distortions.

Markdown

# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection
Create a virtual environment and install dependencies

Bash

pip install -r requirements.txt
Place the trained model best_pneumonia_model.h5 in the project root.

Run the Streamlit app

Bash

streamlit run app.py
Open http://localhost:8501 in your browser.

Model Architecture
Base Model: DenseNet121 pretrained on ImageNet.

Custom Head: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Training Strategy:

Stage 1: Train classifier head (base frozen)

Stage 2: Fine-tune full model

Augmentations: Horizontal flips, rotation, brightness/contrast adjustments, CLAHE, elastic/grid/optical distortions.

Dataset
RSNA Pneumonia Detection Challenge

Chest X-rays labeled for pneumonia presence.

Preprocessing: Images resized to 224×224, normalized, converted to 3-channel format.

Markdown

# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection
Create a virtual environment and install dependencies

Bash

pip install -r requirements.txt
Place the trained model best_pneumonia_model.h5 in the project root.

Run the Streamlit app

Bash

streamlit run app.py
Open http://localhost:8501 in your browser.

Model Architecture
Base Model: DenseNet121 pretrained on ImageNet.

Custom Head: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Training Strategy:

Stage 1: Train classifier head (base frozen)

Stage 2: Fine-tune full model

Augmentations: Horizontal flips, rotation, brightness/contrast adjustments, CLAHE, elastic/grid/optical distortions.

Dataset
RSNA Pneumonia Detection Challenge

Chest X-rays labeled for pneumonia presence.

Preprocessing: Images resized to 224×224, normalized, converted to 3-channel format.

Grad-CAM Visualization
The model generates Grad-CAM heatmaps to visualize important regions:

heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
superimposed_img = heatmap * 0.4 + original_img

Usage
Upload X-ray images or paste them from the clipboard.

AI predicts PNEUMONIA or NORMAL.

View the confidence score and Grad-CAM heatmap side by side.

Remove images from session if needed.

Markdown

# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection
Create a virtual environment and install dependencies

Bash

pip install -r requirements.txt
Place the trained model best_pneumonia_model.h5 in the project root.

Run the Streamlit app

Bash

streamlit run app.py
Open http://localhost:8501 in your browser.

Model Architecture
Base Model: DenseNet121 pretrained on ImageNet.

Custom Head: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Training Strategy:

Stage 1: Train classifier head (base frozen)

Stage 2: Fine-tune full model

Augmentations: Horizontal flips, rotation, brightness/contrast adjustments, CLAHE, elastic/grid/optical distortions.

Dataset
RSNA Pneumonia Detection Challenge

Chest X-rays labeled for pneumonia presence.

Preprocessing: Images resized to 224×224, normalized, converted to 3-channel format.

Grad-CAM Visualization
The model generates Grad-CAM heatmaps to visualize important regions:

Python

heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
superimposed_img = heatmap * 0.4 + original_img
Grad-CAM allows better interpretation of model predictions.

Usage
Upload X-ray images or paste them from the clipboard.

AI predicts PNEUMONIA or NORMAL.

View the confidence score and Grad-CAM heatmap side by side.

Remove images from session if needed.

Dependencies
Python 3.8+

TensorFlow / Keras

Streamlit

OpenCV

pydicom

numpy, pandas, matplotlib

albumentations

streamlit_paste_button

Markdown

# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection
Create a virtual environment and install dependencies

Bash

pip install -r requirements.txt
Place the trained model best_pneumonia_model.h5 in the project root.

Run the Streamlit app

Bash

streamlit run app.py
Open http://localhost:8501 in your browser.

Model Architecture
Base Model: DenseNet121 pretrained on ImageNet.

Custom Head: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Training Strategy:

Stage 1: Train classifier head (base frozen)

Stage 2: Fine-tune full model

Augmentations: Horizontal flips, rotation, brightness/contrast adjustments, CLAHE, elastic/grid/optical distortions.

Dataset
RSNA Pneumonia Detection Challenge

Chest X-rays labeled for pneumonia presence.

Preprocessing: Images resized to 224×224, normalized, converted to 3-channel format.

Grad-CAM Visualization
The model generates Grad-CAM heatmaps to visualize important regions:

Python

heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
superimposed_img = heatmap * 0.4 + original_img
Grad-CAM allows better interpretation of model predictions.

Usage
Upload X-ray images or paste them from the clipboard.

AI predicts PNEUMONIA or NORMAL.

View the confidence score and Grad-CAM heatmap side by side.

Remove images from session if needed.

Dependencies
Python 3.8+

TensorFlow / Keras

Streamlit

OpenCV

pydicom

numpy, pandas, matplotlib

albumentations

streamlit_paste_button

Disclaimer
This is an experimental AI tool and does not replace professional medical diagnosis.

Always consult a qualified radiologist for medical decisions.

Markdown

# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection
Create a virtual environment and install dependencies

Bash

pip install -r requirements.txt
Place the trained model best_pneumonia_model.h5 in the project root.

Run the Streamlit app

Bash

streamlit run app.py
Open http://localhost:8501 in your browser.

Model Architecture
Base Model: DenseNet121 pretrained on ImageNet.

Custom Head: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Training Strategy:

Stage 1: Train classifier head (base frozen)

Stage 2: Fine-tune full model

Augmentations: Horizontal flips, rotation, brightness/contrast adjustments, CLAHE, elastic/grid/optical distortions.

Dataset
RSNA Pneumonia Detection Challenge

Chest X-rays labeled for pneumonia presence.

Preprocessing: Images resized to 224×224, normalized, converted to 3-channel format.

Grad-CAM Visualization
The model generates Grad-CAM heatmaps to visualize important regions:

Python

heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
superimposed_img = heatmap * 0.4 + original_img
Grad-CAM allows better interpretation of model predictions.

Usage
Upload X-ray images or paste them from the clipboard.

AI predicts PNEUMONIA or NORMAL.

View the confidence score and Grad-CAM heatmap side by side.

Remove images from session if needed.

Dependencies
Python 3.8+

TensorFlow / Keras

Streamlit

OpenCV

pydicom

numpy, pandas, matplotlib

albumentations

streamlit_paste_button

Disclaimer
This is an experimental AI tool and does not replace professional medical diagnosis.

Always consult a qualified radiologist for medical decisions.

License
This project is unlicensed. Use at your own discretion.

Markdown

# AI-Powered Pneumonia Detection from Chest X-Rays

A **Streamlit web application** that detects pneumonia from chest X-ray images using a **DenseNet121 Convolutional Neural Network (CNN)**. Includes **Grad-CAM visualization** to highlight areas of the X-ray that the model focuses on during prediction.

---
## Features
- **Binary Classification:** Detects **Pneumonia** vs **Normal** X-rays.
- **Grad-CAM Visualization:** Highlights regions of interest in the X-ray image.
- **Supports Multiple Formats:** Upload `.dcm`, `.jpg`, `.jpeg`, or `.png`.
- **Clipboard Support:** Paste X-ray images directly for analysis.
- **Confidence Scores:** Displays model's probability for pneumonia.

---
## Demo

![Demo Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fradiopaedia.org%2Farticles%2Fpneumonia&psig=AOvVaw0ypYgjO7SIZB2kIdrFtJPx&ust=1755985589799000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMDCkZyyn48DFQAAAAAdAAAAABAE)

---
## Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/pneumonia-detection.git](https://github.com/yourusername/pneumonia-detection.git)
   cd pneumonia-detection
Create a virtual environment and install dependencies

Bash

pip install -r requirements.txt
Place the trained model best_pneumonia_model.h5 in the project root.

Run the Streamlit app

Bash

streamlit run app.py
Open http://localhost:8501 in your browser.

Model Architecture
Base Model: DenseNet121 pretrained on ImageNet.

Custom Head: Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Training Strategy:

Stage 1: Train classifier head (base frozen)

Stage 2: Fine-tune full model

Augmentations: Horizontal flips, rotation, brightness/contrast adjustments, CLAHE, elastic/grid/optical distortions.

Dataset
RSNA Pneumonia Detection Challenge

Chest X-rays labeled for pneumonia presence.

Preprocessing: Images resized to 224×224, normalized, converted to 3-channel format.

Grad-CAM Visualization
The model generates Grad-CAM heatmaps to visualize important regions:

Python

heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
superimposed_img = heatmap * 0.4 + original_img
Grad-CAM allows better interpretation of model predictions.

Usage
Upload X-ray images or paste them from the clipboard.

AI predicts PNEUMONIA or NORMAL.

View the confidence score and Grad-CAM heatmap side by side.

Remove images from session if needed.

Dependencies
Python 3.8+

TensorFlow / Keras

Streamlit

OpenCV

pydicom

numpy, pandas, matplotlib

albumentations

streamlit_paste_button

Disclaimer
This is an experimental AI tool and does not replace professional medical diagnosis.

Always consult a qualified radiologist for medical decisions.

License
This project is unlicensed. Use at your own discretion.

Author
Developed by justbeingsniper
