# English Handwriting Recognition (62 Classes)

This repository contains a Deep Learning pipeline for recognizing 62 different handwriting characters, including uppercase letters (A-Z), lowercase letters (a-z), and digits (0-9). The model is based on the **EfficientNet_V2_S** architecture and achieves approximately **89-90% accuracy**.

## 📊 Project Structure
```text
├── dataset/                # Raw data (Img folder + english.csv)
├── models/                 # Trained model weights (.pth)
│   └── final_handwriting_model.pth
├── notebooks/              # Research and experimentation
│   └── english_recog.ipynb
├── src/                    # Source code
│   ├── test_letters/       # Sample images for testing
│   │   └── k.png
│   ├── inference.py        # Prediction script
│   └── train.py            # Training script
├── classes.npy             # Class mapping for inference
├── label_encoder.pkl       # Serialized label encoder (legacy)
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

## 🚀 Performance & Results
- **Model:** EfficientNet_V2_S (Transfer Learning)
- **Accuracy:** ~89.15%
- **Loss:** CrossEntropy with Label Smoothing (0.1)
- **Optimizer:** AdamW with OneCycleLR Scheduler

### Key Insights from Analysis:
*   **High Precision:** The model is exceptionally accurate at identifying distinct shapes like `A, B, 3, 4, 8, y, g`.
*   **Scale Ambiguity:** Most errors occur in "twin shapes" where the only difference is scale (e.g., `0` vs `O` vs `o` or `S` vs `s`). Because characters are analyzed in isolation without word context, the model reaches a mathematical limit on these specific classes.
*   **Vertical Strokes:** Occasional confusion exists between `1`, `I`, and `l`, which is a common challenge in isolated character OCR.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/english-handwriting-recog.git
   cd english-handwriting-recog
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset:**
   Ensure your `dataset/` folder contains the `Img/` directory and `english.csv` from the [English Handwriting Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset).

## 💻 Usage

### Training
To retrain the model from scratch using the script version:
```bash
cd src
python train.py
```
*Note: This script will automatically save the best model weights to `../models/` and class names to `../classes.npy`.*

### Inference (Prediction)
To test the model on a single image:
```bash
cd src
python inference.py test_letters/k.png
```
You can also provide a path to any other image:
```bash
python inference.py path/to/your/image.png
```

## 🧠 Technical Highlights
*   **RAM Optimization:** The training script resizes images to 224x224 immediately during the loading phase, allowing the entire dataset to reside in RAM for ultra-fast epoch times.
*   **Advanced Augmentation:** We utilize `Grayscale` conversion (to focus on stroke patterns), `RandomAdjustSharpness`, and `RandomPerspective` to simulate variations in ink, paper, and writing angles.
*   **OneCycleLR:** We implement a dynamic learning rate scheduler that "warms up" the training to prevent local minima and "cools down" to fine-tune weights.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙌 Acknowledgments
- Dataset provided by dhruvildave on Kaggle.
- Pre-trained weights from the PyTorch Torchvision library.