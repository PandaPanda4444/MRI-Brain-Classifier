# Brain MRI Classification System

## Overview
A deep learning application that classifies brain MRI scans as either "Healthy" or "Diseased" using Convolutional Neural Networks (CNN). The system provides a user-friendly web interface for medical professionals to upload and analyze brain MRI images.

## Features
- Real-time MRI scan classification
- Bilingual interface (English/Arabic)
- High accuracy neural network model
- Confidence score for predictions
- Visual feedback with color-coded results

## Tech Stack
- Python 3.10+
- TensorFlow for deep learning
- Streamlit for web interface
- OpenCV for image processing
- Scikit-learn for data preprocessing

## Project Structure
```
brain_mri_classifier/
├── app.py              # Streamlit web application
├── model_training.py   # CNN model training script
├── requirements.txt    # Project dependencies
├── models/
│   └── model.h5       # Trained model file
├── data/              # Training data directory
│   ├── healthy/       # Healthy brain MRI scans
│   └── diseased/      # Diseased brain MRI scans
└── README.md          # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/PandaPanda4444/MRI-Brain-Classifier.git
   cd MRI-Brain-Classifier
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Model Architecture
- Input: Grayscale MRI images (128x128 pixels)
- Convolutional layers with ReLU activation
- MaxPooling layers for feature extraction
- Dropout layers for regularization
- Dense layers for classification
- Output: Binary classification (Healthy/Diseased)

## How the Model Works

### 1. Image Preprocessing
- Input MRI scans are converted to grayscale
- Images are resized to 128x128 pixels for consistency
- Pixel values are normalized to range [0,1]
- Images are expanded to include batch and channel dimensions

### 2. Model Architecture Flow
1. **Input Layer**
   - Accepts 128x128x1 grayscale images

2. **Feature Extraction**
   - Multiple convolutional layers detect features:
     * Edge detection
     * Texture patterns
     * Anatomical structures
   - MaxPooling layers reduce spatial dimensions
   - Dropout layers prevent overfitting

3. **Classification**
   - Flattened features are passed through dense layers
   - Final sigmoid activation for binary classification
   - Output: Probability of "Healthy" vs "Diseased"

### 3. Prediction Process
1. User uploads MRI scan through web interface
2. Image is preprocessed to match training format
3. Model performs forward pass to generate prediction
4. Confidence score is calculated from output probability
5. Results are displayed with color-coded feedback:
   - Green for healthy prediction
   - Red for disease detection

### 4. Model Training Details
- Trained on balanced dataset of healthy and diseased MRI scans
- Data augmentation used to improve generalization:
  * Random rotations
  * Horizontal flips
  * Brightness/contrast adjustments
- Early stopping implemented to prevent overfitting
- Model evaluated on separate validation set

## Performance
- Training Accuracy: ~95%
- Validation Accuracy: ~92%
- Early stopping to prevent overfitting

## Usage
1. Open the web interface
2. Select preferred language (English/Arabic)
3. Upload a brain MRI scan (JPG, JPEG, or PNG format)
4. Click "Classify" to get the prediction
5. View results with confidence score

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis.

## Contact
Mojtba Allam - mojtba.allam@gmail.com