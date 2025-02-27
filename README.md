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
   git clone https://github.com/PandaPanda4444/brain-mri-classifier.git
   cd brain-mri-classifier
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
Project Link: [https://github.com/PandaPanda4444/brain-mri-classifier](https://github.com/PandaPanda4444/brain-mri-classifier)
