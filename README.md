<<<<<<< HEAD
# Brain MRI Classifier

This project demonstrates a simple brain MRI classification system using a Convolutional Neural Network (CNN). The system classifies brain images as "Healthy" or "Diseased" (e.g., showing tumors).

## Project Structure

```
brain_mri_classifier/
├── data/
│   ├── healthy/
│   │   ├── healthy1.jpg
│   │   └── ... (more healthy images)
│   └── diseased/
│       ├── diseased1.jpg
│       └── ... (more diseased images)
├── models/
│   └── model.h5         # Trained model (generated after training)
├── model_training.py    # Script to train the model
├── app.py               # Streamlit app for image classification UI
├── requirements.txt     # Required packages
└── README.md           # Project documentation
```

## Setup Instructions

1. **Clone or Download the Project**
2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**

   - Place your brain MRI images in the `data/healthy/` and `data/diseased/` folders.

4. **Train the Model**

   Run the training script:

   ```bash
   python model_training.py
   ```

   The trained model will be saved as `models/model.h5`.

5. **Run the App**

   Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

6. **Use the App**

   - Open the provided URL (usually `http://localhost:8501`).
   - Upload an MRI image and click **Classify** to see the result.

## Disclaimer

This project is for educational and research purposes only. It is not validated for clinical use.
=======
# Brain MRI Classification System

## Overview
A deep learning application that classifies brain MRI scans as either "Healthy" or "Diseased" using Convolutional Neural Networks (CNN). The system provides a user-friendly web interface for medical professionals to upload and analyze brain MRI images.

## Features
- Real-time MRI scan classification
- User-friendly web interface
- High accuracy neural network model
- Confidence score for predictions
- Visual feedback with color-coded results

## Tech Stack
- Python 3.10+
- TensorFlow for deep learning
- Streamlit for web interface
- OpenCV for image processing
- Scikit-learn for data preprocessing

## Live Demo
[Click here to try the app](your-streamlit-url-here)

## Installation & Setup
1. Clone the repository:
bash

git clone https://github.com/PandaPanda4444/brain-mri-classifier.git
&& cd brain-mri-classifier

2. Install dependencies:
bash

pip install -r requirements.txt

3. Run the application:
bash

streamlit run app.py

## Project Structure

brain_mri_classifier/

├── app.py # Streamlit web application

├── model_training.py # CNN model training script

├── requirements.txt # Project dependencies

├── models/

│ └── model.h5 # Trained model file

└── data/ # Training data directory

├── healthy/ # Healthy brain MRI scans

└── diseased/ # Diseased brain MRI scans


## Usage
1. Open the web interface
2. Upload a brain MRI scan (JPG, JPEG, or PNG format)
3. Click "Classify" to get the prediction
4. View results with confidence score

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

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis.

## Contact
Mojtba Allam - [mojtba.allam@gmail.com]
Project Link: [https://github.com/PandaPanda4444/brain-mri-classifier](https://github.com/PandaPanda4444/brain-mri-classifier)
>>>>>>> 9a57e46030f08ec4f5005d3f902a14f8afeee821
