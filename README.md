# Infant Cry and Emotion Classification

## Overview

The **Infant Cry and Emotion Classification** project aims to recognize and classify infant cries and detect emotions from facial expressions using machine learning techniques. The project integrates a cry classification system with facial emotion detection to provide a comprehensive analysis of infant and emotional states.

## Features

- **Infant Cry Classification**: Utilizes a Support Vector Classification (SVC) model to classify different types of infant cries.
- **Emotion Detection**: Uses the DeepFace library to analyze and identify emotions from facial expressions in images or videos.
- **Real-Time Interaction**: Features a PyQt5 interface for real-time prediction and interaction.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `DeepFace` for emotion detection
  - `OpenCV` for image processing
  - `scikit-learn` for machine learning
  - `joblib` for model serialization
  - `sounddevice` and `scipy` for audio recording (if applicable)
  - `PyQt5`for creating graphical user interface
- **Interface**: PyQt5 for creating interactive web-based applications

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aryan4585/Infant-Cry-Emotion-Classification.git
2. Navigate to the project directory:
   ```bash
   cd Infant-Cry-Emotion-Classification
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the application:
   ```bash
   app.py

## Usage

**PyQt5**: All functionalities are accessible through our PyQt5 application. This app provides a graphical user 
interface to interact with the cry classification and emotion detection features.

![Image](Infant-Cry-Emotion-Classification-/Images/GUI_Interface.png)

## Project Structure
- **Cry Dataset**: Directory for input files and test data.
- **app.py**: Main script to run the PyQt5 interface.
- **cry_recognition_model.pkl**: Saved SVC based Cry Classification model.
- **ICR_model.ipynb**: Script for cry classification using the SVC model.
- **label_encoder.pkl**: Saved file of label encoder for Cry classification model.
- **requirements.txt**: List of Python dependencies.

## Contributing
If you wish to contribute to this project:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request with a detailed description of your changes.

## Contact
For questions or feedback, please reach out to aryansharma6012@gmail.com.
