# Sign Language Alphabet
Final project for McGill AI Society Introductory ML Bootcamp (Winter 2020).

Training data retrieved from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

## Project Description
This webapp, built using Flask, adopts the Keras Sequential CNN model to classify sign language hand gesture image (28x28 grayscale) as English alphabet letter. The model achieves 96.2% validation accuracy.

## Repository Structure
- `Deliverables/`: submitted MAIS 202 deliverables
- `Model/`: contains `model_pkl`, the compiled CNN model 
- `Static/`: static folder for images uploaded via the webapp
- `Templates/`: html template for the main web page
- `app.py`: webapp for sign alphabet image classification

## Running the App
1. Install all packages in requirements.txt. 
2. Run `python app.py` from the main directory. 
3. Go to http://localhost:5000 on your browser.


