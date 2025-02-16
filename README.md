# Game Prediction: Malenia Dataset Analysis

This repository contains the code and analysis for predicting the `Phantom_Death` outcome in the Malenia dataset using two machine learning models: a Fully Connected Neural Network (FCN) and Logistic Regression.

## Dataset
The dataset used in this project is `malenia.csv`, which contains information about game sessions, including features like `Level`, `Health_Pct`, `Phantom_Build`, `Host_Build`, `Location`, `Phase`, `Phantom_Count`, and the target variable `Phantom_Death`.
Link: 'https://www.kaggle.com/datasets/jordancarlen/host-deaths-to-malenia-blade-of-miquella?resource=download  '

## Code Overview
1. **Data Preprocessing**:
   - Handling missing values.
   - Encoding categorical variables using One-Hot Encoding and Label Encoding.
   - Normalizing numerical features using StandardScaler.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizing the distribution of the target variable.
   - Detecting outliers in numerical features.

3. **Model Building**:
   - **Fully Connected Neural Network (FCN)**: A 4-layer neural network with ReLU activation functions and a sigmoid output layer.
   - **Logistic Regression**: A simple linear model for binary classification.

4. **Model Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
   - Comparison of FCN and Logistic Regression performance.

## Results
- **FCN Metrics**:
  - Accuracy: 0.5052
  - Precision: 0.5092
  - Recall: 0.5673
  - F1-Score: 0.5367
  - ROC-AUC: 0.5045
  - Confusion Matrix: [[106, 134], [106, 139]]

- **Logistic Regression Metrics**:
  - Accuracy: 0.5010
  - Precision: 0.5051
  - Recall: 0.6041
  - F1-Score: 0.5502
  - ROC-AUC: 0.5000
  - Confusion Matrix: [[95, 145], [97, 148]]

## Requirements
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, tensorflow

## How to Run
1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook or Python script.

## License
This project is licensed under the MIT License.
