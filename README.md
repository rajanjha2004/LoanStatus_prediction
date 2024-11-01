Loan Status Prediction 

This project predicts whether a loan will be approved or not based on various factors such as gender, marital status, dependents, education, employment, property area, etc. 
The model is built using a Support Vector Machine (SVM) classifier.

Dataset

The dataset contains information about loan applicants and whether their loan was approved (Loan_Status column).
Features used include: Gender, Married, Dependents, Education, Self_Employed, Property_Area, etc.

Steps:

1. Data Preprocessing:
Null values are removed.
Categorical data is converted to numerical using .replace().

2. Model:
The dataset is split into training and testing sets using train_test_split().
An SVM classifier with a linear kernel is used to train the model.

3. Evaluation:
Accuracy is measured using the accuracy_score metric on both the training and testing datasets.

4. Results:
The model achieves a decent level of accuracy for predicting loan status.

Installation:

1. Install dependencies:
pip install pandas numpy seaborn scikit-learn
2. Run the script to train and evaluate the model.