import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
dataset=pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")
dataset.head()
dataset.shape
dataset.isnull().sum()
dataset.shape
dataset=dataset.dropna()
dataset.isnull().sum()
dataset['Gender'].value_counts()
dataset['Married'].value_counts()
dataset['Dependents'].value_counts()
dataset['Education'].value_counts()
dataset['Self_Employed'].value_counts()
dataset['Property_Area'].value_counts()
dataset['Loan_Status'].value_counts()
dataset.replace({'Gender':{'Male':0, 'Female':1}, 'Married':{'Yes':0, 'No':1}, 'Dependents':{0:0,1:1,'3+':4}, 'Education':{'Graduate':0, 'Not Graduate':1}, 'Self_Employed':{'No':0, 'Yes':1}, 'Property_Area':{'Semiurban':	233,'Urban':	202,'Rural':	179}, 'Loan_Status':{'Y':1, 'N':0}}, inplace=True)
dataset.head()
x = dataset.drop(['Loan_ID', 'Loan_Status'], axis=1) # Use parentheses to call the drop method and pass a list of columns to be dropped.
y = dataset['Loan_Status']
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.1, stratify=y, random_state=2)
x.shape, x_train.shape, x_test.shape
classifyer=svm.SVC(kernel='linear')
classifyer.fit(x_train, y_train)
prediction=classifyer.predict(x_train)
accuracy=accuracy_score(prediction, y_train)
print("Accuracy of trained Dataset --> ", np.multiply(accuracy, 100), "%")
prediction=classifyer.predict(x_test)
accuracy=accuracy_score(prediction, y_test)
print("Accuracy of test Dataset --> ", np.multiply(accuracy, 100), "%")
