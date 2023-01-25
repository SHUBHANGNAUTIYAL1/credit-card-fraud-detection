import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data=pd.read_csv('creditcard.csv')
#print(credit_card_data.head())
#credit_card_data.info()
#print(credit_card_data.isnull().sum())
#distribution of legit transaction and fraudulent transaction
print(credit_card_data['Class'].value_counts())

#seperating the data for analysis
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]

print(legit.shape)
print(fraud.shape)

#stastical measures of the data
legit.Amount.describe()
fraud.Amount.describe()

#compare the values for both transaction
credit_card_data.groupby('Class').mean()


legit_sample=legit.sample(n=492)
new_dataset=pd.concat([legit_sample,fraud],axis=0)
#print(new_dataset.head())

new_dataset['Class'].value_counts()
#print(new_dataset.groupby('Class').mean())

#Spliting the data into Features & Targets

X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

#split data into training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#MODEL TRAINING
model=LogisticRegression()
model.fit(X_train,Y_train)
#Model Evaluation

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('ACCURACY ON TRAINING DATA',training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('ACCURACY ON TESTING DATA',test_data_accuracy)
