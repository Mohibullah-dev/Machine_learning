###########################
# Include library
###########################

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest,chi2

###########################
# Load data
###########################

df = pd.read_csv("/home/ali/programing/MachineLearning/pipelines/train (1).csv")

x = df.drop("Survived",axis=1)
y = df["Survived"]

##########################
# train test split
##########################

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)
print(x_train)

trf1 = ColumnTransformer([
    ("imput_age",SimpleImputer(),[2])],
    remainder='passthrough'
)

trf2 = ColumnTransformer([
    ("ohe_sex_embarked",OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[2,6])
],remainder= "passthrough")


trf3 = ColumnTransformer([
    ('Scale',MinMaxScaler(),slice(0,10))
],remainder="passthrough")

trf4 = SelectKBest(score_func=chi2,k=8)

trf5 = DecisionTreeClassifier()

pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5)

pipe.fit(x_train,y_train)

print(pipe.fit(x_train,y_train))