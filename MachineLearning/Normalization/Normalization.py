###############################
# Include library 
##############################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
###############################
# load dataset 
##############################

df = pd.read_csv("/home/ali/programing/MachineLearning/Normalization/house_prices.csv")

print(df.isnull().sum().sum())
print(df.shape)
print(df.head(5))


x = df.drop("price_lakh",axis=1)
y = df["price_lakh"]


#############################
# Train Test split
############################

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

#############################
# Normalization
############################
Norm = MinMaxScaler()
Norm.fit(x_train)
x_train_Norm = Norm.transform(x_train)
x_test_Norm = Norm.transform(x_test)

############################
# convert to dataframe
###########################

x_train_Norm = pd.DataFrame(x_train_Norm,columns=x_train.columns)
x_test_Norm = pd.DataFrame(x_test_Norm,columns=x_test.columns)

print(x_train_Norm)

print(x_train_Norm.describe())

nom = x_train - x_train.min()
dom = x_train.max() - x_train.min()
x_train_manual = nom / dom
print(x_train_manual.head())
