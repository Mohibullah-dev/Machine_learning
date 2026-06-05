###############################
# Include library 
##############################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
###############################
# load dataset 
##############################

df = pd.read_csv("/home/ali/programing/MachineLearning/Encoding/customer.csv",usecols=["review","education","purchased"])
print(df.head(5))

##############################
# Train Test split
#############################

x = df.drop('purchased',axis=1)
y = df['purchased']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

############################
# ordinal encoding
###########################

oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])

oe.fit(x_train)

x_train = oe.transform(x_train)
x_test = oe.transform(x_test)

############################
# convert to DataFrame
############################

x_train = pd.DataFrame(x_train,columns=x.columns)
x_test = pd.DataFrame(x_test,columns=x.columns)

print(x_train)

############################
# label encoding
###########################

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


############################
# convert to DataFrame
############################

y_train = pd.DataFrame(y_train, columns=['purchased'])
y_test = pd.DataFrame(y_test, columns=['purchased'])

print(y_train)