###############################
# Include library 
##############################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
###############################
# load dataset 
##############################

df = pd.read_csv("/home/ali/programing/datasets/Social_Network_Ads.csv")

print(df.shape)
print(df.isnull().sum().sum())
print(df.head())

x = df.drop("Purchased",axis=1)
y = df['Purchased']


###############################
# Train test split 
##############################

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

# print(x_train.shape)
# print(x_test.shape)

#############################
# scaling 
############################

scaler = StandardScaler()
scaler.fit(x_train)

# print("mean: ",scaler.mean_)


x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled,columns=x_test.columns)

#############################
# plot
############################
plt.scatter(x = x_train['Age'],y = x_train['EstimatedSalary'],c='black')
plt.show()
plt.scatter(x = x_train_scaled['Age'],y = x_train_scaled['EstimatedSalary'],c='red')
plt.show()

##############################
# Model
#############################
lr_scaled = LogisticRegression()
lr_scaled.fit(x_train_scaled,y_train)
LogisticRegression()
y_pred_scaled = lr_scaled.predict(x_test_scaled)

##############################
# accuracy score 
#############################

print("Scaled_accuracy :",accuracy_score(y_test,y_pred_scaled))

print("End")