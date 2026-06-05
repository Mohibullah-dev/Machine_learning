###############################
# Include library 
##############################
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

###############################
# load dataset 
##############################

df = pd.read_csv(
    "/home/ali/programing/MachineLearning/Encoding/train (1).csv",
    usecols=["MSZoning", "Street", "LotShape", "Neighborhood", "SaleType"]
)

print(df.head())

###############################
# One Hot Encoding
##############################

ohe = OneHotEncoder(sparse_output=False)  # makes output readable array

ohe_transform = ohe.fit_transform(
    df[["MSZoning", "Street"]]
)

print(ohe_transform)

encoded_df = pd.DataFrame(
    ohe_transform,
    columns=ohe.get_feature_names_out()
)

print(encoded_df.sample(10))
