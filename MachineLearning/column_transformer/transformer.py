import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("/home/ali/programing/MachineLearning/column_transformer/covid_toy.csv")

transformer = ColumnTransformer(transformers=[
    ('tnf1', SimpleImputer(), ["fever"]),
    ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
    ('tnf3', OneHotEncoder(sparse_output=False, drop='first'), ['gender', 'city'])
], remainder='passthrough')

df_transform = transformer.fit_transform(df)

feature_names = transformer.get_feature_names_out()
df_final = pd.DataFrame(df_transform, columns=feature_names)
print(df_final.head())