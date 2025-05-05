import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\pc\\Desktop\\PROJECTS\\Titanic\\data\\train.csv")
print(df.head)
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe)
print(df.isnull().sum())
