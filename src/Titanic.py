import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\pc\\Desktop\\PROJECTS\\Titanic\\data\\train.csv")


df.drop(columns="Cabin", inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)
df= df[df["Embarked"].notnull()]