import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\pc\\Desktop\\PROJECTS\\Titanic\\data\\train.csv")

#clean NaN values
df.drop(columns="Cabin", inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)
df= df[df["Embarked"].notnull()]
correlation = df.corr(numeric_only=True)
print (correlation["Survived"].sort_values(ascending=False))

# "Pclass" is a non-human attribute with strong correlation to survival

print(df["Pclass"].value_counts())
print(df.groupby("Pclass")["Age"].mean())
print(df.groupby("Survived")["Sex"].count())
print(df.groupby("Sex")["Survived"].mean())
print(df.groupby(["Pclass", "Sex", "Survived"]).size())
print(df.groupby(["Pclass", "Sex"])["Survived"].mean() * 100)