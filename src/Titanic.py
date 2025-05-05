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

#Divide the "Age" column

age_bins = [0, 25, 50, float('inf')]
age_labels = ['<25', '<50', '>50']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)