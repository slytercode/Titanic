import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\pc\\Desktop\\PROJECTS\\Titanic\\data\\train.csv")

#clean NaN values
df.drop(columns="Cabin", inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)
df= df[df["Embarked"].notnull()]


#Divide the "Age" column#

age_bins = [0, 25, 50, float('inf')]
age_labels = ['<25', '<50', '>50']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

survived_group = (df[df["Survived"] == 1].groupby(["Pclass", "Sex", "AgeGroup"]).size())
survived_table = survived_group.unstack(level=["AgeGroup"])
totali = df.groupby(["Pclass","Sex","AgeGroup"]).size().unstack(level=["AgeGroup"])
percentages_survived = (survived_table/totali)*100
percentages_survived = percentages_survived.round(2)
print(percentages_survived)

#Create plot visualization#

ax = percentages_survived.plot(kind="bar", figsize=(15, 6), color=["skyblue", "lightgreen", "salmon"], edgecolor="black")
for container in ax.containers:
	ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=7, padding=3)
plt.title("Survival Rate by Class, Gender and Age Group")
plt.xlabel("Class and Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Age Group")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()