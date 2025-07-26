
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filepath):
	"""Load Titanic dataset from a CSV file."""
	return pd.read_csv(filepath)

def clean_data(df):
	"""Clean the Titanic dataset: drop 'Cabin', fill 'Age', drop missing 'Embarked'."""
	if 'Cabin' in df.columns:
		df = df.drop(columns="Cabin")
	df["Age"].fillna(df["Age"].mean(), inplace=True)
	df = df[df["Embarked"].notnull()]
	return df

def add_age_group(df):
	"""Add an 'AgeGroup' column to the DataFrame."""
	age_bins = [0, 25, 50, float('inf')]
	age_labels = ['<25', '<50', '>50']
	df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
	return df

def calculate_survival_rates(df):
	"""Calculate survival rates by Pclass, Sex, and AgeGroup."""
	survived_group = (df[df["Survived"] == 1].groupby(["Pclass", "Sex", "AgeGroup"]).size())
	survived_table = survived_group.unstack(level=["AgeGroup"])
	totali = df.groupby(["Pclass","Sex","AgeGroup"]).size().unstack(level=["AgeGroup"])
	percentages_survived = (survived_table/totali)*100
	percentages_survived = percentages_survived.round(2)
	return percentages_survived

def plot_survival_rates(percentages_survived):
	"""Plot the survival rates as a bar chart."""
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

def main():
	# Use relative path for portability
	data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TitanicDataset.csv")
	df = load_data(data_path)
	df = clean_data(df)
	df = add_age_group(df)
	percentages_survived = calculate_survival_rates(df)
	print(percentages_survived)
	plot_survival_rates(percentages_survived)

if __name__ == "__main__":
	main()