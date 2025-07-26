
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


# --- Business Insights & Analytical Mindset ---
def business_insights(percentages_survived):
	"""
	Print business-oriented insights based on survival rates.
	Demonstrates analytical thinking and ability to translate data into actionable recommendations.
	"""
	print("\n--- Business Insights ---")
	# Example: Find the group with the lowest survival rate
	min_rate = percentages_survived.min().min()
	min_group = percentages_survived.stack().idxmin()
	print(f"The group with the lowest survival rate is: Class {min_group[0]}, Gender {min_group[1]}, Age Group {min_group[2]} ({min_rate}%).")
	# Example: Find the group with the highest survival rate
	max_rate = percentages_survived.max().max()
	max_group = percentages_survived.stack().idxmax()
	print(f"The group with the highest survival rate is: Class {max_group[0]}, Gender {max_group[1]}, Age Group {max_group[2]} ({max_rate}%).")
	# Example: Business recommendation
	print("\nRecommendation: If you were designing safety protocols or marketing for cruise lines, focus on improving survival odds for the most vulnerable groups (e.g., lower class, older males). Use this insight to inform resource allocation and communication strategies.")

def eda_summary(df):
	"""
	Perform exploratory data analysis and print summary statistics and missing value analysis.
	Business-oriented comments are included to demonstrate analytical mindset.
	"""
	print("\n--- Exploratory Data Analysis (EDA) ---")
	print("\nData Overview:")
	print(df.head())
	print("\nSummary Statistics:")
	print(df.describe(include='all'))
	print("\nMissing Values:")
	print(df.isnull().sum())
	# Business context: Highlight potential data quality issues
	print("\nBusiness Note: Addressing missing values is crucial for reliable business insights. Incomplete data can lead to biased decisions, so we ensure all key features are clean before analysis.")

def main():
	# Use relative path for portability
	data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TitanicDataset.csv")
	df = load_data(data_path)
	# EDA and data quality check before further analysis
	eda_summary(df)
	df = clean_data(df)
	df = add_age_group(df)
	percentages_survived = calculate_survival_rates(df)
	print("\n--- Survival Rate Table ---")
	print(percentages_survived)
	plot_survival_rates(percentages_survived)
	# Integrate business insights after analysis and visualization
	business_insights(percentages_survived)

if __name__ == "__main__":
	main()