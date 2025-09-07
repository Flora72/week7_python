# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    iris_raw = load_iris()
    iris_df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

    print("Dataset loaded successfully")
    print(iris_df.head())
    print(iris_df.info())
    print("Missing values:\n", iris_df.isnull().sum())

except Exception as e:
    print("Error loading dataset:", e)

# Task 2: Basic Data Analysis
print("\n📈 Basic Statistics:")
print(iris_df.describe())

print("\n Grouped Means by Species:")
grouped = iris_df.groupby('species').mean()
print(grouped)

# Task 3: Data Visualization
# Line Chart (simulated time series)
plt.figure(figsize=(10, 5))
plt.plot(iris_df.index, iris_df['sepal length (cm)'], label='Sepal Length')
plt.title('Simulated Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.savefig("line_chart.png")
plt.close()

# Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.savefig("bar_chart.png")
plt.close()

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(iris_df['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.savefig("histogram.png")
plt.close()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.savefig("scatter_plot.png")
plt.close()

print("\n Visualizations saved as PNG files.")
