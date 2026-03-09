import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze():
    df = pd.read_csv("cloud_workload_dataset.csv")
    
    # 1. Basic Info
    print("--- Dataset Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    # 2. Missing Values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # 3. Correlation Heatmap (Numerical columns only)
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
    # Using a clearer plot with tight layout to prevent label clipping
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Heatmap of Cloud Resource Metrics")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() 
    plt.savefig("correlation_heatmap.png")
    plt.close()
    
    # 4. Class Distribution (Scheduler_Type)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Scheduler_Type', data=df, palette='viridis')
    plt.title("Distribution of Scheduler Types")
    plt.savefig("scheduler_distribution.png")
    plt.close()
    
    print("\nEDA completed. Plots saved as 'correlation_heatmap.png' and 'scheduler_distribution.png'.")

if __name__ == "__main__":
    analyze()
