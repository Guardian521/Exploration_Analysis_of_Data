import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_csv('../train.csv')


# Univariate Analysis
def univariate_analysis(df):
    # Descriptive statistics
    print("Descriptive Statistics:")
    columns=['var_81','var_139','var_12','var_53','var_66','var_26']
    # Histograms and density plots
    plt.figure(figsize=(14, 6))
    for i, column in enumerate(columns, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(df[column], shade=True)
        plt.title(f'Density Plot of {column}')
    plt.tight_layout()
    plt.show()

    # Box plots
    plt.figure(figsize=(14, 10))
    for i, column in enumerate(columns, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df[column])
        plt.title(f'Box Plot of {column}')
    plt.tight_layout()
    plt.show()


# Perform the univariate analysis
univariate_analysis(df)
