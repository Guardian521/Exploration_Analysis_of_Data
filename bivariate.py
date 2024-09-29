import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_csv('../train.csv')


# Bivariate Analysis
def bivariate_analysis(df):

    # Correlation matrix
    correlation_matrix = df.iloc[:, 2:201].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)
    correlation_matrix.to_csv('corr.csv')



# Perform the bivariate analysis
bivariate_analysis(df)
