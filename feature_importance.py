from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the iris dataset
df = pd.read_csv('../train.csv', encoding='utf-8')
X, y = df.iloc[:, 2:202], df['target']

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Get feature importances
feature_names = X.columns
importances = clf.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance and get the top 10 features
top_10_features = feature_importances.sort_values(by='Importance', ascending=False).head(10)

# Print top 10 feature importances
for feature, importance in zip(top_10_features['Feature'], top_10_features['Importance']):
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# Plot top 10 feature importances
indices = np.argsort(importances)[::-1][:10]
plt.figure()
plt.title("Top 10 Feature Importances")
plt.bar(range(10), importances[indices], align="center")
plt.xticks(range(10), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, 10])
plt.show()
