"""
This file contains the implementation of backend methods for titanic datasets

"""

# import os
# import io
# import numpy as np
# import base64
# import matplotlib.pyplot as plt
# from Backend.load_datasets import load_Dataset

# df = load_Dataset()

# def calculate_survival_rate_by_class():
#     """Calculates survival rates by class."""
#     survival_rates = df.groupby("Pclass")["Survived"].mean() * 100
#     return f"Survival rate by class: {survival_rates.to_dict()}"

# def generate_histogram(column, title, xlabel):
#     """Generates a histogram and saves it as an image."""
#     if column not in df.columns:
#         return {"type": "error", "message": f"Column '{column}' not found in the dataset."}
#     fig, ax = plt.subplots(figsize=(8, 5))
#     plt.figure(figsize=(8, 5))
#     plt.hist(df[column].dropna(), bins=20, edgecolor='black')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel("Number of Passengers")
#     filename = f"{column}_histogram.png"
#     plt.savefig(filename)
#     plt.close()

#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
#     encoded = base64.b64encode(buf.read()).decode('utf-8')
#     return {
#         "status": "success",
#         "image_path": filename,
#         "image_encoded": encoded
#     }

# def get_embarked_counts():
#     """Returns counts of passengers embarked from each port."""
#     counts = df['Embarked'].value_counts().to_dict()
#     return f"Number of passengers embarked from each port: {counts}"

# def get_overall_survival_rate():
#     """Returns the overall survival rate."""
#     overall_rate = df['Survived'].mean() * 100
#     return f"Overall survival rate: {overall_rate:.2f}%."

# def get_num_survived():
#     """Returns the number of survivors."""
#     num_survived = df[df['Survived'] == 1].shape[0]
#     return f"Number of survivors: {num_survived}."

# def get_num_died():
#     """Returns the number of passengers who died."""
#     num_died = df[df['Survived'] == 0].shape[0]
#     return f"Number of deaths: {num_died}."

# def get_class_distribution():
#     """Returns the distribution of passenger classes."""
#     distribution = df['Pclass'].value_counts().to_dict()
#     return f"Passenger class distribution: {distribution}"

# def query_correlation():
#     """Returns the correlation between numerical columns and survival."""
#     numeric = df.select_dtypes(include=np.number).dropna()
#     numeric = numeric.drop(columns=['PassengerId'], errors='ignore')
#     corr = numeric.corr()['Survived']
#     return corr.to_string(float_format='%.2f')


# def survival_rate_by_gender():
#     """Calculates survival rates by gender."""
#     survival_rates = df.groupby('Sex')['Survived'].mean() * 100
#     return f"Survival rate by gender: {survival_rates.to_dict()}"
# def get_most_common_fare():
#     """Returns the most common ticket fare."""
#     most_common_fare = df['Fare'].mode()[0]
#     count = (df['Fare'] == most_common_fare).sum()
#     return f"The most common ticket fare was ${most_common_fare:.2f}, paid by {count} passengers."

# def survival_rate_by_embarked():
#     """Calculates survival rates by embarkation point."""
#     survival_rates = df.groupby('Embarked')['Survived'].mean() * 100
#     return f"Survival rate by embarkation point: {survival_rates.to_dict()}"
