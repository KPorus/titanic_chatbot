import kagglehub
import pandas as pd
import os
print(os.getcwd())

def download():
    path = kagglehub.dataset_download("brendan45774/test-file")
    print("Path to dataset files:", path)


def load_Dataset():
    df = pd.read_csv("https://raw.githubusercontent.com/KPorus/machine-learning-and-python/refs/heads/main/titanic.csv")
    print("Dataset loaded successfully.")
    # print(df)
    return df