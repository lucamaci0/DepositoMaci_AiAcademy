import os
import pandas as pd

datasets_dir = "other/Archivio Datasets/02_Lesson"
dataset = "AEP_hourly.csv"
dataset_dir = os.path.join(datasets_dir,dataset)

df = pd.read_csv(dataset_dir, parse_dates=["Datetime"])
print(df.head())
