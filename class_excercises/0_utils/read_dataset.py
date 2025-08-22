import os
import pandas as pd

datasets_dir = "other/Archivio Datasets"
lesson_name = "XX_Lesson"
dataset_name = "dataset_name.csv"
dataset_path = os.path.join(datasets_dir, lesson_name, dataset_name)

df = pd.read_csv(dataset_path)