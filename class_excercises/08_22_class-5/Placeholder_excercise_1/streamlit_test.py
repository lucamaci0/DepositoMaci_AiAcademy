import os
import pandas as pd
import streamlit as st
 
st.write("""
# My first app
Hello *world!*
""")
 
datasets_dir = "other/Archivio Datasets"
lesson_name = "05_Lesson"
dataset_name = "Simple.csv"

dataset_path = os.path.join(datasets_dir, lesson_name, dataset_name)
df = pd.read_csv(dataset_path, parse_dates=["date"]).set_index("date")

st.line_chart(df)
"""
streamlit run 'c:/Users/LH668YN/OneDrive - EY/Desktop/AiAcademy/DepositoMaci_AiAcademy/class_excercises/08_22_class-5/Placeholder_excercise_1/streamlit_test.py'
"""