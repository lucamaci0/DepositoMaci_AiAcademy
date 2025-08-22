import os
import pandas as pd
import streamlit as st
 
datasets_dir = "other/Archivio Datasets"
lesson_name = "05_Lesson"
dataset_name = "Simple.csv"

dataset_path = os.path.join(datasets_dir, lesson_name, dataset_name)
df = pd.read_csv(dataset_path, parse_dates=["date"]).set_index("date")


######################################################
# Title
######################################################

st.title('Counter Example')


######################################################
# Plot df
######################################################

st.line_chart(df)


######################################################
# Create counter with "Increment" button
######################################################

if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('Increment')
if increment:
    st.session_state.count += 1

st.write('Count = ', st.session_state.count)


######################################################
# Create sidebar for user input and allows greeting
######################################################

with st.sidebar.form("name_form"):
  st.header("Imposta il tuo nome e cognome")
  nome = st.text_input("Nome", placeholder="Nome")
  cognome = st.text_input("Cognome", placeholder="Cognome")

  inviato = st.form_submit_button("Saluta")

st.write("Saluto da sidebar:")
if nome.strip() or cognome.strip():
  st.success(f'Ciao {nome or ""} {cognome or ""}!')
else:
  st.warning("Inserisci nome e/o cognome nella sidebar.")


""" Equivalent ways to run code with streamlit:

python -m streamlit run 'C:/Users/LH668YN/OneDrive - EY/Desktop/AiAcademy/DepositoMaci_AiAcademy/class_excercises/08_22_class-5/02_streamlit/streamlit_basics.py'
streamlit run "c:/Users/LH668YN/OneDrive - EY/Desktop/AiAcademy/DepositoMaci_AiAcademy/class_excercises/08_22_class-5/02_streamlit/streamlit_basics.py" 

"""

