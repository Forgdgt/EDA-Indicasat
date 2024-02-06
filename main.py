import streamlit as st 
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(page_title="EDA", page_icon=":bar_chart:", layout="wide")

# Web App Title
st.markdown('''
# **Exploratory Data Analysis**

Esta es una herramienta para la exploracion de datos

**Credit:** Carlos Ricord

---
''')
side_bar=1
# Side bar
st.sidebar.header(str(side_bar)+'. Cargue el archivo CSV')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file")


    
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.header('**Tabla de datos**')
    st.write(df)
    st.header('**Reporte de Datos**')
    pr=ProfileReport(df,explorative=True)
    st_profile_report(pr)



else:
    st.info('Esperando que se cargue datos')
