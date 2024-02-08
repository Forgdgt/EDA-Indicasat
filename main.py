import streamlit as st 
import pandas as pd
import numpy as np
import random
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
from streamlit_pandas_profiling import st_profile_report
from streamlit_option_menu import option_menu
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
#funciones
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def seleccion_datos(df=None):
    
    dataframes = []  # Lista para almacenar los DataFrames generados
    nombres_categorias = []  # Lista para almacenar los nombres de categorías


    tipo_de_seleccion =['Por columnas','Por categoria']
    seleccion = st.radio('Seleccion de datos a Anlizar:',tipo_de_seleccion)

    if df is None:
        st.error("Por favor, carga un archivo CSV primero.")
        return dataframes,nombres_categorias

    if seleccion == 'Por columnas':
        st.subheader('Seleccionar columnas para comparar')
        col1, col2 = st.columns(2)
        columnas_numericas = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

        with col1:
            columna1 = st.selectbox('Seleccione la primera columna:', columnas_numericas)
        with col2:
            columna2 = st.selectbox('Seleccione la segunda columna:', columnas_numericas)
        if df[columna1].count() != df[columna2].count():
            st.error('Las columnas seleccionadas tienen diferentes cantidades de datos.')
        else:
            #cantidad_muestras = st.number_input("Ingrese la cantidad de muestras a tomar:", min_value=1, max_value=df[columna1].count(), value=df[columna1].count())
            #indices_muestra = random.sample(range(df[columna1].count()), cantidad_muestras)
            #muestra_columna1 = df[columna1].iloc[indices_muestra]
            #muestra_columna2 = df[columna2].iloc[indices_muestra]
            muestra_columna1 = df[columna1]
            muestra_columna2 = df[columna2]

            col1.subheader(f'Datos de la columna {columna1}:')
            col1.write(muestra_columna1)

            col2.subheader(f'Datos de la columna {columna2}:')
            col2.write(muestra_columna2)

            # Agregar los DataFrames a la lista
            dataframes.append(muestra_columna1)
            dataframes.append(muestra_columna2)
            nombres_categorias.append(columna1)
            nombres_categorias.append(columna2)


    elif seleccion == 'Por categoria':
        st.subheader('Seleccionar columna para análisis por categoría')
        columna_datos = st.selectbox('Seleccione la columna de datos:', df.select_dtypes(include=['int64', 'float64']).columns)

        columna_categoria = st.selectbox('Seleccione la columna de categoría:', df.select_dtypes(include=['object']).columns)
        
        categorias = df[columna_categoria].unique()
        datos_por_categoria = {}

        for categoria in categorias:
            datos_por_categoria[categoria] = df.loc[df[columna_categoria] == categoria, columna_datos]
        
        
        columns = st.columns(len(categorias))
        for i, (categoria, datos) in enumerate(datos_por_categoria.items()):
            with columns[i]:
                st.write(f"### Categoría: {categoria}")
                st.write(datos)
                dataframes.append(datos)
                nombres_categorias.append(categoria)

    return dataframes,nombres_categorias

def generar_graficos_distribucion(dataframes, nombres_categorias):
    num_plots = len(dataframes)
    st.write(f"### Distribución de:")
    columns = st.columns(num_plots)

    for i, (df, nombre_categoria) in enumerate(zip(dataframes, nombres_categorias)):
        with columns[i]:
            
            st.write(f"{nombre_categoria}")
            sns.histplot(df, kde=True)
            
            st.pyplot()
    
def generar_graficos_qq_plot(dataframes, nombres_categorias):
    num_plots = len(dataframes)
    st.write(f"### QQ-Plot de:")
    columns = st.columns(num_plots)
    
    for i, (df, nombre_categoria) in enumerate(zip(dataframes, nombres_categorias)):
        with columns[i]:
            st.write(f"{nombre_categoria}")
            fig, ax = plt.subplots()
            stats.probplot(df, plot=ax)
            ax.get_lines()[0].set_marker('.')  # Cambiar el marcador para una mejor visualización
            ax.get_lines()[1].set_linestyle('-')  # Cambiar el estilo de línea para una mejor visualización
            st.pyplot(fig)  

def generar_homesaticidad(dataframes, nombres_categorias):
    combined_df = pd.concat(dataframes, axis=1, ignore_index=True)
    combined_df.columns = nombres_categorias  # Asignar nombres de categorías como nombres de columnas


    fig, ax = plt.subplots()
    sns.boxplot(data=combined_df, ax=ax)
    ax.set_ylabel("Valores")
    ax.set_title("Boxplot de todas las categorías juntas")
    ax.set_xticklabels(combined_df.columns, rotation=45, ha='right')  # Rotar etiquetas en el eje x
    st.pyplot(fig)

    # Calcular homocedasticidad
    homoscedasticity_result = pg.homoscedasticity(combined_df)
    st.write("### Prueba de homocedasticidad:")
    st.write(homoscedasticity_result)
    homoscedasticity_passed = homoscedasticity_result['homoscedasticity'][0]

    if homoscedasticity_passed:
        st.success("Los datos cumplen con la homocedasticidad")
    else:
        st.warning("Los datos no cumplen con la homocedasticidad")

    return homoscedasticity_passed


        

def normality_test(dataframes, nombres_categorias):
    num_plots = len(dataframes)
    st.write(f"### Shapiro-Wilk Test para:")
    columns = st.columns(num_plots)

    for i, (df, nombre_categoria) in enumerate(zip(dataframes, nombres_categorias)):
        with columns[i]:
            st.write(f"{nombre_categoria}")
            normality_result = pg.normality(df)
            st.write(normality_result)
            p_value = normality_result['pval'][0]
            if p_value > 0.05:
                st.success(f"Los datos siguen una distribución normal (p-value = {p_value})")
            else:
                st.warning(f"Los datos no siguen una distribución normal (p-value = {p_value})")
                

def t_student(dataframes, nombres_categorias):
    st.subheader("T-test")
    st.markdown('''Test estadístico empleado para analizar si dos muestras proceden de poblaciones con la misma media. 
                        Para ello, cuantifica la diferencia entre la media de las dos muestras y, teniendo en cuenta la varianza de estas.''')
    st.markdown('''
                H0: No hay diferencias entre las medias (p-value alto): μx = μy
                
                Ha: Hay diferencias entre las medias (p-value bajo): μx ≠ μy
                 ''')
    
    st.markdown('**Independecia**')
    st.markdown('Las observaciones tienen que ser independientes unas de las otras.')
    tipo_de_seleccion =['Independientes','No Independietes']
    seleccion = st.radio('Tipo de muestra:',tipo_de_seleccion)
    
    generar_graficos_distribucion(dataframes, nombres_categorias)
    generar_graficos_qq_plot(dataframes, nombres_categorias)
    normality_test(dataframes, nombres_categorias)
    st.markdown('**homocedasticidad**')
    st.markdown('''La varianza de ambas poblaciones comparadas debe de ser igual. Tal como ocurre con la condición de normalidad, 
                si no se dispone de información de las poblaciones, esta condición se ha de asumir a partir de las muestras. 
                En caso de no cumplirse esta condición, se puede emplear la corrección de Welch''')
    #generar_homesaticidad(dataframes, nombres_categorias)


def cambio_de_tipo(df=None):
    # Mostrar los tipos de datos actuales y permitir al usuario cambiarlos
    st.sidebar.header("Configuración de tipos de datos")
    original_data_types = {}
    for column in main_data.columns:
        original_data_types[column] = str(main_data[column].dtype)
    # Crear un diccionario para almacenar los tipos de datos seleccionados por el usuario
    modified_data_types = original_data_types.copy()
    # Iterar sobre las columnas del DataFrame
    for column in main_data.columns:
        # Mostrar el tipo de dato original y permitir al usuario cambiarlo
        data_type = st.sidebar.selectbox(f"Tipo de dato para {column} (Original: {original_data_types[column]})", 
                                         options=["int", "float64", "str","object"], index=["int", "float64", "str","object"].index(original_data_types[column]))
        modified_data_types[column] = data_type

    # Convertir los tipos de datos según lo seleccionado por el usuario y mostrar el resultado
    modified_data = main_data.astype(modified_data_types)
    return modified_data

        

#Inicio
st.set_page_config(page_title="EDA", page_icon=":bar_chart:", layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)


# Web App Title
st.markdown('''
# **Exploratory Data Analysis**

Esta es una herramienta para la exploracion de datos

**Credit:** Carlos Ricord

---
''')

#1.  Side bar Menu
with st.sidebar:
    side_bar=1
    st.header(str(side_bar)+'. Cargue el archivo CSV')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file")

#2. Horizontal Menu
selected = option_menu(
        menu_title= "Menu Principal",
        options=["Analisis Descriptivo","Analisis Comparativo"],
        orientation="horizontal"
    )
#cargando arhivo si existe
if uploaded_file:   
    main_data = pd.read_csv(uploaded_file)
    data=cambio_de_tipo(main_data)

if selected =="Analisis Descriptivo":
    if uploaded_file:
        st.header('**Tabla de datos**')
        st.write(data)
        st.header('**Reporte de Datos**')
        pr=ProfileReport(data,explorative=True)
        st_profile_report(pr)
    else:
        st.info('Esperando que se cargue datos')

if selected =="Analisis Comparativo":
    if uploaded_file:
        st.header('**Datos utilizados en el analisis**')
        st.markdown('''Utilice esta seccion para hacer los filtros necesarios a la data principal.
                    Los filtros que se apliquen a esta data se estaran aplicando para todos los analisis''')
        tabla_filtrada=filter_dataframe(data)
        st.dataframe(tabla_filtrada)
        dataframes, nombres_categorias =seleccion_datos(tabla_filtrada)
        
        st.header("Variables Continuas (numericas)")
        st.markdown("---")
        t_student(dataframes, nombres_categorias)
        

    else:
        st.info('Esperando que se cargue datos')



