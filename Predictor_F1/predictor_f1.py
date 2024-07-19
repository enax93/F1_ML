import pickle
import pandas as pd
import streamlit as st

# Definir la función para predecir los grupos de posiciones
def predictor_grupos(model_path, df):
    """
    Cargamos el modelo guardado anteriormente para predecir el grupo de posición de los corredores de F1.

    Parámetros:
    - model_path: Ruta al archivo del modelo guardado
    - df: Un DataFrame que contiene las siguientes columnas:
      ['Race', 'Driver', 'Constructor', 'Circuit', 'Grid', 'Laps', 'Rank', 'Year', 'Date', 'Position_Order']

    Retorna:
    - Un diccionario de DataFrames con las predicciones del grupo de posición y la descripción del grupo por carrera.
    """
    # Cargar el modelo desde el archivo
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Verificar las columnas necesarias del DataFrame
    required_columns = ['Race', 'Driver', 'Constructor', 'Circuit', 'Grid', 'Laps', 'Rank', 'Year', 'Date', 'Position_Order']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La columna {col} falta en el DataFrame de entrada.")
    
    # Definir la descripción de los grupos de posición
    position_desc = {
        1: "Posición 1-3",
        2: "Posición 4-6",
        3: "Posición 7-9",
        4: "Posición 10-14",
        5: "Posición 15+"
    }
    
    # Crear un diccionario para almacenar los DataFrames de resultados por carrera
    resultados_por_carrera = {}
    
    # Agrupar el DataFrame por la columna 'Race'
    carreras = df['Race'].unique()
    
    for carrera in carreras:
        df_carrera = df[df['Race'] == carrera].copy()
        
        # Realizar las predicciones para la carrera actual
        predictions = model.predict(df_carrera)
        
        # Añadir las predicciones y la descripción al DataFrame de la carrera
        df_carrera['Position_Group'] = predictions
        df_carrera['Position_Description'] = df_carrera['Position_Group'].map(position_desc)
        
        # Determinar si la predicción fue correcta
        df_carrera['Correct_Prediction'] = df_carrera.apply(
            lambda row: "Sí" if (
                (row['Position_Order'] <= 3 and row['Position_Description'] == "Posición 1-3") or
                (4 <= row['Position_Order'] <= 6 and row['Position_Description'] == "Posición 4-6") or
                (7 <= row['Position_Order'] <= 9 and row['Position_Description'] == "Posición 7-9") or
                (10 <= row['Position_Order'] <= 14 and row['Position_Description'] == "Posición 10-14") or
                (row['Position_Order'] >= 15 and row['Position_Description'] == "Posición 15+")
            ) else "No",
            axis=1
        )
        
        # Contar cuántas de las primeras 20 posiciones fueron correctamente predichas
        aciertos = df_carrera[df_carrera['Position_Order'] <= 20]['Correct_Prediction'].value_counts().get("Sí", 0)
        
        # Seleccionar las columnas de interés
        result_df = df_carrera[['Driver', 'Position_Order', 'Position_Description', 'Correct_Prediction']]
        
        # Guardar el DataFrame de resultados en el diccionario, incluyendo el conteo de aciertos
        resultados_por_carrera[carrera] = {
            'result_df': result_df,
            'aciertos': aciertos
        }
        
        # Guardar el DataFrame resultante en un archivo CSV
        #result_df.to_csv(f'resultados_{carrera}.csv', index=False)
    
    return resultados_por_carrera

# Ruta al modelo guardado
model_path = '../Models/best_rf_model.pkl'
# Leer el archivo CSV de prueba
df = pd.read_csv('../Test/Test_F1.csv')

resultados = predictor_grupos(model_path, df)

# Mostrar los resultados
for carrera, resultado in resultados.items():
    print(f"Resultados para la carrera: {carrera}")
    print(resultado['result_df'])
    print(f"Aciertos en las primeras 20 posiciones: {resultado['aciertos']}")

# Streamlit App
st.title('Predicciones de Grupos de Posiciones en Carreras de F1')

# Seleccionar carrera
carreras = df['Race'].unique()
selected_race = st.selectbox('Seleccione una carrera', carreras)

# Filtrar DataFrame por la carrera seleccionada
df_carrera = df[df['Race'] == selected_race]

# Seleccionar tipo de predicción
tipo_prediccion = st.radio('Seleccione el tipo de predicción', ['Todos los pilotos', 'Piloto específico'])

if tipo_prediccion == 'Piloto específico':
    pilotos = df_carrera['Driver'].unique()
    selected_piloto = st.selectbox('Seleccione un piloto', pilotos)
    df_piloto = df_carrera[df_carrera['Driver'] == selected_piloto]
    if st.button('Predecir'):
        resultados_piloto = predictor_grupos(model_path, df_piloto)
        st.write(resultados_piloto[selected_race]['result_df'])
        st.write(f"Aciertos: {resultados_piloto[selected_race]['aciertos']}")
else:
    if st.button('Predecir'):
        resultados_carrera = predictor_grupos(model_path, df_carrera)
        st.write(resultados_carrera[selected_race]['result_df'])
        st.write(f"Aciertos: {resultados_carrera[selected_race]['aciertos']}")
