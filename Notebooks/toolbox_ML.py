import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import math
import warnings

warnings.filterwarnings('ignore')

from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def describe_df(df):
    """
    Genera un resumen estadístico de un dataframe proporcionando información sobre el tipo de datos,
    porcentaje de valores nulos, valores únicos y cardinalidad de cada columna, pero con las filas
    y columnas completamente intercambiadas respecto a la versión inicial.

    Argumentos:
    - df (DataFrame de pandas): El dataframe a describir.

    Retorna:
    - DataFrame: Un nuevo dataframe con las estadísticas de cada columna del dataframe original,
      con las estadísticas como columnas y las características de los datos como filas.
    """

    # Preparar el dataframe de resumen con filas y columnas intercambiadas
    summary = pd.DataFrame({
        'Data type': df.dtypes,
        'Percent missing (%)': df.isna().mean() * 100,
        'Unique values': df.nunique(),
        'Cardinality percent (%)': (df.nunique() / len(df)) * 100
    })

    return summary.transpose()  # Transponer el resultado 

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Esta función analiza las columnas de un DataFrame para sugerir el tipo de variable que representan.
    Utiliza la cardinalidad y el porcentaje de cardinalidad de cada columna para determinar si se trata
    de una variable binaria, categórica, numérica continua o numérica discreta.
    
    Argumentos:
    - df (DataFrame de pandas): El DataFrame que contiene las variables a analizar.
    - umbral_categoria (int): Umbral que define el límite máximo de cardinalidad para considerar
      una variable como categórica. Si la cardinalidad de una columna es menor que este umbral, se
      sugiere que la variable es categórica.
    - umbral_continua (float): Umbral que define el porcentaje mínimo de cardinalidad para considerar
      una variable como numérica continua. Si la cardinalidad de una columna es mayor o igual que
      `umbral_categoria` y el porcentaje de cardinalidad es mayor o igual que este umbral, se sugiere
      que la variable es numérica continua.
      
    Retorna:
    - DataFrame: Un DataFrame que contiene dos columnas: "nombre_variable" y "tipo_sugerido". Cada
      fila del DataFrame representa una columna del DataFrame de entrada, con el nombre de la columna
      y el tipo sugerido de variable.
    """

    # Inicializar una lista para almacenar los resultados
    resultados = []
    
    # Iterar sobre cada columna del dataframe
    for columna in df.columns:
        # Calcular la cardinalidad de la columna
        cardinalidad = df[columna].nunique()
        
        # Calcular el porcentaje de cardinalidad
        porcentaje_cardinalidad = cardinalidad / len(df)
        
        # Determinar el tipo de variable
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                tipo_sugerido = "Numerica Continua"
            else:
                tipo_sugerido = "Numerica Discreta"
        
        # Agregar el resultado a la lista
        resultados.append({'nombre_variable': columna, 'tipo_sugerido': tipo_sugerido})
    
    # Convertir la lista de resultados en un DataFrame y devolverlo
    return pd.DataFrame(resultados)

def get_features_num_regresion(df, target_col, umbral_corr, pvalue= None):
    """
    Esta función devuelve las features para la creacion de un modelo de machine learning.

    Estas features deben ser variables numericas y disponer de una correlacón y significacion estadistica significativa
    con el target, definidos previamente por el usuario. La significacion estadistica es nula por defecto.

    Argumentos:
    - df (DataFrame de pandas): un dataframe pandas sobre el que realizar el estudio.
    - target_col (str): la columna seleccionada como target para nuestro modelo.
    - umbral_corr (float): la correlacion minima exigida a una variable con el target para ser designado como feature. 
      Debe estar comprendido entre 0 y 1.
    - pvalue (float o None): la significacion estadistica Pearson maxima exigida a una variable para ser designada como feature 
      (generalmente 0.005). Por defecto, es None

    Retorna:
    - Lista con las columnas designadas como features para el modelo. Tipo lista compuesto por cadenas de texto.
    """

    cardinalidad = df[target_col].nunique() / len(df[target_col])

    if (umbral_corr < 0) or (umbral_corr > 1):

        print('Variable umbral_corr incorrecto.')
        return None

    elif df[target_col].dtype not in ['int8', 'int16', 'int32','int64', 'float16', 'float32', 'float64']:

        print('La columna seleccionada como target debe ser numerica.')
        return None
    
    elif cardinalidad < 0: # este no se si ponerlo

        print('Tu variable target tiene una cardinalidad muy baja para ser target.')
        return None
    
    lista_numericas = []
    for column in df.columns:
        
        if df[column].dtypes in ['int8', 'int16', 'int32','int64', 'float16', 'float32', 'float64']:
            lista_numericas.append(column)

    lista_numericas.remove(target_col)
    lista_features = []
    for columna in lista_numericas:

        no_nulos = df.dropna(subset= [target_col, columna])
        corr, pearson = pearsonr(no_nulos[target_col], no_nulos[columna])

        if pvalue != None:
            if (abs(corr) >= umbral_corr) and (pearson <= pvalue):
                lista_features.append(columna)
        else:
            if abs(corr) >= umbral_corr:
                lista_features.append(columna)
    
    return lista_features

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):

    """
    Esta función realiza una serie de comprobaciones de validez sobre los argumentos de entrada, como si el primer argumento es un DataFrame, si la columna objetivo está presente en el DataFrame y si las columnas especificadas para considerar son válidas. Luego, filtra las columnas numéricas basadas en su correlación con la columna objetivo y, opcionalmente, en el valor de p-value.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame sobre el que se realizará el análisis.
    - target_col (str): La columna objetivo que se utilizará en el análisis de correlación.
    - columns (lista de str): La lista de columnas a considerar en el análisis de correlación.
    - umbral_corr (float): El umbral de correlación mínimo requerido para que una variable sea considerada relevante. Debe estar entre 0 y 1.
    - pvalue (float o None): El valor de p-value máximo aceptable para que una variable sea considerada relevante. Por defecto, es None.

    La función luego divide las columnas filtradas en grupos de hasta 4 y genera pairplots utilizando `sns.pairplot()`, mostrando las relaciones entre estas variables y la columna objetivo. Finalmente, devuelve una  lista de las columnas filtradas que cumplen los criterios de correlación y p-value. Si no hay variables que cumplan los criterios, imprime un mensaje de error y devuelve None.
    """

    # Comprobación de valores de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame.")
        return None
    
    if target_col not in df.columns:
        print("Error: 'target_col' debe ser una columna válida del DataFrame.")
        return None
    
    if not isinstance(columns, list):
        print("Error: 'columns' debe ser una lista de nombres de columnas.")
        return None
    
    for col in columns:
        if col not in df.columns:
            print(f"Error: '{col}' no es una columna válida del DataFrame.")
            return None
    
    if not isinstance(umbral_corr, (int, float)):
        print("Error: 'umbral_corr' debe ser un valor numérico.")
        return None
    
    if not isinstance(pvalue, (float, int, type(None))):
        print("Error: 'pvalue' debe ser un valor numérico o None.")
        return None
    
    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar en el rango [0, 1].")
        return None
    
    # Verificar que target_col sea una variable numérica continua del DataFrame
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: 'target_col' debe ser una variable numérica continua del DataFrame.")
        return None

    # Si la lista de columnas está vacía, seleccionar todas las variables numéricas
    if not columns:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Filtrar columnas según correlación y p-value si se proporcionan
    filtered_columns = []
    for col in columns:
        if col != target_col:
            correlation = pearsonr(df[target_col], df[col])[0]
            if abs(correlation) > umbral_corr:
                if pvalue is not None:
                    _, p_val = pearsonr(df[target_col], df[col])
                    if p_val < (1 - pvalue):
                        filtered_columns.append(col)
                else:
                    filtered_columns.append(col)
    
    if not filtered_columns:
        print("No hay variables que cumplan los criterios de correlación y p-value.")
        return None
    
    # Dividir las columnas filtradas en grupos de máximo 4 para pintar pairplots
    num_plots = (len(filtered_columns) // 3) + 1
    for i in range(num_plots):
        cols_to_plot = [target_col] + filtered_columns[i*3:(i+1)*3]
        sns.pairplot(df[cols_to_plot])
        plt.show()
    
    return filtered_columns

def get_features_cat_regression(df, target_col, p_value=0.05):
    """
    Identifica características categóricas relevantes para un modelo de regresión.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame sobre el que se realizará el análisis.
    - target_col (str): La columna objetivo que se utilizará en el análisis.
    - p_value (float): El valor de p máximo aceptable para considerar una característica como relevante.
      Por defecto, es 0.05.

    Retorna:
    - Lista con las columnas categóricas consideradas relevantes para el modelo de regresión.
      Tipo lista compuesto por cadenas de texto.
    """

    if df.empty:
        print("El dataframe esta vacío")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La columna que has puesto no es una columna numerica")
        return None
    if not isinstance(p_value, float) or 0 > p_value or 1 < p_value:
        print("El p_value no tiene un valor valido, recuerda que tiene que estar entre 0 y 1")
        return None
    if target_col not in df:
        print("La columna no esta en el Dataframe, cambiala por una valida")
        return None
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    relevant_columns = []
    
    for col in categorical_columns:
        grouped = df.groupby(col)[target_col].apply(list).to_dict()
        f_vals = []
        for key, value in grouped.items():
            f_vals.append(value)
        f_val, p_val = stats.f_oneway(*f_vals)
        if p_val <= p_value:
            relevant_columns.append(col)

    return relevant_columns


def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Realiza un análisis de las características categóricas en relación con una columna objetivo para un modelo de regresión.

    Argumentos:
    - df (DataFrame de pandas): El DataFrame que contiene los datos.
    - target_col (str): La columna objetivo para el análisis.
    - columns (list): Lista de columnas categóricas a considerar. Si está vacía, se considerarán todas las columnas categóricas del DataFrame.
    - pvalue (float): El nivel de significancia para determinar la relevancia estadística de las variables categóricas. Por defecto, es 0.05.
    - with_individual_plot (bool): Indica si se debe mostrar un histograma agrupado para cada variable categórica significativa. Por defecto, es False.

    Retorna:
    - Lista de las columnas categóricas que muestran significancia estadística con respecto a la columna objetivo.
    """

    # Verificar que dataframe sea un DataFrame de pandas
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El argumento 'dataframe' debe ser un DataFrame de pandas")

    # Verificar que target_col esté en el dataframe
    if target_col != "" and target_col not in df.columns:
        raise ValueError("La columna 'target_col' no existe en el DataFrame")

    # Verificar que las columnas en columns estén en el dataframe
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame")

    # Verificar que pvalue sea un valor válido
    if not isinstance(pvalue, (int, float)):
        raise ValueError("El argumento 'pvalue' debe ser un valor numérico")
    
    # Verificar que with_individual_plot sea un valor booleano
    if not isinstance(with_individual_plot, bool):
        raise ValueError("El argumento 'with_individual_plot' debe ser un valor booleano")

    # Si columns está vacío, seleccionar todas las variables categóricas
    if not columns:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    # Lista para almacenar las variables categóricas que cumplen con las condiciones
    significant_categorical_variables = []

    # Iterar sobre las columnas seleccionadas
    for col in columns:
        # Verificar si la columna es categórica
        if df[col].dtype == 'object':
            # Calcular el test de chi-cuadrado entre la columna categórica y target_col
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            
            # Verificar si el p-value es menor que el umbral de significancia
            if p_val < pvalue:
                # Agregar la columna a la lista de variables categóricas significativas
                significant_categorical_variables.append(col)

                sns.histplot(data=df, x=col, hue=target_col, multiple="stack")
                plt.title(f"Histograma agrupado de {col} según {target_col}")
                plt.show()
            else:
                print(f"No se encontró significancia estadística para la variable categórica '{col}' con '{target_col}'")

    # Si no se encontró significancia estadística para ninguna variable categórica
    if not significant_categorical_variables:
        print("No se encontró significancia estadística para ninguna variable categórica")

    # Devolver las variables categóricas que cumplen con las condiciones
    return significant_categorical_variables

def paramns_check(df:pd.DataFrame, target_col:str, columns:list, pvalue:float) -> bool:
    
    """
    Esta es una funcion de comprobacion para los parametros.

    Comprobamos que:

    .- el parametro df es un dataframe de pandas
    .- el target seleccionado es categorico, definido por un str que referencia clases, en caso de ser numerico corresponderia mapearlo a str
    .- que las columnas proporcionadas son numericas 
    .- que el pvalue es numerico y esta entre 0 y 1

    La función devuelve un booleano que certifica si los parametros introducidos son adecuados.
    """
    
    try:
        if type(df) != pd.core.frame.DataFrame:
            return False
        if df[target_col].dtype != 'object':
            return False
        for col in columns:
            pd.to_numeric(df[col])
        if (float(pvalue) > 1) or (float(pvalue) < 0):
            return False
    except:
        return False
    
    return True

def eval_model(target, predictions, problem_type, metrics):
        
    import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def eval_model(target, predictions, problem_type, metrics):
        
        """
    Evalúa un modelo de regresión o clasificación en base a las métricas especificadas.

    Argumentos:
    - target (array-like): Valores verdaderos de los datos objetivo.
    - predictions (array-like): Valores predichos por el modelo.
    - problem_type (str): Tipo de problema, 'regression' para regresión o 'classification' para clasificación.
    - metrics (list of str): Lista de métricas a evaluar. Las métricas posibles dependen del tipo de problema.

    Retorna:
    - tuple: Tupla con los resultados de las métricas solicitadas, en el orden en que aparecen en la lista de entrada.
    """
        results = []

        if problem_type == 'regression':
            if not all(metric in ['RMSE', 'MAE', 'MAPE', 'GRAPH'] for metric in metrics):
                raise ValueError('Las metricas para regresion deben ser "RMSE", "MAE", "MAPE", "GRAPH".')
            
            for metric in metrics:
                if metric == 'RMSE':
                    rmse = np.sqrt(mean_squared_error(target, predictions))
                    print(f'RMSE: {rmse}')
                    results.append(rmse)
                elif metric == 'MAE':
                    mae = mean_absolute_error(target, predictions)
                    print(f'MAE: {mae}')
                    results.append(mae)
                elif metric == 'MAPE':
                    try:
                        mape = np.mean(np.abs((target - predictions / target))) * 100
                    except ZeroDivisionError:
                         raise ValueError('No se puede calcuar el MAPE porque el target contiene valores 0')
                    print(f'MAPE: {mape}')
                    results.append(mape)
                elif metric == 'GRAPH':
                    plt.figure(figsize = (12, 8))
                    plt.scatter(target, predictions, alpha = 0.3)
                    plt.xlabel('Target')
                    plt.ylabel('Predictions')
                    plt.title('Target Vs Predictions')
                    plt.grid(True)
                    plt.show()
        
        elif problem_type == 'classification':
            print(type(metrics))
            if not all(metric.startswith(('ACCURACY', 'PRECISION', 'RECALL', 'CLASS REPORT', 'MATRIX')) for metric in metrics):
                raise ValueError('Las metricas para regresion deben ser "ACCURACY", "PRECISION", "RECALL", "CLASS_REPORT", "MATRIX", "MATRIX_RECALL", "MATRIX_PRED" o "PRECISION_X", "RECALL_X".')
            
            for metric in metrics:
                if metric == 'ACCURACY':
                    accuracy = accuracy_score(target, predictions)
                    print(f'Accuracy: {accuracy}')
                    results.append(accuracy)
                elif metric == 'PRECISION':
                    precision = precision_score(target, predictions, average = 'macro')
                    print(f'Precision: {precision}')
                    results.append(precision)
                elif metric == 'RECALL':
                    recall = recall_score(target, predictions, average = 'macro')
                    print(f'Recall: {recall}')
                    results.append(recall)
                elif metric == 'CLASS_REPORT':
                    report = classification_report(target, predictions)
                    print('Classification Report')
                    print(report)
                elif metric == 'MATRIX':
                    con_matrix = confusion_matrix(target, predictions)
                    print('Confusion Matrix')
                    print(con_matrix)
                    disp = ConfusionMatrixDisplay(confusion_matrix = con_matrix)
                    disp.plot()
                    plt.show()
                elif metric == 'MATRIX_RECALL':
                    mat_rec = confusion_matrix(target, predictions, normalize = True)
                    print('Confusion Matrix Normalize Recall')
                    print(mat_rec)
                    disp = ConfusionMatrixDisplay(confusion_matrix = mat_rec)
                    disp.plot()
                    plt.show()
                elif metric == 'MATRIX_PRED':
                    mat_pred = confusion_matrix(target, predictions, normalize = 'pred')
                    print(f'Confusion Matrix Normalize Predictions')
                    print(mat_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix = mat_pred)
                    disp.plot()
                    plt.show()
                elif metric.startswith('PRECISION_'):
                    label = metric.split('_')[1]
                    try:
                        precision = precision_score(target, predictions, labels = [label], average = 'micro')
                        print(f'Precision for class "{label}: {precision}')
                        results.append(precision)
                    except ValueError:
                        raise ValueError(f'La clase "{label}" no esta presente en el target.')
                elif metric.startswith('RECALL_'):
                    label = metric.split(_)[1]
                    try:
                        recall = recall_score(target, predictions, labels = [label], average = 'micro')
                        print(f'Recall for class "{label}": {recall}')
                        results.append(recall)
                    except ValueError:
                        raise ValueError(f'La clase "{label}" no esta presente en el target.')
        else:
            raise ValueError('El tipo de problema debe ser "regression" o "classification".')
        
        return tuple(results)

import pandas as pd
import numpy as np
from scipy.stats import f_oneway

def get_features_num_classification(df, target_col, pvalue=0.05):

    """
    Identifica columnas numéricas en un DataFrame que tienen un resultado significativo
    en la prueba ANOVA con respecto a una columna objetivo categórica.

    Parámetros:
    df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
    target_col (str): El nombre de la columna objetivo en el DataFrame. Esta debe ser 
                      una columna categórica con baja cardinalidad (10 o menos valores únicos).
    pvalue (float): El nivel de significancia para la prueba ANOVA. El valor predeterminado es 0.05.

    Retorna:
    list: Una lista de nombres de columnas numéricas que tienen una relación significativa con 
          la columna objetivo según la prueba ANOVA.
          Retorna None si alguna de las comprobaciones de los argumentos de entrada falla, 
          e imprime un mensaje indicando la razón.
    """
    
    # Comprobación de que el DataFrame no está vacío
    if df.empty:
        print("El DataFrame está vacío.")
        return None
    
    # Comprobación de que target_col está en el DataFrame
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no se encuentra en el DataFrame.")
        return None
       
    # Comprobación de que target_col es categórico
    if not pd.api.types.is_categorical_dtype(df[target_col]) and df[target_col].nunique() > 10:
        print(f"La columna '{target_col}' tiene demasiadas categorías (más de 10).")
        return None
    
    # Comprobación de que pvalue es un float y está en el rango correcto
    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("El valor de 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    # Filtrar las columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Eliminar la columna objetivo de la lista de columnas numéricas
    numeric_cols.remove(target_col)

    # Lista para almacenar las columnas que cumplen con el criterio
    significant_columns = []
    
    for col in numeric_cols:
        groups = [df[col][df[target_col] == category] for category in df[target_col].unique()]
        f_stat, p_val = f_oneway(*groups)
        if p_val <= pvalue:
            significant_columns.append(col)
    
    return significant_columns


def plot_features_num_classification(df:pd.DataFrame, target_col:str= '', columns:list= [], pvalue:float= 0.05) -> list:
    # version con generador de indices
    """
    Parámetros:
    .- df: un dataframe de pandas
    .- target_col: el nombre de la variable target (debe ser categorica objeto/str, si contiene numeros, procede mapearla)
    .- columns: el nombre de las variables numericas del df, adjuntas en una lista (vacia por defecto)
    .- pvalue: el valor con que queremos comprobar la significancia estadistica, 0.05 por defecto

    Esta funcion cumple tras objetivos: a saber:

    1.- retorna una lista con los nombres de las features numericas que superan un test anova de significancia estadistica superior al establecido en pvalue
    2.- printa una relacion de graficas comparativas de correlacion target-variables numericas para su estudio y miniEDA
    3.- printa una relacion de graficas comparativas de colinealidad entre las distinta variables numericas para su estudio y miniEDA

    Explicamos la funcion mas en detalle a continuacion.
    """

    paramns_ok = paramns_check(df, target_col, columns, pvalue) # comprobamos que los parametros son adecuados, si no lo son retornamos None y printamos que no lo son
    if not paramns_ok:
        print('Los parametros introduciodos son incorrectos.')
        return None

    if not columns: # si no adjuntamos lista de var numericas, cogemos todas las numericas del df
        columns = df.describe().columns.tolist()

    col_anova = [] # creamos lista vacia donde almacenaremos los nombres de var numericas que cumplen el test anova

    # a continuacion realizamos el test anova
    grps = df[target_col].unique().tolist() # almacenamo los diferentes valores posibles del target en una lista
    for feature in columns: # iteramos las var numricas
        prov_list = [] # lista provisional donde almacenaremos las series de realcion de cada var numrica con los diferentes valores del target
        
        for grp in grps:
            prov_list.append(df[df[target_col] == grp][feature]) # agregamos a la lista las series que comentabamos antes
        
        f_st, p_va = f_oneway(*prov_list) # realizamos el test anova sobre la var numerica de turno (en iteracion actual) en relacion con cada valor del target y comprobamos su pvalue en funcion de su varianza
        if p_va <= pvalue: # si hay significancia estadistica recahazamos H0(medias similares) y adjuntamos el nombre de la feature a col_anova 
            col_anova.append(feature) 
    
    # empezamos con las graficas
    col_anova.insert(0, target_col) # adjuntamos el target a col_anova porque lo necesitaremos para comparar y graficar

    # creamos una primera serie de graficas relacion target(categorica) con las features numericas
    # utilizaremos subplots para reflejar cada grafica individualmente. Estos subplots son referenciados mediante arrays, importante

    q_lineas = math.ceil((len(col_anova)-1)/5) # calculamos la cantidad de lineas que compondra en la figura grafica / array (cada linea comprendera 5 subplots / columnas)

    # vamos a jugar con generadores, uno simple en realidad, no lo hemos visto en temario pero para este caso resulta de mucha utilidad
    # para movernos por los subplots de la figura grafica de turno deberemos iterar las columnas segun grafiquemos diferentes relaciones target-features
    # este generador genera los indices para el subplot
    def gen_indice():
        
        while True:
            for linea in range(100):
                for columna in range(5):
                    yield linea, columna

    contador_indice = gen_indice() # instanciamos el generador


    fig, axs = plt.subplots(q_lineas, 5, figsize=(20, 4*q_lineas)) # generamos la figura grafica con la cantidad de lineas y 5 columnas, tamño acorde a la q de lineas
    fig.suptitle('Correlación target categorico VS features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9)

    columna = 0 # comenzamos en la linea 0, primera
    indice = next(contador_indice) # primer indice [0, 0]
    # comenzamos a iterar las features que tenemos que graficas
    for feature_index in range(1, len(col_anova)): # rango 1 hata final porque la primera es el target y no queremos graficar target-target
    
        try: # presumimos que la grafica dispondra de mas de 1 linea y graficaremos en consecuencia... 
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[indice], label= i)
            axs[indice].legend()
            indice = next(contador_indice) # siguiente indice
        except IndexError: # ...si la figura solo dispone de 1 linea la graficacion dara error y graficamos en consecuencia
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[columna], label= i)
            axs[columna].legend()
            columna += 1 # siguiente columna
    plt.show() # mostramos la figura grafica

    # graficamos la colinealidad
    sns.pairplot(df[col_anova], hue= target_col) # pairplot para todas las features numericas que han superado la significancia estadistica
    plt.suptitle('Colinealidad features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9) 
    plt.show() # mostramos grafica
    col_anova.remove(target_col) # quitamos el target de la lista de features que han superado el test (ya ha sido util para graficar)
    
    return col_anova # devolvemos los nombres de las features que han superado la significancia estadistica

def plot_features_cat_classification(df, target_col="", columns=[], mi_threshold=0.0, normalize=False):
    
    """
    Pinta las distribuciones de las columnas categoricas que pasan un threshold de informacion mutua con respecto a una columna objetivo haciendo uso de la funcionget_features_cat_classification
    
    Parámetros: 
    - df->dataframe objetivo 
    - target_col->columna(s) objetivo, pueden ser varias
    - mi_threshold->limite usado para la comprobacion de informacion mutua de las columnas
    - normalize->booleano que indica si se ha de normalizar o no a la hora de comprobar la informacion mutua
    
    Rertorna:
    - Plots de las variables que han pasado el limite de informacion mutua, representando la relacion entre esa columna y la columna objetivo
    """
    if not isinstance(df, pd.DataFrame):
        print("El dataframe proporcionado en realidad no es un dataframe")
        return None
    
    if target_col == "":
        print("Especifica una columna")
        return None
    
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no esta en el datarame")
        return None

    if not isinstance(df[target_col].dtype, pd.CategoricalDtype):
        df[target_col] = df[target_col].astype('category')
    
    if not columns:
        columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)
    
    if not all(col in df.columns for col in columns):
        print("Comprueba que todas las columnas espeficadas esten en el dataframe")
        return None
    
    selected_columns = get_features_cat_classification(df, target_col, normalize, mi_threshold)
    
    if not selected_columns:
        print("Ninguna columna cumple con la condicion de la informacion mutua")
        return None
    
    for col in selected_columns:
        plt.figure(figsize=(10, 6))
        df.groupby([col, target_col]).size().unstack().plot(kind='bar', stacked=True)
        plt.title(f'Distribucion de {target_col} con {col}')
        plt.xlabel(col)
        plt.ylabel('Contador')
        plt.legend(title=target_col)
        plt.show()



def get_features_cat_classification(dataframe, target_col, normalize=False, mi_threshold=0.0):
    # Validar el DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame de pandas.")
        return None
    
    # Validar el nombre de la columna objetivo
    if target_col not in dataframe.columns:
        print(f"Error: '{target_col}' no es una columna del DataFrame.")
        return None
    
    # Verificar que la columna objetivo es categórica o numérica discreta de baja cardinalidad
    if not (isinstance(dataframe[target_col].dtype, pd.CategoricalDtype) or 
            (dataframe[target_col].dtype in ['int64', 'float64', 'object'] and dataframe[target_col].nunique() < 20)):
        print(f"Error: La columna '{target_col}' debe ser categórica o numérica discreta con baja cardinalidad.")
        return None
    
    # Validar que normalize es un booleano
    if not isinstance(normalize, bool):
        print("Error: El argumento 'normalize' debe ser un booleano.")
        return None
    
    # Validar que mi_threshold es un número
    if not isinstance(mi_threshold, (int, float)):
        print("Error: El argumento 'mi_threshold' debe ser un número.")
        return None
    
    # Validar el rango de mi_threshold si normalize es True
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("Error: 'mi_threshold' debe ser un valor flotante entre 0 y 1 cuando 'normalize' es True.")
        return None
    
    # Seleccionar características categóricas y numéricas discretas
    cat_features = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_features = [col for col in cat_features if col != target_col]
    
    # Codificar características categóricas
    X = pd.get_dummies(dataframe[cat_features])
    y = dataframe[target_col].astype('category').cat.codes  # Codificar la columna objetivo como categorías numéricas

    # Cálculo de la información mutua
    mi = mutual_info_classif(X, y, discrete_features=True)
    
    # Normalización de la información mutua si se requiere
    if normalize:
        total_mi = sum(mi)
        if total_mi == 0:
            print("Error: La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi = mi / total_mi
    
    # Seleccionar características basadas en el umbral
    selected_features = [cat_features[i] for i in range(len(cat_features)) if mi[i] >= mi_threshold]
    
    return selected_features       