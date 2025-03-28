from utils import db_connect
engine = db_connect()

# your code here
# Your code here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.feature_selection import chi2, SelectKBest
import numpy as np
from joblib import dump
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


url = "https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv"
dataframe = pd.read_csv(url, sep=";")
guardar = "/workspaces/Tutorial-de-Proyecto-de-Regresi-n-Log-stica/data/raw/bank-marketing.csv"
dataframe.to_csv(guardar, index=False)
dataframe.head()

dataframe.info()

features = dataframe.drop(columns=["y"]).columns

fig, ax = plt.subplots(len(features), 2, figsize=(25, 34), 
                       gridspec_kw={'height_ratios': [6, 1] * (len(features) // 2)})

for i, feature in enumerate(features):
    row = (i // 2) * 2 
    col = i % 2
    sns.histplot(ax=ax[row, col], data=dataframe, x=feature)
    sns.boxplot(ax=ax[row + 1, col], data=dataframe, x=feature).set(xlabel=None)


plt.tight_layout()
plt.show()

# Listado de columnas categóricas
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
for col in categorical_columns:
    dataframe[f"{col}_n"] = pd.factorize(dataframe[col])[0]

dataframe["y_n"] = dataframe["y"].map({"yes": 1, "no": 0})
if dataframe.isnull().any().any():
    print("Missing values found")
    dataframe = dataframe.dropna(axis=1)
else:
    print("No missing values")

if dataframe.duplicated().any():
    print("Duplicates found")
    dataframe.drop_duplicates(inplace=True)
else:
    print("No duplicates")
dataframe = dataframe.select_dtypes(include='number')
dataframe.describe()

def identify_outliers(df):
    outlier_info = {}
    for col in df.select_dtypes(include='number').columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1 

        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if not outliers.empty:
            outlier_info[col] = {
                'outliers_count': outliers.shape[0],
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers[col].tolist()
            }
    for col, info in outlier_info.items():
        print(f"Column: {col} - {info['outliers_count']} outliers found")
        print(f"  Lower Bound: {info['lower_bound']}, Upper Bound: {info['upper_bound']}")
        print(f"  Sample Outliers: {info['outlier_values'][:5]}...\n")
    return outlier_info
outlier_info = identify_outliers(dataframe)

def remove_outliers(df, outlier_info):
    for col, info in outlier_info.items():
        lower_limit = info['lower_bound']
        upper_limit = info['upper_bound']
        
        if lower_limit < 0:
            df = df[df[col] >= lower_limit]
        if upper_limit > 0:
            df = df[df[col] <= upper_limit]
    return df
df_without_outliers = remove_outliers(dataframe, outlier_info=outlier_info)
df_without_outliers.describe().round(2)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Función de división de los datos en entrenamiento y prueba
def train_split(target, dataframe, test_size=0.1, random_state=42):
    return train_test_split(dataframe.drop(columns=target), dataframe[target], test_size=test_size, random_state=random_state)

# Función para guardar datos
def save_data(X_train, X_test, y_train, y_test, prefix, suffix=""):
    X_train.to_csv(f'../data/processed/{prefix}_X_train{suffix}.csv', index=False)
    X_test.to_csv(f'../data/processed/{prefix}_X_test{suffix}.csv', index=False)
    y_train.to_csv(f'../data/processed/{prefix}_y_train{suffix}.csv', index=False)
    y_test.to_csv(f'../data/processed/{prefix}_y_test{suffix}.csv', index=False)

# Función de escalado
def scale_data(X_train, X_test, scaler, filename, scaler_type='std'):
    scaler.fit(X_train)
    
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    dump(scaler, f'../data/processed/{scaler_type}_transform_{filename}.sav')

    return X_train_scaled, X_test_scaled

# Función de selección de las mejores características (SelectKBest)
def select_k_best(X_train, X_test, y_train, k, filename: str):
    selection_model = SelectKBest(f_classif, k=k)
    selection_model.fit(X_train, y_train)
    cols = selection_model.get_support()

    dump(selection_model, f'../data/processed/selection_model_{filename}.sav')

    return (pd.DataFrame(selection_model.transform(X_train), columns=X_train.columns.values[cols]),
            pd.DataFrame(selection_model.transform(X_test), columns=X_test.columns.values[cols]))

# Función para generar un heatmap de correlación
def plot_correlation_heatmap(df, columns):
    corr_matrix = df[columns].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.75}, annot_kws={"size": 10})

    plt.title("Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(fontsize=12, rotation=45, ha="right")
    plt.yticks(fontsize=12)
    plt.show()

# Preparación de los datos
TARGET = 'y_n'
X_train_with_outliers, X_test_with_outliers, y_train, y_test = train_split(TARGET, dataframe, test_size=0.2, random_state=42)
X_train_without_outliers, X_test_without_outliers, y_train_without_outliers, y_test_without_outliers = train_split(TARGET, df_without_outliers, test_size=0.2, random_state=42)

# Guardar los datos procesados (con y sin outliers)
save_data(X_train_with_outliers, X_test_with_outliers, y_train, y_test, 'with_outliers', '_ts_02_rs_42')
save_data(X_train_without_outliers, X_test_without_outliers, y_train_without_outliers, y_test_without_outliers, 'without_outliers', '_ts_02_rs_42')

# Escalado (StandardScaler y MinMaxScaler)
X_train_with_outliers_std, X_test_with_outliers_std = scale_data(X_train_with_outliers, X_test_with_outliers, StandardScaler(), 'with_outliers', 'std')
X_train_without_outliers_std, X_test_without_outliers_std = scale_data(X_train_without_outliers, X_test_without_outliers, StandardScaler(), 'without_outliers', 'std')

X_train_with_outliers_minmax, X_test_with_outliers_minmax = scale_data(X_train_with_outliers, X_test_with_outliers, MinMaxScaler(), 'with_outliers', 'minmax')
X_train_without_outliers_minmax, X_test_without_outliers_minmax = scale_data(X_train_without_outliers, X_test_without_outliers, MinMaxScaler(), 'without_outliers', 'minmax')

# Selección de las mejores características (SelectKBest)
K = X_train_with_outliers.shape[1]
X_train_with_outliers_selected, X_test_with_outliers_selected = select_k_best(X_train_with_outliers, X_test_with_outliers, y_train, K, 'with_outliers')
X_train_without_outliers_selected, X_test_without_outliers_selected = select_k_best(X_train_without_outliers, X_test_without_outliers, y_train_without_outliers, K, 'without_outliers')

# Correlación entre las características
plot_correlation_heatmap(dataframe, X_train_with_outliers.columns)

X_entrenamiento_con_outliers_std_seleccionado = X_train_with_outliers_std[X_train_with_outliers_selected.columns]
X_entrenamiento_sin_outliers_std_seleccionado = X_train_without_outliers_std[X_train_without_outliers_selected.columns]

X_prueba_con_outliers_std_seleccionado = X_test_with_outliers_std[X_test_with_outliers_selected.columns]
X_prueba_sin_outliers_std_seleccionado = X_test_without_outliers_std[X_test_without_outliers_selected.columns]

X_entrenamiento_con_outliers_minmax_seleccionado = X_train_with_outliers_minmax[X_train_with_outliers_selected.columns]
X_entrenamiento_sin_outliers_minmax_seleccionado = X_train_without_outliers_minmax[X_train_without_outliers_selected.columns]

X_prueba_con_outliers_minmax_seleccionado = X_test_with_outliers_minmax[X_test_with_outliers_selected.columns]
X_prueba_sin_outliers_minmax_seleccionado = X_test_without_outliers_minmax[X_test_without_outliers_selected.columns]

# Diccionarios de conjuntos de datos de entrenamiento y prueba
dfs_entrenamiento = {
    'X_entrenamiento_con_outliers_seleccionado': X_train_with_outliers_selected,
    'X_entrenamiento_con_outliers_norm_seleccionado': X_entrenamiento_con_outliers_std_seleccionado,
    'X_entrenamiento_con_outliers_minmax_seleccionado': X_entrenamiento_con_outliers_minmax_seleccionado,
}

dfs_entrenamiento_sin_outliers = {
    'X_entrenamiento_sin_outliers_seleccionado': X_train_without_outliers_selected,
    'X_entrenamiento_sin_outliers_norm_seleccionado': X_entrenamiento_sin_outliers_std_seleccionado,
    'X_entrenamiento_sin_outliers_minmax_seleccionado': X_entrenamiento_sin_outliers_minmax_seleccionado
}

dfs_prueba = {
    'X_prueba_con_outliers_seleccionado': X_test_with_outliers_selected,
    'X_prueba_con_outliers_norm_seleccionado': X_prueba_con_outliers_std_seleccionado,
    'X_prueba_con_outliers_minmax_seleccionado': X_prueba_con_outliers_minmax_seleccionado,
}

dfs_prueba_sin_outliers = {
    'X_prueba_sin_outliers_seleccionado': X_test_without_outliers_selected,
    'X_prueba_sin_outliers_norm_seleccionado': X_prueba_sin_outliers_std_seleccionado,
    'X_prueba_sin_outliers_minmax_seleccionado': X_prueba_sin_outliers_minmax_seleccionado
}

# Guardar los conjuntos de datos en archivos CSV
for nombre, df in dfs_entrenamiento.items():
    df.to_csv(f"../data/processed/{nombre}.csv", index=False)

for nombre, df in dfs_prueba.items():
    df.to_csv(f'../data/processed/{nombre}.csv', index=False)

for nombre, df in dfs_entrenamiento_sin_outliers.items():
    df.to_csv(f"../data/processed/{nombre}.csv", index=False)

for nombre, df in dfs_prueba_sin_outliers.items():
    df.to_csv(f'../data/processed/{nombre}.csv', index=False)



# Importar métricas necesarias
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Convertir diccionarios en listas de valores
entrenamiento = list(dfs_entrenamiento.values())
prueba = list(dfs_prueba.values())
entrenamiento_sin_outliers = list(dfs_entrenamiento_sin_outliers.values())
prueba_sin_outliers = list(dfs_prueba_sin_outliers.values())

resultados = []

# Modelo con datos originales
for indice, df_entrenamiento in enumerate(entrenamiento):
    modelo = LogisticRegression(random_state=42)
    modelo.fit(df_entrenamiento, y_train)
    y_predicho = modelo.predict(prueba[indice])

    resultados.append({
        'indice': indice,
        'df_entrenamiento': list(dfs_entrenamiento.keys())[indice],
        'Puntaje de Precisión': round(accuracy_score(y_test, y_predicho), 4),
        'Reporte de Clasificación': classification_report(y_test, y_predicho),
        'Matriz de Confusión': confusion_matrix(y_test, y_predicho)
    })

# Modelo sin outliers
for indice, df_entrenamiento in enumerate(entrenamiento_sin_outliers):
    modelo = LogisticRegression(random_state=42)
    modelo.fit(df_entrenamiento, y_train_without_outliers)
    y_predicho = modelo.predict(prueba_sin_outliers[indice])

    resultados.append({
        'indice': indice,
        'df_entrenamiento': list(dfs_entrenamiento_sin_outliers.keys())[indice],
        'Puntaje de Precisión': round(accuracy_score(y_test_without_outliers, y_predicho), 4),
        'Reporte de Clasificación': classification_report(y_test_without_outliers, y_predicho),
        'Matriz de Confusión': confusion_matrix(y_test_without_outliers, y_predicho)
    })

# Ordenar resultados por puntaje de precisión en orden descendente
resultados = sorted(resultados, key=lambda x: x['Puntaje de Precisión'], reverse=True)
mejor_resultado = resultados[0]

# Imprimir el mejor resultado
print(f"Índice: {mejor_resultado['indice']}")
print(f"\ndf_entrenamiento: {mejor_resultado['df_entrenamiento']}")
print(f"\nPuntaje de Precisión: {mejor_resultado['Puntaje de Precisión']}")
print(f"\nReporte de Clasificación: {mejor_resultado['Reporte de Clasificación']}")
print(f"\nMatriz de Confusión: {mejor_resultado['Matriz de Confusión']}")

# Búsqueda de hiperparámetros con GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

hiperparametros = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grid = GridSearchCV(LogisticRegression(random_state=42, class_weight="balanced"), hiperparametros, scoring='f1', cv=5, verbose=0)

def advertencia(*args, **kwargs):
    pass

import warnings
warnings.warn = advertencia

# Entrenar modelo con los mejores hiperparámetros
grid.fit(dfs_entrenamiento_sin_outliers.get(mejor_resultado["df_entrenamiento"]), y_train_without_outliers)

print(f'Los mejores hiperparámetros son: {grid.best_params_}')

# Evaluar el modelo con los mejores hiperparámetros
modelo_grid = LogisticRegression(**grid.best_params_)
modelo_grid.fit(dfs_entrenamiento_sin_outliers.get(mejor_resultado["df_entrenamiento"]), y_train_without_outliers)
y_predicho = modelo_grid.predict(dfs_prueba_sin_outliers.get(list(dfs_prueba_sin_outliers)[mejor_resultado["indice"]]))
precisión_modelo_grid = round(accuracy_score(y_test_without_outliers, y_predicho), 4)

print(f'La precisión del modelo con los hiperparámetros es: {precisión_modelo_grid * 100}%, un aumento de {round(precisión_modelo_grid - (resultados[0]["Puntaje de Precisión"]), 4) * 100}% respecto al modelo predeterminado')

# Imprimir reporte final
print(f"Puntaje de Precisión: {precisión_modelo_grid}")
print(f"\nReporte de Clasificación: {classification_report(y_test_without_outliers, y_predicho)}")
print(f"\nMatriz de Confusión: {confusion_matrix(y_test_without_outliers, y_predicho)}")

# Mostrar matriz de confusión como gráfico
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_without_outliers, y_predicho)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()