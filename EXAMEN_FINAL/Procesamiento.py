import pandas as pd
# a. Cargado del archivo
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(df.head())

# b. Limpieza de datos
# ¿Valores nulos?
print(df.isnull().sum())

# Ejemplo: Valores nulos en la columna'bmi'
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# c. Variables categóricas a variables numéricas con One-Hot Encoding.
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# d. Escalado de Datos: mejorar el rendimiento de algunos algoritmos de machine learning.
from sklearn.preprocessing import StandardScaler

numerical_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# e. Análisis Exploratorio de Datos (EDA) - GRÁFICOS
'''
import seaborn as sns
import matplotlib.pyplot as plt
# Distribución de la variable objetivo
sns.countplot(x='stroke', data=df)
plt.title('Distribución de Accidentes Cerebrovasculares')
plt.show()
# Relación entre edad y accidentes cerebrovasculares
sns.boxplot(x='stroke', y='age', data=df)
plt.title('Relación entre Edad y Accidentes Cerebrovasculares')
plt.show()
# Matriz de correlación
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()
'''
# Valores cercanos a 1: correlación positiva fuerte
# Valores cercanos a -1: correlación negativa fuerte
# Valores cercanos a 0: correlación débil

# ------------------- BALANCEO DE DATOS -------------------
print("------------------- BALANCEO DE DATOS -------------------")
# Distribución de la variable objetivo
print(df['stroke'].value_counts())

from sklearn.utils import resample
# Separar las clases
df_majority = df[df.stroke == 0]
df_minority = df[df.stroke == 1]
# Submuestrear la clase mayoritaria
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # muestra sin reemplazo
                                   n_samples=len(df_minority), # coincidir con la clse minoritaria
                                   random_state=42) # resultados reproducibles
# Combinar las clases
df_balanced = pd.concat([df_majority_downsampled, df_minority])
# Nueva distribución
print(df_balanced['stroke'].value_counts())

# ------------------- MODELADO PREDICTIVO -------------------
print("------------------- MODELADO PREDICTIVO -------------------")
# División de datos
from sklearn.model_selection import train_test_split
# Definición de variables X_res y y_res
X_res = df_balanced.drop('stroke', axis=1)
y_res = df_balanced['stroke']
# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Entrenamiento y evaluación del modelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
# Iniciar
model = RandomForestClassifier(random_state=42)
# Entrenar
model.fit(X_train, y_train)
# Predecir
y_pred = model.predict(X_test)
# Evaluar
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))

# *********** Justificación de Técnicas de Preprocesamiento y Balanceo ***********
# Valores Faltantes: Se han manejado porque los valores faltantes pueden limitar el análisis
# y reducir la precisión del modelo.
# Codificación de Variables Categóricas: Necesaria para usar estas variables en modelos de
# machine learning.
# Normalización o Escalado: Algunos modelos de machine learning, como SVM y KNN, requieren
# que las características estén en la misma escala. (INVESTIGADO)
# Balanceo de Datos: El balanceo es crucial porque un dataset desbalanceado puede llevar a
# que el modelo se sesgue hacia la clase mayoritaria, ignorando la clase minoritaria, que en
# este caso es la de mayor interés.

# ------------------- CLASIFICACIÓN -------------------
print("******************* CLASIFICACIÓN *******************")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Inicializar el modelo
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Entrenar el modelo
rf_model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rf = rf_model.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rf))

# ------------ 2. 
from sklearn.svm import SVC

# Inicializar el modelo
svm_model = SVC(kernel='rbf', probability=True, class_weight='balanced')

# Entrenar el modelo
svm_model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_svm = svm_model.predict(X_test)

# Evaluar el modelo
print(classification_report(y_test, y_pred_svm))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_svm))


# ********************* Evaluación de Confiabilidad y Matriz de Confusión **********************
print("******************* EVALUACIÓN DE CONFIABILIDAD Y MATRIZ DE CONFUSIÓN *******************")
from sklearn.model_selection import train_test_split
# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
# Inicializar el modelo
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
# Entrenar el modelo
rf_model.fit(X_train, y_train)
# Predecir en el conjunto de prueba
y_pred_rf = rf_model.predict(X_test)
from sklearn.metrics import classification_report, roc_auc_score
# Evaluar el modelo
print(classification_report(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rf))

print("----- Matriz de confusion -----")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Generar
conf_matrix = confusion_matrix(y_test, y_pred_rf)
# Visualizar
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Matriz de Confusión - Random Forest')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.show()

'''
    Reporte de Clasificación: Proporciona métricas como precisión, recall, y F1-score para cada clase. Es útil para evaluar el rendimiento del modelo en ambas clases (0 y 1).
    AUC-ROC: Mide la capacidad del modelo para distinguir entre las clases. Un valor más cercano a 1 indica un mejor rendimiento.
    Matriz de Confusión: Muestra el número de predicciones correctas e incorrectas clasificadas por cada clase. Es útil para identificar dónde el modelo comete errores.

'''

# ********************* SPLITS **********************
print("******************* SPLITS *******************")

import numpy as np
# Función para evaluar el modelo con una división específica
def evaluate_model(split_ratio):
    auc_scores = []
    for _ in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=split_ratio, random_state=None)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred_rf)
        auc_scores.append(auc_score)
    return auc_scores
# Evaluar con configuración académica (80/20)
academic_auc_scores = evaluate_model(0.2)
# Calcular la mediana de AUC-ROC
academic_median_auc = np.median(academic_auc_scores)
print("Mediana de AUC-ROC (80/20):", academic_median_auc)
# Evaluar con configuración de investigación (50/50)
research_auc_scores = evaluate_model(0.5)
# Calcular la mediana de AUC-ROC
research_median_auc = np.median(research_auc_scores)
print("Mediana de AUC-ROC (50/50):", research_median_auc)

# ********************* IMPLEMENTACIÓN DE PCA **********************
print("******************* IMPLEMENTACIÓN DE PCA *******************")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMote
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Cargar el dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Manejar valores nulos en 'bmi'
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Codificar variables categóricas
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# Escalar columnas numéricas
scaler = StandardScaler()
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Separar características y variable objetivo
X = df.drop('stroke', axis=1)
y = df['stroke']

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Inicializar el modelo
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Función para evaluar el modelo con PCA
def evaluate_model_with_pca(n_components):
    auc_scores = []
    pca = PCA(n_components=n_components)
    X_res_pca = pca.fit_transform(X_res)
    
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X_res_pca, y_res, test_size=0.3, random_state=None)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred_rf)
        auc_scores.append(auc_score)
    return np.median(auc_scores)

# Evaluar el modelo con diferentes números de componentes principales
components_list = [12, 10, 11, 9, 5, 3]
pca_results = {n: evaluate_model_with_pca(n) for n in components_list}

# Mostrar los resultados
for n, auc in pca_results.items():
    print(f"Mediana de AUC-ROC con {n} componentes: {auc}")

# La aplicación de PCA permite reducir la dimensionalidad del dataset 
# mientras se conserva la mayor cantidad posible de varianza.