# Librerias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Especificar la ruta relativa a la carpeta donde se encuentran los datasets
dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')

# Leer los archivos CSV desde la carpeta 'datasets'
df = pd.read_csv(os.path.join(dataset_dir, 'heart.csv'))


# Convertir columnas categóricas a numéricas
df['Sex'] = df['Sex'].replace({'M': 1, 'F': 0})
df['ChestPainType'] = df['ChestPainType'].replace({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3})
df['RestingECG'] = df['RestingECG'].replace({'Normal': 0, 'ST': 1, 'LVH': 2})
df['ExerciseAngina'] = df['ExerciseAngina'].replace({'Y': 1, 'N': 0})
df['ST_Slope'] = df['ST_Slope'].replace({'Up': 0, 'Flat': 1, 'Down': 2})

# Definir variables independientes (X) y dependiente (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Función para imprimir métricas de evaluación
def evaluar_modelo(nombre, y_true, y_pred):
    print(f"--- {nombre} ---")
    print("Precisión:", accuracy_score(y_true, y_pred) * 100, "%")
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))
    print("Reporte de clasificación:")
    print(classification_report(y_true, y_pred))
    print("\n")

# 1. Regresión Logística
print("Entrenando Regresión Logística...")
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
evaluar_modelo("Regresión Logística", y_test, y_pred_logistic)


# 2. Arboles de decisión sin Optimización de Hiperparámetros
print("Entrenando Árboles de Decisión...")
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
evaluar_modelo("Árboles de Decisión", y_test, y_pred_tree)

# 3. Arboles de Decisión con Hiperparámetros Optimizados
# Hiperparámetros para árboles de decisión
param_grid_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
tree_model = DecisionTreeClassifier(random_state=42) # Crear el modelo de Árbol de Decisión
grid_search_tree = GridSearchCV(estimator=tree_model, param_grid=param_grid_tree, cv=5, scoring='accuracy', verbose=2, n_jobs=-1) # Configurar GridSearchCV
grid_search_tree.fit(X_train, y_train) # Entrenar GridSearchCV para Árboles de Decisión
best_tree_model = grid_search_tree.best_estimator_ # Obtener el mejor modelo
y_pred_best_tree = best_tree_model.predict(X_test) # Hacer predicciones con el modelo optimizado
accuracy_best_tree = accuracy_score(y_test, y_pred_best_tree) # Evaluar el modelo con accuracy
evaluar_modelo("Árbol de Decisión Optimizado", y_test, y_pred_best_tree) # Mostrar métricas de evaluación
print(f"Mejores hiperparámetros de Árbol de Decisión: {grid_search_tree.best_params_}") # Mostrar los mejores hiperparámetros


# 4. Random Forest sin Optimización de Hiperparámetros
print("Entrenando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
evaluar_modelo("Random Forest", y_test, y_pred_rf)

# 5. Random Forest con Hiperparámetros Optimizados
# Definir los hiperparámetros para Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],           # Número de árboles
    'max_depth': [None, 10, 20, 30],           # Profundidad de los árboles
    'min_samples_split': [2, 5, 10],           # Muestras mínimas para dividir un nodo
    'min_samples_leaf': [1, 2, 4],             # Muestras mínimas por hoja
    'max_features': ['sqrt', 'log2']           # Número de características a considerar
}
rf_model = RandomForestClassifier(random_state=42) # Crear el modelo de Random Forest
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=2, n_jobs=-1) # Configurar GridSearchCV
grid_search_rf.fit(X_train, y_train) # Entrenar GridSearchCV
best_rf_model = grid_search_rf.best_estimator_ # Obtener el mejor modelo
y_pred_best_rf = best_rf_model.predict(X_test) # Hacer predicciones con el modelo optimizado
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf) # Evaluar el modelo con accuracy
print(f"Mejores hiperparámetros de Random Forest: {grid_search_rf.best_params_}") # Mostrar los mejores hiperparámetros
evaluar_modelo("Random Forest Optimizado", y_test, y_pred_best_rf) # Mostrar métricas de evaluación


# Comparación de precisiones
print("Comparación de Precisión de los Modelos:")
mejores_modelos = {
    'Regresión Logística': accuracy_logistic,
    'Árboles de Decisión sin Optimización': accuracy_tree,
    'Árbol de Decisión Optimizado': accuracy_best_tree,
    'Random Forest sin Optimzación': accuracy_rf,
    'Random Forest Optimizado': accuracy_best_rf
}

for modelo, precision in mejores_modelos.items():
    print(f"{modelo}: {precision * 100:.2f}%")

mejor_modelo = max(mejores_modelos, key=mejores_modelos.get)
print(f"\nEl modelo con mejor precisión es: {mejor_modelo} con precisión de {mejores_modelos[mejor_modelo] * 100:.2f}%")
