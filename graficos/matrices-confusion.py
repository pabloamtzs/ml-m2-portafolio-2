import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Función para plotear matriz de confusión
def plot_confusion_matrix(cm, modelo_nombre):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['No Enfermo', 'Enfermo'], yticklabels=['No Enfermo', 'Enfermo'])
    plt.title(f'Matriz de Confusión - {modelo_nombre}')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.show()

# Matrices de confusión
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_best_tree = confusion_matrix(y_test, y_pred_best_tree)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)

# Plotear las matrices
plot_confusion_matrix(cm_logistic, 'Regresión Logística')
plot_confusion_matrix(cm_tree, 'Árboles de Decisión')
plot_confusion_matrix(cm_best_tree, 'Árbol de Decisión Optimizado')
plot_confusion_matrix(cm_rf, 'Random Forest')
plot_confusion_matrix(cm_best_rf, 'Random Forest Optimizado')
