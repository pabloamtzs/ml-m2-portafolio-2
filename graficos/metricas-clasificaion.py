import numpy as np
import matplotlib.pyplot as plt

# Datos para el gráfico de métricas
modelos = ['Regresión Logística', 'Árboles de Decisión', 'Árbol de Decisión Optimizado', 'Random Forest', 'Random Forest Optimizado']
precision = [0.77, 0.74, 0.68, 0.85, 0.84]
recall = [0.91, 0.87, 0.87, 0.91, 0.93]
f1_score = [0.86, 0.83, 0.79, 0.90, 0.90]

x = np.arange(len(modelos))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, precision, width, label='Precisión')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

ax.set_xlabel('Modelos')
ax.set_ylabel('Métricas')
ax.set_title('Comparación de Métricas de Clasificación por Modelo')
ax.set_xticks(x)
ax.set_xticklabels(modelos, rotation=45, ha='right')
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

plt.show()
