import matplotlib.pyplot as plt

# Datos de precisión de los modelos
modelos = ['Regresión Logística', 'Árboles de Decisión', 'Árbol de Decisión Optimizado', 'Random Forest', 'Random Forest Optimizado']
precisiones = [84.24, 80.98, 77.17, 88.59, 89.13]

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(modelos, precisiones, color='skyblue')
plt.xlabel('Precisión (%)')
plt.title('Precisión de los Modelos')
plt.xlim(70, 95)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
