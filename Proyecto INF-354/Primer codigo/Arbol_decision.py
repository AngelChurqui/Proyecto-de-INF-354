import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Cargar el dataset
data = pd.read_csv('/content/drive/MyDrive/dataset/IBM HR Data new.csv')

# Preprocesamiento de datos
def data_clean(data):
    # Aquí puedes realizar el preprocesamiento necesario para tu conjunto de datos
    # Asegúrate de eliminar columnas no relevantes o que contengan datos no numéricos
    # y realizar cualquier otra transformación necesaria
    
    return data

data = data_clean(data)
X = data.drop(['Attrition'], axis=1)
y = data['Attrition']

# Aplicar muestreo excesivo para equilibrar las clases
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Escalar características
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Construir el clasificador de árbol de decisión
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Calcular la precisión en los conjuntos de entrenamiento y prueba
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("Precisión en el conjunto de entrenamiento:", train_accuracy)
print("Precisión en el conjunto de prueba:", test_accuracy)

# Calcular la matriz de confusión
y_pred = model.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(confusion)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del Clasificador:", accuracy)
