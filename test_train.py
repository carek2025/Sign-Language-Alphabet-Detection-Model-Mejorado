# import numpy as np
# import os
# # Ruta absoluta al directorio de datos
# DATA_DIR = r'C:\Users\labin\OneDrive\Desktop\proyecto_final_version_2_mediapipe_cnn_svm\sign_data
# def load_landmarks(classes):
#     X, y = [], []
#     for class_name in classes:
#         landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
#         if os.path.exists(landmarks_path):
#             landmarks = np.load(landmarks_path)
#             X.append(landmarks)
#             y.extend([class_name] * landmarks.shape[0])
#         else:
#             print(f"Archivo no encontrado: {landmarks_path}")
#     X = np.vstack(X)  # Combina todos los *landmarks* en una matriz
#     y = np.array(y)   # Etiquetas como array
#     return X, 
# # Lista de clases de A a Z
# classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
# X, y = load_landmarks(classes)
# print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características por muestra.")



# import numpy as np
# import os

# # Ruta absoluta al directorio donde están los archivos .npy
# DATA_DIR = r'C:\Users\labin\OneDrive\Desktop\proyecto_final_version_2_mediapipe_cnn_svm\sign_data'

# # Lista de clases de A a Z
# classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# # Función para contar las muestras por clase
# def count_samples_per_class(classes):
#     total_samples = 0
#     for class_name in classes:
#         landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
#         if os.path.exists(landmarks_path):
#             landmarks = np.load(landmarks_path)
#             num_samples = landmarks.shape[0]
#             total_samples += num_samples
#             print(f"Clase {class_name}: {num_samples} muestras")
#             if num_samples < 100:
#                 print(f"  ¡Advertencia! La clase {class_name} tiene menos de 100 muestras.")
#         else:
#             print(f"Archivo no encontrado para la clase {class_name}: {landmarks_path}")
#     print(f"Total de muestras cargadas: {total_samples}")

# # Ejecutar la función
# count_samples_per_class(classes)


# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical

# # Ruta absoluta al directorio de datos
# DATA_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm\sign_data'

# # Paso 1: Cargar los datos
# def load_landmarks(classes):
#     X, y = [], []
#     for class_name in classes:
#         landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
#         if os.path.exists(landmarks_path):
#             landmarks = np.load(landmarks_path)
#             X.append(landmarks)
#             y.extend([class_name] * landmarks.shape[0])
#         else:
#             print(f"Archivo no encontrado: {landmarks_path}")
#     X = np.vstack(X)
#     y = np.array(y)
#     return X, y

# classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
# X, y = load_landmarks(classes)
# print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características por muestra.")

# # Paso 2: Normalizar coordenadas z
# z_indices = np.arange(2, X.shape[1], 3)
# scaler = StandardScaler()
# X[:, z_indices] = scaler.fit_transform(X[:, z_indices])

# # Paso 3: Codificar etiquetas
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# y_categorical = to_categorical(y_encoded)
# print(f"Etiquetas codificadas: {le.classes_}")

# # Paso 4: Dividir datos
# X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
# print(f"Entrenamiento: {X_train.shape[0]} muestras, Validación: {X_val.shape[0]} muestras")

# # Paso 5: Definir modelo
# num_classes = len(le.classes_)
# model = models.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(63,)),
#     layers.Dropout(0.3),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(num_classes, activation='softmax')
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# # Paso 6: Configurar callbacks
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
#     tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
# ]

# # Paso 7: Entrenar modelo
# history = model.fit(X_train, y_train,
#                     batch_size=32,
#                     epochs=2000,
#                     validation_data=(X_val, y_val),
#                     callbacks=callbacks)

# # Paso 8: Evaluar modelo
# val_loss, val_accuracy = model.evaluate(X_val, y_val)
# print(f"Precisión en validación: {val_accuracy:.4f}")

# # Paso 9: Guardar modelo y clases
# model.save('sign_model.h5')
# np.save('classes.npy', le.classes_)
# print("Modelo y clases guardados exitosamente.")

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Rutas   C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm\sign_data\K
BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
DATA_DIR = os.path.join(BASE_DIR, 'sign_data')
# Cargar los landmarks
def load_landmarks(classes):
    X, y = [], []
    for class_name in classes:
        landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
        if os.path.exists(landmarks_path):
            landmarks = np.load(landmarks_path)
            X.append(landmarks)
            y.extend([class_name] * landmarks.shape[0])
        else:
            print(f"Archivo no encontrado: {landmarks_path}")
    X = np.vstack(X)
    y = np.array(y)
    return X, y
# Clases A-Z
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
X, y = load_landmarks(classes)
print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características por muestra.")
# Normalizar coordenadas z
z_indices = np.arange(2, X.shape[1], 3)
scaler = StandardScaler()
X[:, z_indices] = scaler.fit_transform(X[:, z_indices])
# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
print(f"Etiquetas codificadas: {le.classes_}")
# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
print(f"Entrenamiento: {X_train.shape[0]} muestras, Validación: {X_val.shape[0]} muestras")
# Definir modelo
num_classes = len(le.classes_)
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# Callbacks
callbacks = [
    EarlyStopping(patience=50, restore_best_weights=True),
    ModelCheckpoint(os.path.join(BASE_DIR, 'best_model.h5'), save_best_only=True)
]
# Entrenar modelo
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=500,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)
# Evaluar modelo
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Precisión en validación: {val_accuracy:.4f}")
# Guardar modelo y clases
model.save(os.path.join(BASE_DIR, 'best_model.h5'))
np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
print("Modelo y clases guardados exitosamente.")