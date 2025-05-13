import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Rutas
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

# Guardar el scaler
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler guardado en scaler.pkl")