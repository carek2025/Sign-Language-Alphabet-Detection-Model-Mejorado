# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import os
# import time

# # Configuración de MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# # Rutas
# BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
# DATA_DIR = os.path.join(BASE_DIR, 'sign_data')

# # Función para extraer landmarks de un frame
# def extract_landmarks(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#     if results.multi_hand_landmarks:
#         landmarks = results.multi_hand_landmarks[0]
#         landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
#         return landmarks_array, True
#     return None, False

# # Función para cargar el modelo y las clases
# def load_model_and_classes():
#     model_path = os.path.join(BASE_DIR, 'best_model.h5')
#     classes_path = os.path.join(BASE_DIR, 'classes.npy')
    
#     if not os.path.exists(model_path):
#         print(f"Error: No se encontró el modelo en {model_path}")
#         exit()
#     if not os.path.exists(classes_path):
#         print(f"Advertencia: No se encontró el archivo de clases en {classes_path}")
#         print("Creando clases A-Z por defecto...")
#         classes = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
#         np.save(classes_path, classes)
    
#     try:
#         model = load_model(model_path)
#         classes = np.load(classes_path, allow_pickle=True)
#         le = LabelEncoder()
#         le.classes_ = classes
#         return model, le
#     except Exception as e:
#         print(f"Error al cargar el modelo o las clases: {e}")
#         exit()

# # Función para capturar nuevos datos para una clase
# def capture_new_class_data(class_name, num_images=100):
#     cap = cv2.VideoCapture(0)
#     class_dir = os.path.join(DATA_DIR, class_name)
#     os.makedirs(class_dir, exist_ok=True)
#     landmarks_list = []

#     print(f"Capturando {num_images} imágenes para la clase '{class_name}'. Presiona 's' para empezar.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             cv2.putText(frame, "Mano detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('s') and valid:
#             break

#     captured = 0
#     while captured < num_images:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             landmarks_list.append(landmarks)
#             captured += 1
#             print(f"Imagen {captured}/{num_images} capturada para '{class_name}'.")
#             time.sleep(0.1)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     if landmarks_list:
#         np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), np.array(landmarks_list))
#     return np.array(landmarks_list)

# # Función para entrenar incrementalmente
# def incremental_training(model, le, new_class_name, new_landmarks):
#     # Normalizar coordenadas z de los nuevos landmarks
#     z_indices = np.arange(2, new_landmarks.shape[1], 3)
#     scaler = StandardScaler()
#     new_landmarks[:, z_indices] = scaler.fit_transform(new_landmarks[:, z_indices])

#     # Preparar datos nuevos
#     new_y = [new_class_name] * new_landmarks.shape[0]
#     old_classes = le.classes_.tolist()
#     all_classes = old_classes + [new_class_name]
#     le.classes_ = np.array(all_classes)
#     new_y_encoded = le.transform(new_y)
#     new_y_categorical = tf.keras.utils.to_categorical(new_y_encoded, num_classes=len(all_classes))

#     # Crear un nuevo modelo con una salida adicional
#     new_model = Sequential([
#         Dense(128, activation='relu', input_shape=(63,)),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(len(all_classes), activation='softmax')
#     ])
#     new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Transferir pesos del modelo antiguo
#     for i in range(len(model.layers) - 1):
#         new_model.layers[i].set_weights(model.layers[i].get_weights())
#     last_layer_weights = model.layers[-1].get_weights()
#     new_weights = np.zeros((last_layer_weights[0].shape[0], len(all_classes)))
#     new_weights[:, :-1] = last_layer_weights[0]
#     new_bias = np.zeros(len(all_classes))
#     new_bias[:-1] = last_layer_weights[1]
#     new_model.layers[-1].set_weights([new_weights, new_bias])

#     # Entrenar solo con los nuevos datos
#     new_model.fit(new_landmarks, new_y_categorical, epochs=10, batch_size=32)

#     # Guardar el nuevo modelo y clases
#     new_model.save(os.path.join(BASE_DIR, 'sign_model.h5'))
#     np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
#     return new_model, le

# # Main: Inferencia en tiempo real y opción para añadir nuevas clases
# def main():
#     model, le = load_model_and_classes()
#     cap = cv2.VideoCapture(0)
#     print("Presiona 'q' para salir, 'a' para añadir una nueva clase.")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             # Normalizar coordenadas z (como en entrenamiento)
#             z_indices = np.arange(2, landmarks.shape[0], 3)
#             landmarks[z_indices] = (landmarks[z_indices] - np.mean(landmarks[z_indices])) / np.std(landmarks[z_indices])
#             prediction = model.predict(np.expand_dims(landmarks, axis=0))
#             sign_class = le.inverse_transform([np.argmax(prediction)])[0]
#             cv2.putText(frame, f'Sign: {sign_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         cv2.imshow('Sign Language Recognition', cv2.flip(frame, 1))
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('a'):
#             class_name = input("Nombre de la nueva clase: ")
#             new_landmarks = capture_new_class_data(class_name)
#             if new_landmarks.size > 0:
#                 model, le = incremental_training(model, le, class_name, new_landmarks)
#                 print(f"Modelo actualizado con la nueva clase '{class_name}'.")

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()



# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import os
# import time

# # Configuración de MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# # Rutas  C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm\inference.py
# BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
# DATA_DIR = os.path.join(BASE_DIR, 'sign_data')

# # Función para extraer landmarks de un frame
# def extract_landmarks(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#     if results.multi_hand_landmarks:
#         landmarks = results.multi_hand_landmarks[0]
#         landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
#         return landmarks_array, True
#     return None, False

# # Función para cargar el modelo y las clases
# def load_model_and_classes():
#     model_path = os.path.join(BASE_DIR, 'best_model.h5')
#     classes_path = os.path.join(BASE_DIR, 'classes.npy')
    
#     if not os.path.exists(model_path):
#         print(f"Error: No se encontró el modelo en {model_path}")
#         exit()
#     if not os.path.exists(classes_path):
#         print(f"Advertencia: No se encontró el archivo de clases en {classes_path}")
#         print("Creando clases A-Z por defecto...")
#         classes = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
#         np.save(classes_path, classes)
    
#     try:
#         model = load_model(model_path)
#         classes = np.load(classes_path, allow_pickle=True)
#         le = LabelEncoder()
#         le.classes_ = classes
#         return model, le
#     except Exception as e:
#         print(f"Error al cargar el modelo o las clases: {e}")
#         exit()

# # Función para capturar nuevos datos para una clase
# def capture_new_class_data(class_name, num_images=100):
#     cap = cv2.VideoCapture(0)
#     class_dir = os.path.join(DATA_DIR, class_name)
#     os.makedirs(class_dir, exist_ok=True)
#     landmarks_list = []

#     print(f"Capturando {num_images} imágenes para la clase '{class_name}'. Presiona 's' para empezar.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: No se pudo leer el frame.")
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             cv2.putText(frame, "Mano detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('s') and valid:
#             break

#     captured = 0
#     while captured < num_images:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: No se pudo leer el frame.")
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             landmarks_list.append(landmarks)
#             captured += 1
#             print(f"Imagen {captured}/{num_images} capturada para '{class_name}'.")
#             time.sleep(0.1)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     if landmarks_list:
#         np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), np.array(landmarks_list))
#         print(f"Landmarks guardados en {os.path.join(DATA_DIR, f'{class_name}_landmarks.npy')}")
#     return np.array(landmarks_list)

# # Función para entrenar incrementalmente
# def incremental_training(model, le, new_class_name, new_landmarks):
#     # Normalizar coordenadas z de los nuevos landmarks
#     z_indices = np.arange(2, new_landmarks.shape[1], 3)
#     scaler = StandardScaler()
#     new_landmarks[:, z_indices] = scaler.fit_transform(new_landmarks[:, z_indices])

#     # Preparar datos nuevos
#     new_y = [new_class_name] * new_landmarks.shape[0]
#     old_classes = le.classes_.tolist()
#     all_classes = old_classes + [new_class_name]
#     le.classes_ = np.array(all_classes)
#     new_y_encoded = le.transform(new_y)
#     new_y_categorical = tf.keras.utils.to_categorical(new_y_encoded, num_classes=len(all_classes))

#     # Crear un nuevo modelo con una salida adicional
#     new_model = Sequential([
#         Dense(256, activation='relu', input_shape=(63,)),
#         Dropout(0.4),
#         Dense(128, activation='relu'),
#         Dropout(0.4),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(len(all_classes), activation='softmax')
#     ])
#     new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

#     # Transferir pesos del modelo antiguo
#     for i in range(len(model.layers) - 1):
#         new_model.layers[i].set_weights(model.layers[i].get_weights())
#     last_layer_weights = model.layers[-1].get_weights()
#     new_weights = np.zeros((last_layer_weights[0].shape[0], len(all_classes)))
#     new_weights[:, :-1] = last_layer_weights[0]
#     new_bias = np.zeros(len(all_classes))
#     new_bias[:-1] = last_layer_weights[1]
#     new_model.layers[-1].set_weights([new_weights, new_bias])

#     # Entrenar con los nuevos datos
#     new_model.fit(new_landmarks, new_y_categorical, epochs=20, batch_size=32, validation_split=0.2)

#     # Guardar el nuevo modelo y clases
#     new_model.save(os.path.join(BASE_DIR, 'best_model.h5'))
#     np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
#     return new_model, le

# # Main: Inferencia en tiempo real y opción para añadir nuevas clases
# def main():
#     model, le = load_model_and_classes()
#     cap = cv2.VideoCapture(0)
#     print("Presiona 'q' para salir, 'a' para añadir una nueva clase.")
    
#     # Normalizador para coordenadas z
#     scaler = StandardScaler()
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: No se pudo leer el frame.")
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             # Normalizar coordenadas z
#             z_indices = np.arange(2, landmarks.shape[0], 3)
#             landmarks[z_indices] = scaler.fit_transform(landmarks[z_indices].reshape(-1, 1)).flatten()
#             prediction = model.predict(np.expand_dims(landmarks, axis=0))
#             sign_class = le.inverse_transform([np.argmax(prediction)])[0]
#             cv2.putText(frame, f'Sign: {sign_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         else:
#             cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         cv2.imshow('Sign Language Recognition', cv2.flip(frame, 1))
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('a'):
#             class_name = input("Nombre de la nueva clase: ")
#             new_landmarks = capture_new_class_data(class_name)
#             if new_landmarks.size > 0:
#                 model, le = incremental_training(model, le, class_name, new_landmarks)
#                 print(f"Modelo actualizado con la nueva clase '{class_name}'.")

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()

#---------------------------ULTIMO TIPO DE INFERENCIA----------------------------------

# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import os
# import time

# # Configuración de MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Rutas
# BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
# DATA_DIR = os.path.join(BASE_DIR, 'sign_data2')

# # Función para extraer landmarks de un frame
# def extract_landmarks(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#     if results.multi_hand_landmarks:
#         landmarks = results.multi_hand_landmarks[0]
#         landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
#         return landmarks_array, True
#     return None, False

# # Función para cargar el modelo, las clases y el scaler
# def load_model_and_classes():
#     model_path = os.path.join(BASE_DIR, 'best_model.h5')
#     classes_path = os.path.join(BASE_DIR, 'classes.npy')
#     scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    
#     if not os.path.exists(model_path):
#         print(f"Error: No se encontró el modelo en {model_path}")
#         exit()
#     if not os.path.exists(classes_path):
#         print(f"Advertencia: No se encontró el archivo de clases en {classes_path}")
#         print("Creando clases A-Z por defecto...")
#         classes = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
#         np.save(classes_path, classes)
#     if not os.path.exists(scaler_path):
#         print(f"Advertencia: No se encontró el scaler en {scaler_path}")
#         print("Continuando sin normalización de coordenadas z...")
#         scaler = None
#     else:
#         with open(scaler_path, 'rb') as f:
#             scaler = pickle.load(f)
    
#     try:
#         model = load_model(model_path)
#         classes = np.load(classes_path, allow_pickle=True)
#         le = LabelEncoder()
#         le.classes_ = classes
#         return model, le, scaler
#     except Exception as e:
#         print(f"Error al cargar el modelo o las clases: {e}")
#         exit()

# # Función para capturar nuevos datos para una clase
# def capture_new_class_data(class_name, num_images=100, scaler=None):
#     cap = cv2.VideoCapture(0)
#     class_dir = os.path.join(DATA_DIR, class_name)
#     os.makedirs(class_dir, exist_ok=True)
#     landmarks_list = []

#     print(f"Capturando {num_images} imágenes para la clase '{class_name}'. Presiona 's' para empezar.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: No se pudo leer el frame.")
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             cv2.putText(frame, "Mano detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('s') and valid:
#             break

#     captured = 0
#     while captured < num_images:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: No se pudo leer el frame.")
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             # Normalizar coordenadas z si hay scaler
#             if scaler is not None:
#                 z_indices = np.arange(2, landmarks.shape[0], 3)
#                 landmarks_z = landmarks[z_indices].reshape(1, -1)  # Forma (1, 21)
#                 landmarks[z_indices] = scaler.transform(landmarks_z).flatten()
#             landmarks_list.append(landmarks)
#             captured += 1
#             print(f"Imagen {captured}/{num_images} capturada para '{class_name}'.")
#             time.sleep(0.1)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     if landmarks_list:
#         np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), np.array(landmarks_list))
#         print(f"Landmarks guardados en {os.path.join(DATA_DIR, f'{class_name}_landmarks.npy')}")
#     return np.array(landmarks_list)

# # Función para entrenar incrementalmente
# def incremental_training(model, le, new_class_name, new_landmarks, scaler):
#     # Normalizar coordenadas z de los nuevos landmarks si hay scaler
#     if scaler is not None:
#         z_indices = np.arange(2, new_landmarks.shape[1], 3)
#         new_landmarks[:, z_indices] = scaler.transform(new_landmarks[:, z_indices])

#     # Preparar datos nuevos
#     new_y = [new_class_name] * new_landmarks.shape[0]
#     old_classes = le.classes_.tolist()
#     all_classes = old_classes + [new_class_name]
#     le.classes_ = np.array(all_classes)
#     new_y_encoded = le.transform(new_y)
#     new_y_categorical = tf.keras.utils.to_categorical(new_y_encoded, num_classes=len(all_classes))

#     # Crear un nuevo modelo con una salida adicional
#     new_model = Sequential([
#         Dense(256, activation='relu', input_shape=(63,)),
#         Dropout(0.4),
#         Dense(128, activation='relu'),
#         Dropout(0.4),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(len(all_classes), activation='softmax')
#     ])
#     new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                       loss='categorical_crossentropy',
#                       metrics=['accuracy'])

#     # Transferir pesos del modelo antiguo
#     for i in range(len(model.layers) - 1):
#         new_model.layers[i].set_weights(model.layers[i].get_weights())
#     last_layer_weights = model.layers[-1].get_weights()
#     new_weights = np.zeros((last_layer_weights[0].shape[0], len(all_classes)))
#     new_weights[:, :-1] = last_layer_weights[0]
#     new_bias = np.zeros(len(all_classes))
#     new_bias[:-1] = last_layer_weights[1]
#     new_model.layers[-1].set_weights([new_weights, new_bias])

#     # Entrenar con los nuevos datos
#     new_model.fit(new_landmarks, new_y_categorical, epochs=20, batch_size=32, validation_split=0.2)

#     # Guardar el nuevo modelo y clases
#     new_model.save(os.path.join(BASE_DIR, 'best_model.h5'))
#     np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
#     return new_model, le

# # Main: Inferencia en tiempo real y opción para añadir nuevas clases
# def main():
#     model, le, scaler = load_model_and_classes()
#     cap = cv2.VideoCapture(0)
#     print("Presiona 'q' para salir, 'a' para añadir una nueva clase.")
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: No se pudo leer el frame.")
#             break
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             # Normalizar coordenadas z si hay scaler
#             if scaler is not None:
#                 z_indices = np.arange(2, landmarks.shape[0], 3)
#                 landmarks_z = landmarks[z_indices].reshape(1, -1)  # Forma (1, 21)
#                 landmarks[z_indices] = scaler.transform(landmarks_z).flatten()
#             prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
#             sign_class = le.inverse_transform([np.argmax(prediction)])[0]
#             confidence = np.max(prediction)
#             cv2.putText(frame, f'Sign: {sign_class} ({confidence:.2f})', (10, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             # Depuración: mostrar predicciones para diagnosticar
#             print(f"Predicción: {sign_class}, Confianza: {confidence:.2f}")
#             results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         else:
#             cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         cv2.imshow('Sign Language Recognition', frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('a'):
#             class_name = input("Nombre de la nueva clase: ")
#             new_landmarks = capture_new_class_data(class_name, scaler=scaler)
#             if new_landmarks.size > 0:
#                 model, le = incremental_training(model, le, class_name, new_landmarks, scaler)
#                 print(f"Modelo actualizado con la nueva clase '{class_name}'.")

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()


import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import time

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Rutas
BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
DATA_DIR = os.path.join(BASE_DIR, 'sign_data')

# Función para extraer landmarks de una imagen o frame
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
        return landmarks_array, True
    return None, False

# Función para cargar el modelo, las clases y el scaler
def load_model_and_classes():
    model_path = os.path.join(BASE_DIR, 'best_model.h5')
    classes_path = os.path.join(BASE_DIR, 'classes.npy')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        exit()
    if not os.path.exists(classes_path):
        print(f"Advertencia: No se encontró el archivo de clases en {classes_path}")
        print("Creando clases A-Z por defecto...")
        classes = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        np.save(classes_path, classes)
    if not os.path.exists(scaler_path):
        print(f"Advertencia: No se encontró el scaler en {scaler_path}")
        print("Continuando sin normalización de coordenadas z...")
        scaler = None
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    try:
        model = tf.keras.models.load_model(model_path)
        classes = np.load(classes_path, allow_pickle=True)
        le = LabelEncoder()
        le.classes_ = classes
        return model, le, scaler
    except Exception as e:
        print(f"Error al cargar el modelo o las clases: {e}")
        exit()

# Función para capturar nuevos datos para una clase (alineada con Preprocesamiento.py)
def capture_new_class_data(class_name, num_images=100, scaler=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return np.array([])
    
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    landmarks_list = []

    print(f"Capturando {num_images} imágenes para la clase '{class_name}'. Asegúrate de que la mano sea detectada. Presiona 's' para empezar.")
    
    # Mostrar video con landmarks en tiempo real
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            cap.release()
            return np.array([])
        
        landmarks, valid = extract_landmarks(frame)
        if valid:
            for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Mano detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Captura', frame)
        if cv2.waitKey(1) & 0xFF == ord('s') and valid:
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            print("No se detectó una mano. Ajusta la posición y vuelve a presionar 's'.")

    # Capturar imágenes
    captured = 0
    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break
        
        landmarks, valid = extract_landmarks(frame)
        if valid:
            # Guardar imagen
            image_path = os.path.join(class_dir, f"{captured}.jpg")
            cv2.imwrite(image_path, frame)
            
            # Normalizar coordenadas z si hay scaler
            if scaler is not None:
                z_indices = np.arange(2, landmarks.shape[0], 3)
                landmarks_z = landmarks[z_indices].reshape(1, -1)
                landmarks[z_indices] = scaler.transform(landmarks_z).flatten()
            
            landmarks_list.append(landmarks)
            captured += 1
            print(f"Imagen {captured}/{num_images} guardada para '{class_name}'.")
            time.sleep(0.5)  # Pausa para ajustar la mano
        else:
            print("No se detectó mano en el frame. Intentando de nuevo...")
        
        # Mostrar frame
        if valid:
            for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Captura', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Guardar landmarks
    if landmarks_list:
        landmarks_array = np.array(landmarks_list)
        np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), landmarks_array)
        print(f"✅ Landmarks guardados en {os.path.join(DATA_DIR, f'{class_name}_landmarks.npy')}")
        print(f"✅ Imágenes guardadas en {os.path.abspath(class_dir)}")
        os.startfile(os.path.abspath(class_dir))  # Abrir carpeta
        return landmarks_array
    else:
        print("No se capturaron landmarks.")
        return np.array([])

# Función para entrenar incrementalmente
def incremental_training(model, le, new_class_name, new_landmarks, scaler):
    if new_landmarks.size == 0:
        print("No hay datos para entrenar la nueva clase.")
        return model, le

    # Actualizar clases
    old_classes = le.classes_.tolist()
    if new_class_name in old_classes:
        print(f"Error: La clase '{new_class_name}' ya existe.")
        return model, le
    
    all_classes = old_classes + [new_class_name]
    le.classes_ = np.array(all_classes)
    
    # Normalizar coordenadas z de los nuevos landmarks
    if scaler is not None:
        z_indices = np.arange(2, new_landmarks.shape[1], 3)
        new_landmarks[:, z_indices] = scaler.transform(new_landmarks[:, z_indices])
    
    # Preparar etiquetas
    new_y = [new_class_name] * new_landmarks.shape[0]
    new_y_encoded = le.transform(new_y)
    new_y_categorical = tf.keras.utils.to_categorical(new_y_encoded, num_classes=len(all_classes))
    
    # Crear nuevo modelo con una salida adicional
    new_model = Sequential([
        Dense(256, activation='relu', input_shape=(63,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, Ascendente(64, activation='relu')),
        Dropout(0.3),
        Dense(len(all_classes), activation='softmax')
    ])
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    # Transferir pesos
    for i in range(len(model.layers) - 1):
        new_model.layers[i].set_weights(model.layers[i].get_weights())
    
    # Inicializar nueva capa de salida
    last_layer_weights = model.layers[-1].get_weights()
    new_weights = np.zeros((last_layer_weights[0].shape[0], len(all_classes)))
    new_weights[:, :-1] = last_layer_weights[0]
    new_bias = np.zeros(len(all_classes))
    new_bias[:-1] = last_layer_weights[1]
    new_model.layers[-1].set_weights([new_weights, new_bias])
    
    # Entrenar con los nuevos datos
    new_model.fit(new_landmarks, new_y_categorical,
                  epochs=20,
                  batch_size=32,
                  validation_split=0.2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    
    # Guardar modelo y clases
    new_model.save(os.path.join(BASE_DIR, 'best_model.h5'))
    np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
    print(f"Modelo actualizado con {len(all_classes)} clases.")
    return new_model, le

# Main: Inferencia en tiempo real y opción para añadir nuevas clases
def main():
    model, le, scaler = load_model_and_classes()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return
    
    print("Presiona 'q' para salir, 'a' para añadir una nueva clase.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break
        
        landmarks, valid = extract_landmarks(frame)
        if valid:
            # Normalizar coordenadas z
            if scaler is not None:
                z_indices = np.arange(2, landmarks.shape[0], 3)
                landmarks_z = landmarks[z_indices].reshape(1, -1)
                landmarks[z_indices] = scaler.transform(landmarks_z).flatten()
            
            # Predecir
            prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
            sign_class = le.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction)
            cv2.putText(frame, f'Sign: {sign_class} ({confidence:.2f})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Dibujar landmarks
            for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Sign Language Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            class_name = input("Nombre de la nueva clase: ")
            new_landmarks = capture_new_class_data(class_name, scaler=scaler)
            if new_landmarks.size > 0:
                model, le = incremental_training(model, le, class_name, new_landmarks, scaler)
                print(f"Modelo actualizado con la nueva clase '{class_name}'.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()