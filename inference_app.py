#--------------------------------------APP VERSION 1(ERROR EN MEDIAPIPE AL ITERAR UANDO NO DETECTA N¿MANOS EN MULTIHAND PRINCIPAL MENTE EN INFERENCE_LOOP)---------------------------------
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import pickle
# import os
# import time
# import tkinter as tk
# from PIL import Image, ImageTk
# import threading

# # Configuración de MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Rutas
# BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
# DATA_DIR = os.path.join(BASE_DIR, 'sign_data')

# # Función para extraer landmarks de una imagen o frame
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
#         model = tf.keras.models.load_model(model_path)
#         classes = np.load(classes_path, allow_pickle=True)
#         le = LabelEncoder()
#         le.classes_ = classes
#         return model, le, scaler
#     except Exception as e:
#         print(f"Error al cargar el modelo o las clases: {e}")
#         exit()

# # Función para capturar nuevos datos para una clase
# def capture_new_class_data(class_name, num_images=100, scaler=None, text_area=None):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         text_area.insert(tk.END, "Error: No se puede acceder a la cámara.\n")
#         return np.array([])
    
#     class_dir = os.path.join(DATA_DIR, class_name)
#     os.makedirs(class_dir, exist_ok=True)
#     landmarks_list = []

#     text_area.insert(tk.END, f"Capturando {num_images} imágenes para la clase '{class_name}'. Asegúrate de que la mano sea detectada. Presiona 's' para empezar.\n")
#     text_area.see(tk.END)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
#             cap.release()
#             return np.array([])
        
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
#         elif cv2.waitKey(1) & 0xFF == ord('s'):
#             text_area.insert(tk.END, "No se detectó una mano. Ajusta la posición y vuelve a presionar 's'.\n")
#             text_area.see(tk.END)

#     captured = 0
#     while captured < num_images:
#         ret, frame = cap.read()
#         if not ret:
#             text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
#             break
        
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             image_path = os.path.join(class_dir, f"{captured}.jpg")
#             cv2.imwrite(image_path, frame)
            
#             if scaler is not None:
#                 z_indices = np.arange(2, landmarks.shape[0], 3)
#                 landmarks_z = landmarks[z_indices].reshape(1, -1)
#                 landmarks[z_indices] = scaler.transform(landmarks_z).flatten()
            
#             landmarks_list.append(landmarks)
#             captured += 1
#             text_area.insert(tk.END, f"Imagen {captured}/{num_images} guardada para '{class_name}'.\n")
#             text_area.see(tk.END)
#             time.sleep(0.5)
#         else:
#             text_area.insert(tk.END, "No se detectó mano en el frame. Intentando de nuevo...\n")
#             text_area.see(tk.END)
        
#         if valid:
#             for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
#     if landmarks_list:
#         landmarks_array = np.array(landmarks_list)
#         np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), landmarks_array)
#         text_area.insert(tk.END, f"✅ Landmarks guardados en {os.path.join(DATA_DIR, f'{class_name}_landmarks.npy')}\n")
#         text_area.insert(tk.END, f"✅ Imágenes guardadas en {os.path.abspath(class_dir)}\n")
#         text_area.see(tk.END)
#         os.startfile(os.path.abspath(class_dir))
#         return landmarks_array
#     else:
#         text_area.insert(tk.END, "No se capturaron landmarks.\n")
#         text_area.see(tk.END)
#         return np.array([])

# # Función para entrenar incrementalmente
# def incremental_training(model, le, new_class_name, new_landmarks, scaler, text_area):
#     if new_landmarks.size == 0:
#         text_area.insert(tk.END, "No hay datos para entrenar la nueva clase.\n")
#         return model, le

#     old_classes = le.classes_.tolist()
#     if new_class_name in old_classes:
#         text_area.insert(tk.END, f"Error: La clase '{new_class_name}' ya existe.\n")
#         return model, le
    
#     all_classes = old_classes + [new_class_name]
#     le.classes_ = np.array(all_classes)
    
#     if scaler is not None:
#         z_indices = np.arange(2, new_landmarks.shape[1], 3)
#         new_landmarks[:, z_indices] = scaler.transform(new_landmarks[:, z_indices])
    
#     new_y = [new_class_name] * new_landmarks.shape[0]
#     new_y_encoded = le.transform(new_y)
#     new_y_categorical = tf.keras.utils.to_categorical(new_y_encoded, num_classes=len(all_classes))
    
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
    
#     for i in range(len(model.layers) - 1):
#         new_model.layers[i].set_weights(model.layers[i].get_weights())
    
#     last_layer_weights = model.layers[-1].get_weights()
#     new_weights = np.zeros((last_layer_weights[0].shape[0], len(all_classes)))
#     new_weights[:, :-1] = last_layer_weights[0]
#     new_bias = np.zeros(len(all_classes))
#     new_bias[:-1] = last_layer_weights[1]
#     new_model.layers[-1].set_weights([new_weights, new_bias])
    
#     new_model.fit(new_landmarks, new_y_categorical,
#                   epochs=20,
#                   batch_size=32,
#                   validation_split=0.2,
#                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    
#     new_model.save(os.path.join(BASE_DIR, 'best_model.h5'))
#     np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
#     text_area.insert(tk.END, f"Modelo actualizado con {len(all_classes)} clases.\n")
#     text_area.see(tk.END)
#     return new_model, le

# # Nueva función para validar el modelo
# def validate_model(model, le, scaler, text_area):
#     text_area.insert(tk.END, "Validando el modelo...\n")
#     text_area.see(tk.END)
    
#     # Cargar datos de todas las clases
#     X, y = [], []
#     classes = le.classes_
#     for class_name in classes:
#         landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
#         if os.path.exists(landmarks_path):
#             landmarks = np.load(landmarks_path)
#             X.append(landmarks)
#             y.extend([class_name] * landmarks.shape[0])
#         else:
#             text_area.insert(tk.END, f"Advertencia: No se encontró {landmarks_path}\n")
#             text_area.see(tk.END)
    
#     if not X:
#         text_area.insert(tk.END, "Error: No se encontraron datos para validación.\n")
#         text_area.see(tk.END)
#         return
    
#     X = np.vstack(X)
#     y = np.array(y)
    
#     # Normalizar coordenadas z
#     if scaler is not None:
#         z_indices = np.arange(2, X.shape[1], 3)
#         X[:, z_indices] = scaler.transform(X[:, z_indices])
    
#     # Codificar etiquetas
#     y_encoded = le.transform(y)
#     y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(classes))
    
#     # Dividir en conjunto de validación
#     X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
#     # Evaluar modelo
#     val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
#     text_area.insert(tk.END, f"Resultados de validación:\n")
#     text_area.insert(tk.END, f"  Pérdida: {val_loss:.4f}\n")
#     text_area.insert(tk.END, f"  Precisión: {val_accuracy:.4f}\n")
#     text_area.see(tk.END)

# # Clase para la interfaz gráfica
# class SignLanguageApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Sign Language Recognition")
#         self.root.configure(bg="#000000")
#         self.running = False
#         self.model, self.le, self.scaler = load_model_and_classes()
#         self.cap = None
        
#         # Estilos
#         self.button_style = {
#             "bg": "#1E3A8A",
#             "fg": "#FFFFFF",
#             "font": ("Arial", 12, "bold"),
#             "bd": 2,
#             "relief": "raised",
#             "activebackground": "#3B82F6"
#         }
        
#         # Layout
#         self.main_frame = tk.Frame(root, bg="#000000")
#         self.main_frame.pack(padx=10, pady=10)
        
#         # Canvas para video
#         self.canvas = tk.Canvas(self.main_frame, width=640, height=480, bg="#000000", highlightthickness=2, highlightbackground="#1E3A8A")
#         self.canvas.grid(row=0, column=0, columnspan=2, pady=10)
        
#         # Área de texto
#         self.text_area = tk.Text(self.main_frame, height=10, width=80, bg="#000000", fg="#FFFFFF", font=("Arial", 10), bd=2, relief="sunken")
#         self.text_area.grid(row=1, column=0, columnspan=2, pady=10)
#         self.text_area.insert(tk.END, "Bienvenido al sistema de reconocimiento de lenguaje de señas.\n")
        
#         # Botones
#         self.btn_start = tk.Button(self.main_frame, text="Iniciar Inferencia", command=self.start_inference, **self.button_style)
#         self.btn_start.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
#         self.btn_add_class = tk.Button(self.main_frame, text="Añadir Clase", command=self.add_class, **self.button_style)
#         self.btn_add_class.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
#         self.btn_validate = tk.Button(self.main_frame, text="Validar Modelo", command=self.validate_model, **self.button_style)
#         self.btn_validate.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
#         self.btn_exit = tk.Button(self.main_frame, text="Salir", command=self.exit_app, **self.button_style)
#         self.btn_exit.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

#     def start_inference(self):
#         if not self.running:
#             self.running = True
#             self.cap = cv2.VideoCapture(0)
#             if not self.cap.isOpened():
#                 self.text_area.insert(tk.END, "Error: No se puede acceder a la cámara.\n")
#                 self.running = False
#                 return
#             self.text_area.insert(tk.END, "Iniciando inferencia en tiempo real...\n")
#             threading.Thread(target=self.inference_loop, daemon=True).start()

#     def inference_loop(self):
#         while self.running and self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
#                 break
            
#             landmarks, valid = extract_landmarks(frame)
#             if valid:
#                 if self.scaler is not None:
#                     z_indices = np.arange(2, landmarks.shape[0], 3)
#                     landmarks_z = landmarks[z_indices].reshape(1, -1)
#                     landmarks[z_indices] = self.scaler.transform(landmarks_z).flatten()
                
#                 prediction = self.model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
#                 sign_class = self.le.inverse_transform([np.argmax(prediction)])[0]
#                 confidence = np.max(prediction)
#                 cv2.putText(frame, f'Sign: {sign_class} ({confidence:.2f})', (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 for hand_landmarks in hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             else:
#                 cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
#             # Mostrar en canvas
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame_rgb)
#             img = img.resize((640, 480), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=img)
#             self.canvas.imgtk = imgtk
#             self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
#             self.root.update()
        
#         if self.cap:
#             self.cap.release()
#         self.running = False
#         self.text_area.insert(tk.END, "Inferencia detenida.\n")
#         self.text_area.see(tk.END)

#     def add_class(self):
#         class_name = tk.simpledialog.askstring("Input", "Nombre de la nueva clase:", parent=self.root)
#         if class_name:
#             new_landmarks = capture_new_class_data(class_name, scaler=self.scaler, text_area=self.text_area)
#             if new_landmarks.size > 0:
#                 self.model, self.le = incremental_training(self.model, self.le, class_name, new_landmarks, self.scaler, self.text_area)
#                 self.text_area.insert(tk.END, f"Modelo actualizado con la nueva clase '{class_name}'.\n")
#                 self.text_area.see(tk.END)

#     def validate_model(self):
#         threading.Thread(target=validate_model, args=(self.model, self.le, self.scaler, self.text_area), daemon=True).start()

#     def exit_app(self):
#         self.running = False
#         if self.cap:
#             self.cap.release()
#         self.root.quit()
#         self.root.destroy()

# def main():
#     root = tk.Tk()
#     app = SignLanguageApp(root)
#     root.mainloop()

# if __name__ == '__main__':
#     main()

#----------------------------------APP VERSION 2 -- FUNCIONAL PERO MODIFICA EL MODELO ORIGINAL ENVEZ DE CREAR UNO NUEVO-------------------
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import pickle
# import os
# import time
# import tkinter as tk
# from PIL import Image, ImageTk
# import threading
# from tkinter import simpledialog

# # Configuración de MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Rutas
# BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
# DATA_DIR = os.path.join(BASE_DIR, 'sign_data')

# # Función para extraer landmarks de una imagen o frame
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
#         model = tf.keras.models.load_model(model_path)
#         classes = np.load(classes_path, allow_pickle=True)
#         le = LabelEncoder()
#         le.classes_ = classes
#         return model, le, scaler
#     except Exception as e:
#         print(f"Error al cargar el modelo o las clases: {e}")
#         exit()

# # Función para capturar nuevos datos para una clase
# def capture_new_class_data(class_name, num_images=100, scaler=None, text_area=None):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         text_area.insert(tk.END, "Error: No se puede acceder a la cámara.\n")
#         return np.array([])
    
#     class_dir = os.path.join(DATA_DIR, class_name)
#     os.makedirs(class_dir, exist_ok=True)
#     landmarks_list = []

#     text_area.insert(tk.END, f"Capturando {num_images} imágenes para la clase '{class_name}'. Asegúrate de que la mano sea detectada. Presiona 's' para empezar.\n")
#     text_area.see(tk.END)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
#             cap.release()
#             return np.array([])
        
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             cv2.putText(frame, "Mano detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('s') and valid:
#             break
#         elif cv2.waitKey(1) & 0xFF == ord('s'):
#             text_area.insert(tk.END, "No se detectó una mano. Ajusta la posición y vuelve a presionar 's'.\n")
#             text_area.see(tk.END)

#     captured = 0
#     while captured < num_images:
#         ret, frame = cap.read()
#         if not ret:
#             text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
#             break
        
#         landmarks, valid = extract_landmarks(frame)
#         if valid:
#             image_path = os.path.join(class_dir, f"{captured}.jpg")
#             cv2.imwrite(image_path, frame)
            
#             if scaler is not None:
#                 z_indices = np.arange(2, landmarks.shape[0], 3)
#                 landmarks_z = landmarks[z_indices].reshape(1, -1)
#                 landmarks[z_indices] = scaler.transform(landmarks_z).flatten()
            
#             landmarks_list.append(landmarks)
#             captured += 1
#             text_area.insert(tk.END, f"Imagen {captured}/{num_images} guardada para '{class_name}'.\n")
#             text_area.see(tk.END)
#             time.sleep(0.5)
#         else:
#             text_area.insert(tk.END, "No se detectó mano en el frame. Intentando de nuevo...\n")
#             text_area.see(tk.END)
        
#         results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         cv2.imshow('Captura', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
#     if landmarks_list:
#         landmarks_array = np.array(landmarks_list)
#         np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), landmarks_array)
#         text_area.insert(tk.END, f"✅ Landmarks guardados en {os.path.join(DATA_DIR, f'{class_name}_landmarks.npy')}\n")
#         text_area.insert(tk.END, f"✅ Imágenes guardadas en {os.path.abspath(class_dir)}\n")
#         text_area.see(tk.END)
#         os.startfile(os.path.abspath(class_dir))
#         return landmarks_array
#     else:
#         text_area.insert(tk.END, "No se capturaron landmarks.\n")
#         text_area.see(tk.END)
#         return np.array([])

# # Función para entrenar incrementalmente
# def incremental_training(model, le, new_class_name, new_landmarks, scaler, text_area):
#     if new_landmarks.size == 0:
#         text_area.insert(tk.END, "No hay datos para entrenar la nueva clase.\n")
#         return model, le

#     old_classes = le.classes_.tolist()
#     if new_class_name in old_classes:
#         text_area.insert(tk.END, f"Error: La clase '{new_class_name}' ya existe.\n")
#         return model, le
    
#     all_classes = old_classes + [new_class_name]
#     le.classes_ = np.array(all_classes)
    
#     if scaler is not None:
#         z_indices = np.arange(2, new_landmarks.shape[1], 3)
#         new_landmarks[:, z_indices] = scaler.transform(new_landmarks[:, z_indices])
    
#     new_y = [new_class_name] * new_landmarks.shape[0]
#     new_y_encoded = le.transform(new_y)
#     new_y_categorical = tf.keras.utils.to_categorical(new_y_encoded, num_classes=len(all_classes))
    
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
    
#     for i in range(len(model.layers) - 1):
#         new_model.layers[i].set_weights(model.layers[i].get_weights())
    
#     last_layer_weights = model.layers[-1].get_weights()
#     new_weights = np.zeros((last_layer_weights[0].shape[0], len(all_classes)))
#     new_weights[:, :-1] = last_layer_weights[0]
#     new_bias = np.zeros(len(all_classes))
#     new_bias[:-1] = last_layer_weights[1]
#     new_model.layers[-1].set_weights([new_weights, new_bias])
    
#     new_model.fit(new_landmarks, new_y_categorical,
#                   epochs=100,
#                   batch_size=32,
#                   validation_split=0.2,
#                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    
#     new_model.save(os.path.join(BASE_DIR, 'best_model.h5'))
#     np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
#     text_area.insert(tk.END, f"Modelo actualizado con {len(all_classes)} clases.\n")
#     text_area.see(tk.END)
#     return new_model, le

# # Función para validar el modelo
# def validate_model(model, le, scaler, text_area):
#     text_area.insert(tk.END, "Validando el modelo...\n")
#     text_area.see(tk.END)
    
#     X, y = [], []
#     classes = le.classes_
#     for class_name in classes:
#         landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
#         if os.path.exists(landmarks_path):
#             landmarks = np.load(landmarks_path)
#             X.append(landmarks)
#             y.extend([class_name] * landmarks.shape[0])
#         else:
#             text_area.insert(tk.END, f"Advertencia: No se encontró {landmarks_path}\n")
#             text_area.see(tk.END)
    
#     if not X:
#         text_area.insert(tk.END, "Error: No se encontraron datos para validación.\n")
#         text_area.see(tk.END)
#         return
    
#     X = np.vstack(X)
#     y = np.array(y)
    
#     if scaler is not None:
#         z_indices = np.arange(2, X.shape[1], 3)
#         X[:, z_indices] = scaler.transform(X[:, z_indices])
    
#     y_encoded = le.transform(y)
#     y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(classes))
    
#     X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
#     val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
#     text_area.insert(tk.END, f"Resultados de validación:\n")
#     text_area.insert(tk.END, f"  Pérdida: {val_loss:.4f}\n")
#     text_area.insert(tk.END, f"  Precisión: {val_accuracy:.4f}\n")
#     text_area.see(tk.END)

# # Clase para la interfaz gráfica
# class SignLanguageApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Sign Language Recognition")
#         self.root.configure(bg="#000000")
#         self.running = False
#         self.model, self.le, self.scaler = load_model_and_classes()
#         self.cap = None
        
#         # Estilos
#         self.button_style = {
#             "bg": "#1E3A8A",
#             "fg": "#FFFFFF",
#             "font": ("Arial", 12, "bold"),
#             "bd": 2,
#             "relief": "raised",
#             "activebackground": "#3B82F6"
#         }
        
#         # Layout
#         self.main_frame = tk.Frame(root, bg="#000000")
#         self.main_frame.pack(padx=10, pady=10)
        
#         # Canvas para video
#         self.canvas = tk.Canvas(self.main_frame, width=640, height=480, bg="#000000", highlightthickness=2, highlightbackground="#1E3A8A")
#         self.canvas.grid(row=0, column=0, columnspan=2, pady=10)
        
#         # Área de texto
#         self.text_area = tk.Text(self.main_frame, height=10, width=80, bg="#000000", fg="#FFFFFF", font=("Arial", 10), bd=2, relief="sunken")
#         self.text_area.grid(row=1, column=0, columnspan=2, pady=10)
#         self.text_area.insert(tk.END, "Bienvenido al sistema de reconocimiento de lenguaje de señas.\n")
        
#         # Botones
#         self.btn_start = tk.Button(self.main_frame, text="Iniciar Inferencia", command=self.start_inference, **self.button_style)
#         self.btn_start.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
#         self.btn_add_class = tk.Button(self.main_frame, text="Añadir Clase", command=self.add_class, **self.button_style)
#         self.btn_add_class.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
#         self.btn_validate = tk.Button(self.main_frame, text="Validar Modelo", command=self.validate_model, **self.button_style)
#         self.btn_validate.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
#         self.btn_exit = tk.Button(self.main_frame, text="Salir", command=self.exit_app, **self.button_style)
#         self.btn_exit.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

#     def start_inference(self):
#         if not self.running:
#             self.running = True
#             self.cap = cv2.VideoCapture(0)
#             if not self.cap.isOpened():
#                 self.text_area.insert(tk.END, "Error: No se puede acceder a la cámara.\n")
#                 self.running = False
#                 return
#             self.text_area.insert(tk.END, "Iniciando inferencia en tiempo real...\n")
#             threading.Thread(target=self.inference_loop, daemon=True).start()

#     def inference_loop(self):
#         while self.running and self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
#                 break
            
#             landmarks, valid = extract_landmarks(frame)
#             if valid:
#                 if self.scaler is not None:
#                     z_indices = np.arange(2, landmarks.shape[0], 3)
#                     landmarks_z = landmarks[z_indices].reshape(1, -1)
#                     landmarks[z_indices] = self.scaler.transform(landmarks_z).flatten()
                
#                 prediction = self.model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
#                 sign_class = self.le.inverse_transform([np.argmax(prediction)])[0]
#                 confidence = np.max(prediction)
#                 cv2.putText(frame, f'Sign: {sign_class} ({confidence:.2f})', (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#                 if results.multi_hand_landmarks:  # Verificar si hay landmarks
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             else:
#                 cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
#             # Mostrar en canvas
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame_rgb)
#             img = img.resize((640, 480), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=img)
#             self.canvas.imgtk = imgtk
#             self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
#             self.root.update()
        
#         if self.cap:
#             self.cap.release()
#         self.running = False
#         self.text_area.insert(tk.END, "Inferencia detenida.\n")
#         self.text_area.see(tk.END)

#     def add_class(self):
#         class_name = tk.simpledialog.askstring("Input", "Nombre de la nueva clase:", parent=self.root)
#         if class_name:
#             new_landmarks = capture_new_class_data(class_name, scaler=self.scaler, text_area=self.text_area)
#             if new_landmarks.size > 0:
#                 self.model, self.le = incremental_training(self.model, self.le, class_name, new_landmarks, self.scaler, self.text_area)
#                 self.text_area.insert(tk.END, f"Modelo actualizado con la nueva clase '{class_name}'.\n")
#                 self.text_area.see(tk.END)

#     def validate_model(self):
#         threading.Thread(target=validate_model, args=(self.model, self.le, self.scaler, self.text_area), daemon=True).start()

#     def exit_app(self):
#         self.running = False
#         if self.cap:
#             self.cap.release()
#         self.root.quit()
#         self.root.destroy()

# def main():
#     root = tk.Tk()
#     app = SignLanguageApp(root)
#     root.mainloop()

# if __name__ == '__main__':
#     main()

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import threading
import glob
from datetime import datetime

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.7)

# Rutas
BASE_DIR = r'C:\Users\Laboratorio de Innov\Desktop\entre_1'
DATA_DIR = os.path.join(BASE_DIR, 'sign_data')

# Función para obtener el siguiente nombre de modelo
def get_next_model_name():
    existing_models = glob.glob(os.path.join(BASE_DIR, 'best_model*.h5'))
    if not existing_models:
        return 'best_model.h5'
    
    suffixes = [''] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    used_suffixes = [os.path.basename(m).replace('best_model', '').replace('.h5', '') for m in existing_models]
    for suffix in suffixes:
        if suffix not in used_suffixes:
            return f'best_model{suffix}.h5'
    raise ValueError("No hay sufijos disponibles para nuevos modelos.")

# Función para extraer landmarks de una imagen o frame
def extract_landmarks(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            return landmarks_array, True
        return None, False
    except Exception as e:
        print(f"Error en extract_landmarks: {e}")
        return None, False

# Función para cargar el modelo, las clases y el scaler
def load_model_and_classes(model_name='best_model.h5'):
    model_path = os.path.join(BASE_DIR, model_name)
    classes_path = os.path.join(BASE_DIR, 'classes.npy')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return None, None, None
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
        return None, None, None

# Función para capturar nuevos datos para una clase
def capture_new_class_data(class_name, num_images=100, scaler=None, text_area=None):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            text_area.insert(tk.END, "Error: No se puede acceder a la cámara.\n")
            return np.array([])
        
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        landmarks_list = []

        text_area.insert(tk.END, f"Capturando {num_images} imágenes para la clase '{class_name}'. Asegúrate de que la mano sea detectada. Presiona 's' para empezar.\n")
        text_area.see(tk.END)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
                cap.release()
                return np.array([])
            
            landmarks, valid = extract_landmarks(frame)
            if valid:
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, "Mano detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Captura', frame)
            if cv2.waitKey(1) & 0xFF == ord('s') and valid:
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                text_area.insert(tk.END, "No se detectó una mano. Ajusta la posición y vuelve a presionar 's'.\n")
                text_area.see(tk.END)

        captured = 0
        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
                break
            
            landmarks, valid = extract_landmarks(frame)
            if valid:
                image_path = os.path.join(class_dir, f"{captured}.jpg")
                cv2.imwrite(image_path, frame)
                
                if scaler is not None:
                    z_indices = np.arange(2, landmarks.shape[0], 3)
                    landmarks_z = landmarks[z_indices].reshape(1, -1)
                    landmarks[z_indices] = scaler.transform(landmarks_z).flatten()
                
                landmarks_list.append(landmarks)
                captured += 1
                text_area.insert(tk.END, f"Imagen {captured}/{num_images} guardada para '{class_name}'.\n")
                text_area.see(tk.END)
                time.sleep(0.5)
            else:
                text_area.insert(tk.END, "No se detectó mano en el frame. Intentando de nuevo...\n")
                text_area.see(tk.END)
            
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow('Captura', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        if landmarks_list:
            landmarks_array = np.array(landmarks_list)
            np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), landmarks_array)
            text_area.insert(tk.END, f"✅ Landmarks guardados en {os.path.join(DATA_DIR, f'{class_name}_landmarks.npy')}\n")
            text_area.insert(tk.END, f"✅ Imágenes guardadas en {os.path.abspath(class_dir)}\n")
            text_area.see(tk.END)
            os.startfile(os.path.abspath(class_dir))
            return landmarks_array
        else:
            text_area.insert(tk.END, "No se capturaron landmarks.\n")
            text_area.see(tk.END)
            return np.array([])
    
    except Exception as e:
        text_area.insert(tk.END, f"Error en captura de datos: {str(e)}\n")
        text_area.see(tk.END)
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        return np.array([])

# Función para entrenar incrementalmente con todas las clases
def incremental_training(model, le, new_class_name, new_landmarks, scaler, text_area, is_new_data=True):
    if new_landmarks.size == 0:
        text_area.insert(tk.END, "No hay datos para entrenar la nueva clase.\n")
        return model, le

    # Obtener todas las clases en sign_data
    class_files = glob.glob(os.path.join(DATA_DIR, '*_landmarks.npy'))
    all_classes = [os.path.basename(f).replace('_landmarks.npy', '') for f in class_files]
    
    # Verificar si la clase existía antes de la captura
    overwrite = False
    if new_class_name in all_classes and not is_new_data:
        response = messagebox.askyesno("Clase existente", 
                                      f"La clase '{new_class_name}' ya existe. ¿Deseas sobrescribir los datos existentes?",
                                      parent=text_area.master)
        if response:
            overwrite = True
            text_area.insert(tk.END, f"Sobrescribiendo datos para la clase '{new_class_name}'.\n")
        else:
            text_area.insert(tk.END, f"Cancelado: No se sobrescribirán los datos de '{new_class_name}'.\n")
            return model, le
    elif new_class_name not in all_classes:
        all_classes.append(new_class_name)
    
    # Actualizar clases en LabelEncoder
    le.classes_ = np.array(all_classes)
    
    # Cargar datos de todas las clases
    X, y = [], []
    text_area.insert(tk.END, "Cargando datos de las clases:\n")
    for class_name in all_classes:
        if class_name == new_class_name and (is_new_data or overwrite):
            landmarks = new_landmarks
        else:
            landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
            if os.path.exists(landmarks_path):
                landmarks = np.load(landmarks_path)
            else:
                text_area.insert(tk.END, f"Advertencia: No se encontró {landmarks_path}\n")
                continue
        X.append(landmarks)
        y.extend([class_name] * landmarks.shape[0])
        text_area.insert(tk.END, f"  - {class_name}: {landmarks.shape[0]} imágenes\n")
    
    if not X:
        text_area.insert(tk.END, "Error: No se encontraron datos para entrenamiento.\n")
        return model, le
    
    X = np.vstack(X)
    y = np.array(y)
    text_area.insert(tk.END, f"Total: {X.shape[0]} muestras cargadas.\n")
    
    # Normalizar coordenadas z
    if scaler is not None:
        z_indices = np.arange(2, X.shape[1], 3)
        X[:, z_indices] = scaler.transform(X[:, z_indices])
    
    # Codificar etiquetas
    y_encoded = le.transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(all_classes))
    
    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    text_area.insert(tk.END, f"Entrenamiento: {X_train.shape[0]} muestras, Validación: {X_val.shape[0]} muestras\n")
    
    # Crear nuevo modelo
    new_model = Sequential([
        Dense(256, activation='relu', input_shape=(63,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(all_classes), activation='softmax')
    ])
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(BASE_DIR, 'temp_model.h5'), save_best_only=True)
    ]
    
    # Entrenar modelo
    text_area.insert(tk.END, "Iniciando entrenamiento...\n")
    text_area.see(tk.END)
    try:
        new_model.fit(X_train, y_train,
                      batch_size=32,
                      epochs=500,
                      validation_data=(X_val, y_val),
                      callbacks=callbacks,
                      verbose=0)  # Silencioso para no saturar el área de texto
    except Exception as e:
        text_area.insert(tk.END, f"Error durante el entrenamiento: {str(e)}\n")
        return model, le
    
    # Guardar modelo
    new_model_name = get_next_model_name()
    new_model.save(os.path.join(BASE_DIR, new_model_name))
    np.save(os.path.join(BASE_DIR, 'classes.npy'), le.classes_)
    text_area.insert(tk.END, f"Modelo guardado como {new_model_name} con {len(all_classes)} clases.\n")
    text_area.see(tk.END)
    
    # Eliminar modelo temporal
    if os.path.exists(os.path.join(BASE_DIR, 'temp_model.h5')):
        os.remove(os.path.join(BASE_DIR, 'temp_model.h5'))
    
    return new_model, le

# Función para validar el modelo
def validate_model(model, le, scaler, text_area):
    text_area.insert(tk.END, "Validando el modelo...\n")
    text_area.see(tk.END)
    
    # Obtener todas las clases en sign_data
    class_files = glob.glob(os.path.join(DATA_DIR, '*_landmarks.npy'))
    classes = [os.path.basename(f).replace('_landmarks.npy', '') for f in class_files]
    
    X, y = [], []
    for class_name in classes:
        landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
        if os.path.exists(landmarks_path):
            landmarks = np.load(landmarks_path)
            X.append(landmarks)
            y.extend([class_name] * landmarks.shape[0])
        else:
            text_area.insert(tk.END, f"Advertencia: No se encontró {landmarks_path}\n")
            text_area.see(tk.END)
    
    if not X:
        text_area.insert(tk.END, "Error: No se encontraron datos para validación.\n")
        text_area.see(tk.END)
        return
    
    X = np.vstack(X)
    y = np.array(y)
    
    if scaler is not None:
        z_indices = np.arange(2, X.shape[1], 3)
        X[:, z_indices] = scaler.transform(X[:, z_indices])
    
    y_encoded = le.transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(le.classes_))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    try:
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        text_area.insert(tk.END, f"Resultados de validación:\n")
        text_area.insert(tk.END, f"  Pérdida: {val_loss:.4f}\n")
        text_area.insert(tk.END, f"  Precisión: {val_accuracy:.4f}\n")
    except Exception as e:
        text_area.insert(tk.END, f"Error durante la validación: {str(e)}\n")
    
    text_area.see(tk.END)

# Clase para la interfaz gráfica
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.configure(bg="#000000")
        self.running = False
        self.model, self.le, self.scaler = load_model_and_classes()
        self.current_model_name = 'best_model.h5' if self.model else None
        self.cap = None
        
        # Estilos
        self.button_style = {
            "bg": "#1E3A8A",
            "fg": "#FFFFFF",
            "font": ("Arial", 12, "bold"),
            "bd": 2,
            "relief": "raised",
            "activebackground": "#3B82F6"
        }
        
        # Layout
        self.main_frame = tk.Frame(root, bg="#000000")
        self.main_frame.pack(padx=10, pady=10)
        
        # Canvas para video
        self.canvas = tk.Canvas(self.main_frame, width=640, height=480, bg="#000000", highlightthickness=2, highlightbackground="#1E3A8A")
        self.canvas.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Área de texto
        self.text_area = tk.Text(self.main_frame, height=10, width=80, bg="#000000", fg="#FFFFFF", font=("Arial", 10), bd=2, relief="sunken")
        self.text_area.grid(row=1, column=0, columnspan=3, pady=10)
        self.text_area.insert(tk.END, "Bienvenido al sistema de reconocimiento de lenguaje de señas.\n")
        if self.current_model_name:
            self.text_area.insert(tk.END, f"Modelo actual: {self.current_model_name}\n")
        
        # Botones
        self.btn_start = tk.Button(self.main_frame, text="Iniciar Inferencia", command=self.start_inference, **self.button_style)
        self.btn_start.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_stop = tk.Button(self.main_frame, text="Detener Inferencia", command=self.stop_inference, **self.button_style)
        self.btn_stop.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_add_class = tk.Button(self.main_frame, text="Añadir Clase", command=self.add_class, **self.button_style)
        self.btn_add_class.grid(row=2, column=2, padx=5, pady=5, sticky="ew")
        
        self.btn_validate = tk.Button(self.main_frame, text="Validar Modelo", command=self.validate_model, **self.button_style)
        self.btn_validate.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_list_models = tk.Button(self.main_frame, text="Listar Modelos", command=self.list_models, **self.button_style)
        self.btn_list_models.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_exit = tk.Button(self.main_frame, text="Salir", command=self.exit_app, **self.button_style)
        self.btn_exit.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

    def start_inference(self):
        if not self.model:
            self.text_area.insert(tk.END, "Error: No hay modelo cargado. Selecciona un modelo primero.\n")
            self.text_area.see(tk.END)
            return
        if not self.running:
            try:
                self.running = True
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usar DSHOW para mejor compatibilidad
                if not self.cap.isOpened():
                    self.text_area.insert(tk.END, "Error: No se puede acceder a la cámara. Verifica que no esté en uso.\n")
                    self.running = False
                    return
                # Verificar compatibilidad del modelo
                output_shape = self.model.output_shape[-1]
                num_classes = len(self.le.classes_)
                if output_shape != num_classes:
                    self.text_area.insert(tk.END, f"Advertencia: El modelo espera {output_shape} clases, pero hay {num_classes} en classes.npy.\n")
                self.text_area.insert(tk.END, f"Iniciando inferencia con {self.current_model_name}...\n")
                threading.Thread(target=self.inference_loop, daemon=True).start()
            except Exception as e:
                self.text_area.insert(tk.END, f"Error al iniciar inferencia: {str(e)}\n")
                self.running = False
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def stop_inference(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.text_area.insert(tk.END, "Inferencia detenida.\n")
            self.text_area.see(tk.END)
            self.canvas.delete("all")
            self.canvas.create_text(320, 240, text="Inferencia detenida", fill="#FFFFFF", font=("Arial", 20))

    def inference_loop(self):
        try:
            while self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.text_area.insert(tk.END, "Error: No se pudo leer el frame.\n")
                    break
                
                landmarks, valid = extract_landmarks(frame)
                if valid:
                    if self.scaler is not None:
                        z_indices = np.arange(2, landmarks.shape[0], 3)
                        landmarks_z = landmarks[z_indices].reshape(1, -1)
                        landmarks[z_indices] = self.scaler.transform(landmarks_z).flatten()
                    
                    prediction = self.model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
                    sign_class = self.le.inverse_transform([np.argmax(prediction)])[0]
                    confidence = np.max(prediction)
                    cv2.putText(frame, f'Sign: {sign_class} ({confidence:.2f})', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                
                self.root.update()
        
        except Exception as e:
            self.text_area.insert(tk.END, f"Error en inferencia: {str(e)}\n")
            self.text_area.see(tk.END)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        self.text_area.insert(tk.END, "Inferencia detenida.\n")
        self.text_area.see(tk.END)
        self.canvas.delete("all")
        self.canvas.create_text(320, 240, text="Inferencia detenida", fill="#FFFFFF", font=("Arial", 20))

    def add_class(self):
        if not self.model:
            self.text_area.insert(tk.END, "Error: No hay modelo cargado. Selecciona un modelo primero.\n")
            self.text_area.see(tk.END)
            return
        class_name = simpledialog.askstring("Input", "Nombre de la nueva clase:", parent=self.root)
        if class_name:
            # Verificar si la clase existía antes de capturar
            pre_exists = os.path.exists(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"))
            new_landmarks = capture_new_class_data(class_name, num_images=100, scaler=self.scaler, text_area=self.text_area)
            if new_landmarks.size > 0:
                self.model, self.le = incremental_training(
                    self.model, self.le, class_name, new_landmarks, self.scaler, self.text_area,
                    is_new_data=not pre_exists  # is_new_data=True si la clase no existía antes
                )
                self.current_model_name = os.path.basename(get_next_model_name())
                self.text_area.insert(tk.END, f"Modelo actualizado con la nueva clase '{class_name}'.\n")
                self.text_area.see(tk.END)

    def validate_model(self):
        if not self.model:
            self.text_area.insert(tk.END, "Error: No hay modelo cargado. Selecciona un modelo primero.\n")
            self.text_area.see(tk.END)
            return
        threading.Thread(target=validate_model, args=(self.model, self.le, self.scaler, self.text_area), daemon=True).start()

    def list_models(self):
        models = glob.glob(os.path.join(BASE_DIR, 'best_model*.h5'))
        if not models:
            self.text_area.insert(tk.END, "No se encontraron modelos en el directorio.\n")
            self.text_area.see(tk.END)
            return
        
        model_window = tk.Toplevel(self.root)
        model_window.title("Seleccionar Modelo")
        model_window.configure(bg="#000000")
        model_window.geometry("400x300")
        
        tk.Label(model_window, text="Modelos disponibles:", bg="#000000", fg="#FFFFFF", font=("Arial", 12, "bold")).pack(pady=10)
        
        model_listbox = tk.Listbox(model_window, bg="#000000", fg="#FFFFFF", font=("Arial", 10), selectbackground="#1E3A8A")
        model_listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        for model_path in models:
            model_name = os.path.basename(model_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
            model_listbox.insert(tk.END, f"{model_name} (Modificado: {mod_time})")
        
        def select_model():
            selection = model_listbox.curselection()
            if selection:
                selected_model = model_listbox.get(selection[0]).split(' ')[0]
                self.model, self.le, self.scaler = load_model_and_classes(selected_model)
                if self.model:
                    self.current_model_name = selected_model
                    self.text_area.insert(tk.END, f"Modelo cargado: {selected_model}\n")
                    self.text_area.see(tk.END)
                else:
                    self.text_area.insert(tk.END, f"Error al cargar el modelo: {selected_model}\n")
                    self.text_area.see(tk.END)
            model_window.destroy()
        
        tk.Button(model_window, text="Seleccionar", command=select_model, **self.button_style).pack(pady=5)
        tk.Button(model_window, text="Cancelar", command=model_window.destroy, **self.button_style).pack(pady=5)

    def exit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()