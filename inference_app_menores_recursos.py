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
BASE_DIR = r'C:\Users\cfppu\OneDrive\Escritorio\proyecto_final_version_2_mediapipe_cnn_svm'
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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
        return landmarks_array, True
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        text_area.insert(tk.END, "Error: No se puede acceder a la cámara.\n")
        return np.array([])
    
    # Reducir resolución de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
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

# Función para entrenar incrementalmente con todos los datos
def incremental_training(model, le, new_class_name, new_landmarks, scaler, text_area):
    if new_landmarks.size == 0:
        text_area.insert(tk.END, "No hay datos para entrenar la nueva clase.\n")
        return model, le

    # Cargar datos de todas las clases
    X, y = [], []
    old_classes = le.classes_.tolist()
    if new_class_name in old_classes:
        text_area.insert(tk.END, f"Error: La clase '{new_class_name}' ya existe.\n")
        return model, le
    
    all_classes = old_classes + [new_class_name]
    le.classes_ = np.array(all_classes)
    
    # Cargar datos de clases existentes (A-Z)
    for class_name in old_classes:
        landmarks_path = os.path.join(DATA_DIR, f"{class_name}_landmarks.npy")
        if os.path.exists(landmarks_path):
            landmarks = np.load(landmarks_path)
            X.append(landmarks)
            y.extend([class_name] * landmarks.shape[0])
        else:
            text_area.insert(tk.END, f"Advertencia: No se encontró {landmarks_path}\n")
    
    # Añadir datos de la nueva clase
    X.append(new_landmarks)
    y.extend([new_class_name] * new_landmarks.shape[0])
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Normalizar coordenadas z
    if scaler is not None:
        z_indices = np.arange(2, X.shape[1], 3)
        X[:, z_indices] = scaler.transform(X[:, z_indices])
    
    # Codificar etiquetas
    y_encoded = le.transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(all_classes))
    
    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
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
    new_model.fit(X_train, y_train,
                  batch_size=32,
                  epochs=500,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks)
    
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
    
    X, y = [], []
    classes = le.classes_
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
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(classes))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    text_area.insert(tk.END, f"Resultados de validación:\n")
    text_area.insert(tk.END, f"  Pérdida: {val_loss:.4f}\n")
    text_area.insert(tk.END, f"  Precisión: {val_accuracy:.4f}\n")
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
        
        # Canvas para video (reducido para mejor rendimiento)
        self.canvas = tk.Canvas(self.main_frame, width=320, height=240, bg="#000000", highlightthickness=2, highlightbackground="#1E3A8A")
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
            self.running = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.text_area.insert(tk.END, "Error: No se puede acceder a la cámara.\n")
                self.running = False
                return
            # Reducir resolución de la cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.text_area.insert(tk.END, f"Iniciando inferencia con {self.current_model_name}...\n")
            threading.Thread(target=self.inference_loop, daemon=True).start()

    def stop_inference(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.text_area.insert(tk.END, "Inferencia detenida.\n")
            self.text_area.see(tk.END)
            self.canvas.delete("all")
            self.canvas.create_text(160, 120, text="Inferencia detenida", fill="#FFFFFF", font=("Arial", 12))

    def inference_loop(self):
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((320, 240), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
            self.root.update()
            time.sleep(0.033)  # Limitar a ~30 FPS
        
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        self.text_area.insert(tk.END, "Inferencia detenida.\n")
        self.text_area.see(tk.END)
        self.canvas.delete("all")
        self.canvas.create_text(160, 120, text="Inferencia detenida", fill="#FFFFFF", font=("Arial", 12))

    def add_class(self):
        if not self.model:
            self.text_area.insert(tk.END, "Error: No hay modelo cargado. Selecciona un modelo primero.\n")
            self.text_area.see(tk.END)
            return
        class_name = simpledialog.askstring("Input", "Nombre de la nueva clase:", parent=self.root)
        if class_name:
            new_landmarks = capture_new_class_data(class_name, num_images=100, scaler=self.scaler, text_area=self.text_area)
            if new_landmarks.size > 0:
                self.model, self.le = incremental_training(self.model, self.le, class_name, new_landmarks, self.scaler, self.text_area)
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