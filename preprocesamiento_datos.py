import cv2
import mediapipe as mp
import numpy as np
import os
import time
import subprocess

# Ruta absoluta para el directorio de datos
DATA_DIR = r'C:\Users\labin\OneDrive\Desktop\proyecto_final_version_2_mediapipe_cnn_svm\sign_data2'

# Configuración de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Función para capturar imágenes de una clase
def capture_images(class_name, num_images=100):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return False

    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f"Capturando {num_images} imágenes para la clase '{class_name}'. Asegúrate de que la mano sea detectada (landmarks visibles). Presiona 's' para empezar.")
    
    # Mostrar video con landmarks en tiempo real
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            cap.release()
            return False
        
        # Procesar frame para detectar manos
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Dibujar landmarks si se detecta una mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Mano detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No se detecta mano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Captura', frame)
        
        # Iniciar captura al presionar 's' solo si hay detección
        if cv2.waitKey(1) & 0xFF == ord('s') and results.multi_hand_landmarks:
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            print("No se detectó una mano. Ajusta la posición de la mano y vuelve a presionar 's'.")

    # Capturar imágenes solo si se detecta una mano
    captured = 0
    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break
        
        # Verificar detección de mano
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            image_path = os.path.join(class_dir, f"{captured}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Imagen {captured+1}/{num_images} guardada para '{class_name}'.")
            captured += 1
            time.sleep(0.5)  # Pausa para permitir ajustar la mano
        else:
            print(f"No se detectó mano en el frame. Intentando de nuevo...")

        # Mostrar frame con landmarks durante la captura
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Captura', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Revisa esta carpeta para tus imágenes: {os.path.abspath(class_dir)}")
    
    os.startfile(os.path.abspath(class_dir))  # Abre la carpeta automáticamente
    return captured == num_images



# Función para extraer landmarks de una imagen
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
        return landmarks_array
    return None

# Función para procesar todas las imágenes de una clase
def process_class(class_name):
    class_dir = os.path.join(DATA_DIR, class_name)
    landmarks_list = []
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        landmarks = extract_landmarks(image_path)
        if landmarks is not None:
            landmarks_list.append(landmarks)
        else:
            print(f"No se detectaron landmarks en {image_file}")
    return np.array(landmarks_list)

# Función principal
def main():
    # Crear directorio principal si no existe
    os.makedirs(DATA_DIR, exist_ok=True)

    # Pedir número de clases
    num_classes = int(input("¿Cuántas clases vas a capturar?: "))

    # Procesar cada clase
    for _ in range(num_classes):
        class_name = input("Nombre de la clase: ")
        
        # Capturar imágenes
        if not capture_images(class_name):
            print(f"Error al capturar imágenes para '{class_name}'. Abortando.")
            continue
        
        # Extraer y guardar landmarks
        landmarks = process_class(class_name)
        if len(landmarks) > 0:
            np.save(os.path.join(DATA_DIR, f"{class_name}_landmarks.npy"), landmarks)
            print(f"Landmarks para '{class_name}' procesados y guardados ({len(landmarks)} imágenes con landmarks).")
        else:
            print(f"No se encontraron landmarks para '{class_name}'.")
    
    print("¡Captura y procesamiento completados para todas las clases!")


if __name__ == '__main__':
    main()