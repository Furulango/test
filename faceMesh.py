#Mediapipe Face Mesh desde la pagina web y la API
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

ZONAS = {
    "frente": [9, 336, 296, 334, 293, 301, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 71, 63, 105, 66, 107],  # Puntos de la frente
    "mejilla_izquierda": [138, 215, 177, 137, 227, 111, 31, 228, 229, 230, 120, 47, 126, 209, 129, 203, 206, 216],  # Puntos de la mejilla izquierda
    "mejilla_derecha": [367, 435, 401, 366, 447, 340, 261, 448, 449, 450, 349, 277, 355, 429, 358, 423, 426, 436],    # Puntos de la mejilla derecha
    "nariz": [2, 326, 328, 290, 392, 439, 278, 279, 420, 399, 419, 351, 168, 122, 196, 174, 198, 49, 48, 219, 64, 98, 97], # Puntos de la nariz
    "ojo_izquierdo": [223, 222, 221, 189, 245, 128, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224],  # Puntos del ojo izquierdo
    "ojo_derecho": [443, 444, 445, 342, 446, 261, 448, 449, 450, 350, 357, 464, 413, 441, 442],   # Puntos del ojo derecho
    "boca": [17, 314, 405, 321, 375, 287, 409, 270, 269, 267, 0, 37, 39, 40, 185, 57, 146, 91, 181, 84],  # Puntos de la boca
    "menton": [17, 314, 405, 321, 375, 287, 432, 434, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135, 214, 212, 57, 61, 146, 91, 181, 84],
    "bozo": [2, 326, 328, 290, 327, 423, 426, 436, 410, 270, 269, 267, 0, 37, 39, 40, 185, 186, 216, 216, 206, 203, 98, 97]
}

def segmentar_rostro(image_path: str, output_path: str = "imagen_segmentada.jpg") -> str:
    image = cv2.imread(image_path)
    
    # Preprocesamiento
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_eq = cv2.equalizeHist(gray_image)
    rgb_image = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2BGR)

    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            for nombre_zona, indices in ZONAS.items():
                puntos = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]
                cv2.polylines(image, [np.array(puntos)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(output_path, image)
    return output_path