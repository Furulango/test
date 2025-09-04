import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


def segmentar_manos(input_path, output_path):
    """
    Función para segmentar manos usando MediaPipe y OpenCV
    
    Args:
        input_path (str): Ruta de la imagen de entrada
        output_path (str): Ruta donde se guardará la imagen procesada
    """
    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)

    # Cargar la imagen
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen desde {input_path}")

    # Convertir a escala de grises y aplicar umbralización
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Crear copia de la imagen para mostrar resultados
    image_with_contours = image.copy()

    # Detectar manos en la imagen original
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Crear una máscara negra del mismo tamaño que la imagen
    mask = np.zeros_like(binary)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer coordenadas de los 21 puntos de la mano
            hand_points = np.array([(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) 
                                    for lm in hand_landmarks.landmark], np.int32)

            # Puntos clave para dividir falanges
            selected_points = [3, 4, 5, 6, 10, 14, 18, 7, 11, 15, 19]

            # Dibujar los puntos seleccionados sobre la imagen
            # for idx in selected_points:
            #     cv2.circle(image_with_contours, tuple(hand_points[idx]), 5, (0, 0, 255), -1)

            # Unir puntos con líneas para marcar las falanges dentro de la mano
            finger_sections = [(3, 6), (4, 7), (6, 10), (10, 14), (14, 18),
                               (7, 11), (11, 15), (15, 19)]
            
            def draw_limited_line(img, p1, p2, mask):
                """Dibuja una línea entre dos puntos, deteniéndose si sale de la máscara"""
                x1, y1 = hand_points[p1]
                x2, y2 = hand_points[p2]

                num_steps = 50
                for t in np.linspace(0, 1, num_steps):
                    xt = int(x1 * (1 - t) + x2 * t)
                    yt = int(y1 * (1 - t) + y2 * t)

                    # Verificar límites de la imagen
                    if xt < 0 or yt < 0 or xt >= image.shape[1] or yt >= image.shape[0]:
                        break

                    if mask[yt, xt] == 0:
                        break

                    if t > 0:
                        cv2.line(img, (prev_x, prev_y), (xt, yt), (255, 0, 255), 2)

                    prev_x, prev_y = xt, yt

            # Generar el contorno de la mano
            finger_tips = [4, 8, 12, 16, 20]
            palm_base = [0, 1, 5, 9, 13, 17]
            mid_fingers = [6, 10, 14, 18]

            contour_points = np.concatenate([hand_points[finger_tips], 
                                             hand_points[palm_base], 
                                             hand_points[mid_fingers]])

            hull = cv2.convexHull(contour_points, returnPoints=True)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Dilatar la máscara para mejorar la cobertura de la mano
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            # Crear masked_binary (máscara final)
            masked_binary = cv2.bitwise_and(binary, binary, mask=dilated_mask)

            # Dibujar landmarks en la imagen
            # mp_drawing.draw_landmarks(image_with_contours, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Dibujar las líneas de las falanges
            for (p1, p2) in finger_sections:
                if (p1, p2) == (4, 7):
                    draw_limited_line(image_with_contours, 7, 4, binary)
                else:
                    draw_limited_line(image_with_contours, p1, p2, binary)
                    draw_limited_line(image_with_contours, p2, p1, binary)

            def draw_reverse_line(img, p1, p2, color, mask):
                """Dibuja una línea desde p1 en sentido contrario a p2, deteniéndose si sale de la máscara"""
                x1, y1 = hand_points[p1]
                x2, y2 = hand_points[p2]

                if x2 != x1:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1

                    num_steps = 100
                    for t in np.linspace(0, -1, num_steps):
                        x_t = int(x1 * (1 - t) + x2 * t)
                        y_t = int(m * x_t + b)

                        if x_t < 0 or y_t < 0 or x_t >= image.shape[1] or y_t >= image.shape[0]:
                            break

                        if mask[y_t, x_t] == 0:
                            break
                    
                    cv2.line(img, (x1, y1), (x_t, y_t), color, 1)

            # Dibujar líneas en sentido contrario
            draw_reverse_line(image_with_contours, 19, 15, (255, 0, 255), dilated_mask)
            draw_reverse_line(image_with_contours, 18, 14, (255, 0, 255), dilated_mask)
            draw_reverse_line(image_with_contours, 3, 5, (255, 0, 255), dilated_mask)

            # Función para calcular la ecuación de la recta entre dos puntos
            def calculate_line_equation(p1, p2):
                x1, y1 = hand_points[p1]
                x2, y2 = hand_points[p2]
                if x2 == x1:
                    return None, None
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                return m, b

            def move_point_up(p, m, b, step=1):
                x, y = hand_points[p]
                y_new = y - step
                x_new = (y_new - b) / m if m != 0 else x
                return int(x_new), int(y_new)

            # Calcular las ecuaciones de las rectas
            m_5_6, b_5_6 = calculate_line_equation(5, 6)
            m_9_10, b_9_10 = calculate_line_equation(9, 10)
            m_17_18, b_17_18 = calculate_line_equation(17, 18)
            m_13_14, b_13_14 = calculate_line_equation(13, 14)

            step = 1
            max_steps = 500

            def is_line_out_of_mask(p1, p2, mask):
                x1, y1 = p1
                x2, y2 = p2

                if mask[y1, x1] == 0 or mask[y2, x2] == 0:
                    return True

                num_steps = 1000
                for t in np.linspace(0, 1, num_steps):
                    xt = int(x1 * (1 - t) + x2 * t)
                    yt = int(y1 * (1 - t) + y2 * t)

                    if xt < 0 or yt < 0 or xt >= mask.shape[1] or yt >= mask.shape[0]:
                        return True

                    if mask[yt, xt] == 0:
                        return True

                return False

            # Mover los puntos hacia arriba hasta tocar el borde
            for _ in range(max_steps):
                p5_new = move_point_up(5, m_5_6, b_5_6, step)
                p9_new = move_point_up(9, m_9_10, b_9_10, step)
                p17_new = move_point_up(17, m_17_18, b_17_18, step)
                p13_new = move_point_up(13, m_13_14, b_13_14, step)

                out_5_9 = is_line_out_of_mask(p5_new, p9_new, masked_binary)
                out_17_13 = is_line_out_of_mask(p17_new, p13_new, masked_binary)
                out_9_13 = is_line_out_of_mask(p9_new, p13_new, masked_binary)

                if out_5_9 and out_17_13 and out_9_13:
                    break

                if not out_5_9:
                    hand_points[5] = p5_new
                    hand_points[9] = p9_new
                if not out_17_13:
                    hand_points[17] = p17_new
                    hand_points[13] = p13_new
                if not out_9_13: 
                    hand_points[9] = p9_new
                    hand_points[13] = p13_new

            # Dibujar las líneas finales
            cv2.line(image_with_contours, tuple(hand_points[5]), tuple(hand_points[9]), (255, 0, 255), 2)
            cv2.line(image_with_contours, tuple(hand_points[17]), tuple(hand_points[13]), (255, 0, 255), 2)
            cv2.line(image_with_contours, tuple(hand_points[9]), tuple(hand_points[13]), (255, 0, 255), 2)

            # for idx in [5, 9, 17, 13]:
            #     cv2.circle(image_with_contours, tuple(hand_points[idx]), 5, (255, 0, 255), -1)

            def draw_reverse_line_from_5_to_9(img, color, mask):
                """Dibuja una línea desde el punto 5 en sentido contrario al punto 9"""
                x5, y5 = hand_points[5]
                x9, y9 = hand_points[9]

                if x9 != x5:
                    m = (y9 - y5) / (x9 - x5)
                    b = y5 - m * x5

                    num_steps = 1000
                    for t in np.linspace(0, -1, num_steps):
                        x_t = int(x5 * (1 - t) + x9 * t)
                        y_t = int(m * x_t + b)

                        if x_t < 0 or y_t < 0 or x_t >= image.shape[1] or y_t >= image.shape[0]:
                            break

                        if mask[y_t, x_t] == 0:
                            break
                    
                    cv2.line(img, (x5, y5), (x_t, y_t), color, 2)

            def draw_reverse_line_from_17_to_13(img, color, mask):
                """Dibuja una línea desde el punto 17 en sentido contrario al punto 13"""
                x17, y17 = hand_points[17]
                x13, y13 = hand_points[13]

                if x13 != x17:
                    m = (y13 - y17) / (x13 - x17)
                    b = y17 - m * x17

                    num_steps = 1000
                    for t in np.linspace(0, -1, num_steps):
                        x_t = int(x17 * (1 - t) + x13 * t)
                        y_t = int(m * x_t + b)

                        if x_t < 0 or y_t < 0 or x_t >= image.shape[1] or y_t >= image.shape[0]:
                            break

                        if mask[y_t, x_t] == 0:
                            break
                    
                    cv2.line(img, (x17, y17), (x_t, y_t), color, 1)

            # Dibujar líneas adicionales
            draw_reverse_line_from_5_to_9(image_with_contours, (255, 0, 255), masked_binary)
            draw_reverse_line_from_17_to_13(image_with_contours, (255, 0, 255), masked_binary)

            # Pulgar
            draw_limited_line(image_with_contours, 2, 6, masked_binary)
            draw_reverse_line(image_with_contours, 2, 6, (255, 0, 255), dilated_mask)

            # Encontrar contornos en la imagen procesada
            contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_with_contours, contours, -1, (255, 0, 255), 2)

    # Guardar la imagen procesada
    cv2.imwrite(output_path, image_with_contours)
    
    # Cerrar MediaPipe
    hands.close()
    
    return True