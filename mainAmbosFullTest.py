from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  
from fpdf import FPDF
import uuid
import os
import logging
import cv2
import mediapipe as mp
import numpy as np
from handsMesh import segmentar_manos
from faceMesh import segmentar_rostro

# ================== LIBRERÍAS Y CONFIGURACIÓN ==================
# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Definir zonas faciales
ZONAS_ROSTRO = {
    "Frente": [9, 336, 296, 334, 293, 301, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 71, 63, 105, 66, 107],
    "Mejilla Izquierda": [138, 215, 177, 137, 227, 111, 31, 228, 229, 230, 120, 47, 126, 209, 129, 203, 206, 216],
    "Mejilla Derecha": [367, 435, 401, 366, 447, 340, 261, 448, 449, 450, 349, 277, 355, 429, 358, 423, 426, 436],
    "Nariz": [2, 326, 328, 290, 392, 439, 278, 279, 420, 399, 419, 351, 168, 122, 196, 174, 198, 49, 48, 219, 64, 98, 97],
    "Ojo Izquierdo": [223, 222, 221, 189, 245, 128, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224],
    "Ojo Derecho": [443, 444, 445, 342, 446, 261, 448, 449, 450, 350, 357, 464, 413, 441, 442],
    "Boca": [17, 314, 405, 321, 375, 287, 409, 270, 269, 267, 0, 37, 39, 40, 185, 57, 146, 91, 181, 84],
    "Menton": [17, 314, 405, 321, 375, 287, 432, 434, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135, 214, 212, 57, 61, 146, 91, 181, 84],
    "Bozo": [2, 326, 328, 290, 327, 423, 426, 436, 410, 270, 269, 267, 0, 37, 39, 40, 185, 186, 216, 216, 206, 203, 98, 97]
}

def procesar_zonas_rostro(image, gray_image, landmarks, zonas):
    h, w = image.shape[:2]
    resultados = {}
    for nombre_zona, indices in zonas.items():
        puntos = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices])
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.fillPoly(mask, [puntos], 255)
        valores_pixeles = gray_image[mask == 255]
        promedio_intensidad = np.mean(valores_pixeles) if len(valores_pixeles) > 0 else 0
        resultados[nombre_zona] = promedio_intensidad
        cv2.polylines(image, [puntos], isClosed=True, color=(255, 0, 255), thickness=2)
    return resultados

def analizar_manos_por_zonas(image_path, output_path_con_numeros):
    """
    Analiza la imagen de manos segmentada, calcula la intensidad promedio
    de los píxeles en cada zona magenta y dibuja el número de la zona en la imagen.
    """
    try:
        imagen_original = cv2.imread(image_path)
        if imagen_original is None:
            logging.warning(f"No se pudo cargar la imagen para analizar manos: {image_path}")
            return None, False

        imagen_con_numeros = imagen_original.copy()
        imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV)

        lower_magenta = np.array([145, 170, 190])
        upper_magenta = np.array([155, 240, 255])

        mascara_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)
        kernel = np.ones((3, 3), np.uint8)
        mascara_magenta = cv2.morphologyEx(mascara_magenta, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mascara_magenta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        umbral_area = 10
        contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > umbral_area]
        contornos_ordenados = sorted(contornos_filtrados, key=cv2.contourArea, reverse=True)

        resultados_manos = {}
        for i, contorno in enumerate(contornos_ordenados):
            mascara_zona_actual = np.zeros_like(mascara_magenta)
            cv2.drawContours(mascara_zona_actual, [contorno], -1, 255, thickness=cv2.FILLED)
            
            zona_actual_pixeles_values = imagen_gris[np.where(mascara_zona_actual == 255)]
            
            promedio_gris = np.mean(zona_actual_pixeles_values) if len(zona_actual_pixeles_values) > 0 else 0
            resultados_manos[f"Zona Mano {i+1}"] = f"{promedio_gris:.2f}"

            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                texto = f"{i+1}"
                cv2.putText(imagen_con_numeros, texto, (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.imwrite(output_path_con_numeros, imagen_con_numeros)
        return resultados_manos, True

    except Exception as e:
        logging.error(f"Error al analizar las manos por zonas: {str(e)}")
        return None, False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ================== CONFIGURACIÓN DE CORS ==================
# << 2. BLOQUE AÑADIDO PARA HABILITAR CORS
origins = [
    "*"  # Permite cualquier origen. Para producción más segura, cambia esto a la URL de tu Vercel app.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

# ================== ENDPOINTS ==================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/procesar-imagen-rostro/")
async def procesar_imagen_rostro(file: UploadFile = File(...)):
    """
    Endpoint para procesar rostro con segmentación y análisis de intensidad por zonas
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        output_path = f"{id_unico}_rostro_segmentado.jpg"
        pdf_path = f"{id_unico}_rostro_segmentado.pdf"
        
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        image = cv2.imread(input_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        results = face_mesh.process(rgb_image)
        intensidades = {}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                intensidades = procesar_zonas_rostro(image, gray_image, face_landmarks, ZONAS_ROSTRO)
        
        cv2.imwrite(output_path, image)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Resultado de Segmentación de Rostro", ln=True, align='C')
        pdf.ln(10)
        pdf.image(output_path, x=10, y=30, w=180)
        pdf.ln(120)

        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Análisis de Intensidad de Píxeles por Zona:", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(90, 8, "Zona del Rostro", 1, 0, 'C')
        pdf.cell(90, 8, "Intensidad Promedio", 1, 1, 'C')

        pdf.set_font("Arial", size=10)
        for zona, promedio in intensidades.items():
            pdf.cell(90, 8, zona, 1, 0, 'L')
            pdf.cell(90, 8, f"{promedio:.2f}", 1, 1, 'C')

        pdf.output(pdf_path)

        if os.path.exists(input_path):
            os.remove(input_path)

        return FileResponse(
            pdf_path, 
            filename="resultado_rostro_segmentado.pdf", 
            media_type="application/pdf",
            background=cleanup_files(output_path, pdf_path)
        )

    except Exception as e:
        logger.error(f"Error al procesar rostro: {str(e)}")
        cleanup_temp_files(locals())
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/procesar-imagen-manos/")
async def procesar_imagen_manos(file: UploadFile = File(...)):
    """
    Endpoint para procesar solo manos
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        output_path = f"{id_unico}_manos_segmentado.jpg"
        pdf_path = f"{id_unico}_manos_segmentado.pdf"
        
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        success = segmentar_manos(input_path, output_path)
        if not success:
            raise HTTPException(status_code=500, detail="Error al procesar las manos")
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="No se pudo generar la imagen segmentada")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Resultado de Segmentación de Manos", ln=True, align='C')
        pdf.ln(10)
        pdf.image(output_path, x=10, y=30, w=180)
        pdf.output(pdf_path)
        
        if os.path.exists(input_path):
            os.remove(input_path)
        
        return FileResponse(
            pdf_path, 
            filename="resultado_manos_segmentado.pdf", 
            media_type="application/pdf",
            background=cleanup_files(output_path, pdf_path)
        )
        
    except Exception as e:
        logger.error(f"Error al procesar manos: {str(e)}")
        cleanup_temp_files(locals())
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/procesar-imagen-local/")
async def procesar_imagen_local(file: UploadFile = File(...)):
    return await procesar_imagen_completa(file)

@app.post("/procesar-imagen-completa/")
async def procesar_imagen_completa(file: UploadFile = File(...)):
    """
    Endpoint para procesar rostro + manos en una sola imagen y generar reporte completo.
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        rostro_output_path = f"{id_unico}_rostro_segmentado.jpg"
        manos_output_path = f"{id_unico}_manos_segmentado.jpg"
        manos_numeros_output_path = f"{id_unico}_manos_numerado.jpg"
        pdf_path = f"{id_unico}_completo_reporte.pdf"
        
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        rostro_success = segmentar_rostro(input_path, rostro_output_path)
        manos_success = segmentar_manos(input_path, manos_output_path)
        
        if not rostro_success and not manos_success:
            raise HTTPException(status_code=500, detail="No se pudo procesar ni rostro ni manos")

        intensidades_rostro = {}
        if rostro_success:
            image_rostro = cv2.imread(input_path)
            if image_rostro is not None:
                gray_image_rostro = cv2.cvtColor(image_rostro, cv2.COLOR_BGR2GRAY)
                rgb_image_rostro = cv2.cvtColor(gray_image_rostro, cv2.COLOR_GRAY2BGR)
                
                results = face_mesh.process(rgb_image_rostro)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        intensidades_rostro = procesar_zonas_rostro(image_rostro, gray_image_rostro, face_landmarks, ZONAS_ROSTRO)
                cv2.imwrite(rostro_output_path, image_rostro)

        intensidades_manos = {}
        if manos_success:
            intensidades_manos, _ = analizar_manos_por_zonas(manos_output_path, manos_numeros_output_path)

        pdf = FPDF()
        
        # Página 1: Todas las imágenes en una sola página
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Reporte de Segmentación y Análisis", ln=True, align='C')
        pdf.ln(5)
        
        # Configuración de posiciones y tamaños
        image_width = 90  # Ancho de cada imagen
        image_height = 60  # Alto de cada imagen
        start_x = 10
        current_y = 25
        
        # Contador de imágenes para organizar la disposición
        images_added = 0
        
        # Rostro segmentado
        if rostro_success and os.path.exists(rostro_output_path):
            # Título para la imagen de rostro
            pdf.set_font("Arial", style='B', size=10)
            pdf.set_xy(start_x, current_y)
            pdf.cell(image_width, 5, txt="Rostro Segmentado", ln=False, align='C')
            
            # Imagen de rostro
            pdf.image(rostro_output_path, x=start_x, y=current_y + 5, w=image_width, h=image_height)
            images_added += 1
        
        # Manos segmentadas (lado derecho si hay rostro, sino en la izquierda)
        if manos_success and os.path.exists(manos_output_path):
            x_pos = start_x + (image_width + 10) if images_added > 0 else start_x
            y_pos = current_y if images_added > 0 else current_y
            
            # Si ya hay una imagen en la fila, poner al lado; si no, nueva fila
            if images_added >= 2:
                y_pos = current_y + image_height + 15
                x_pos = start_x
                images_added = 0
            
            # Título para la imagen de manos
            pdf.set_font("Arial", style='B', size=10)
            pdf.set_xy(x_pos, y_pos)
            pdf.cell(image_width, 5, txt="Manos Segmentadas", ln=False, align='C')
            
            # Imagen de manos
            pdf.image(manos_output_path, x=x_pos, y=y_pos + 5, w=image_width, h=image_height)
            images_added += 1
            
            # Actualizar current_y para la siguiente imagen
            if images_added == 1:
                current_y = y_pos
        
        # Manos numeradas
        if manos_success and os.path.exists(manos_numeros_output_path):
            x_pos = start_x + (image_width + 10) if images_added == 1 else start_x
            y_pos = current_y if images_added == 1 else current_y + image_height + 15
            
            # Título para la imagen de manos numeradas
            pdf.set_font("Arial", style='B', size=10)
            pdf.set_xy(x_pos, y_pos)
            pdf.cell(image_width, 5, txt="Manos Numeradas", ln=False, align='C')
            
            # Imagen de manos numeradas
            pdf.image(manos_numeros_output_path, x=x_pos, y=y_pos + 5, w=image_width, h=image_height)

        # Página 2: Resumen de procesamiento
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Rostro procesado: {'Sí' if rostro_success else 'No'}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Manos procesadas: {'Sí' if manos_success else 'No'}", ln=True, align='L')
        pdf.ln(10)

        # Página 3: Tablas de análisis
        if intensidades_rostro or intensidades_manos:
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(200, 10, txt="Análisis de Intensidad de Píxeles", ln=True, align='C')
            pdf.ln(10)

            if intensidades_rostro:
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(0, 10, "Rostro:", ln=True)
                pdf.ln(2)
                pdf.set_font("Arial", style='B', size=10)
                pdf.cell(90, 8, "Zona del Rostro", 1, 0, 'C')
                pdf.cell(90, 8, "Intensidad Promedio", 1, 1, 'C')
                pdf.set_font("Arial", size=10)
                for zona, promedio in intensidades_rostro.items():
                    pdf.cell(90, 8, zona, 1, 0, 'L')
                    pdf.cell(90, 8, f"{promedio:.2f}", 1, 1, 'C')
                pdf.ln(10)

            if intensidades_manos:
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(0, 10, "Manos:", ln=True)
                pdf.ln(2)
                pdf.set_font("Arial", style='B', size=10)
                pdf.cell(90, 8, "Zona de la Mano", 1, 0, 'C')
                pdf.cell(90, 8, "Intensidad Promedio", 1, 1, 'C')
                pdf.set_font("Arial", size=10)
                
                # Dividir la tabla de manos en dos columnas si es muy larga
                items = list(intensidades_manos.items())
                mid_point = len(items) // 2 + len(items) % 2
                col1 = items[:mid_point]
                col2 = items[mid_point:]
                
                max_len = max(len(col1), len(col2))
                
                for i in range(max_len):
                    if i < len(col1):
                        zona1, prom1 = col1[i]
                        pdf.cell(45, 8, zona1, 1, 0, 'L')
                        pdf.cell(45, 8, str(prom1), 1, 0, 'C')
                    else:
                        pdf.cell(90, 8, "", 1, 0) # Celda vacía

                    if i < len(col2):
                        zona2, prom2 = col2[i]
                        pdf.cell(45, 8, zona2, 1, 0, 'L')
                        pdf.cell(45, 8, str(prom2), 1, 1, 'C')
                    else:
                        pdf.cell(90, 8, "", 1, 1) # Celda vacía

        pdf.output(pdf_path)
        
        cleanup_temp_files({
            'input_path': input_path,
            'rostro_output_path': rostro_output_path,
            'manos_output_path': manos_output_path,
            'manos_numeros_output_path': manos_numeros_output_path
        })
        
        return FileResponse(
            pdf_path, 
            filename="resultado_completo_reporte.pdf", 
            media_type="application/pdf",
            background=cleanup_files(pdf_path)
        )
        
    except Exception as e:
        logger.error(f"Error al procesar imagen completa: {str(e)}")
        cleanup_temp_files(locals())
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# ================== UTILIDADES ==================
async def cleanup_files(*file_paths):
    import asyncio
    await asyncio.sleep(5) # Aumentar tiempo de espera
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Archivo temporal eliminado: {file_path}")
        except Exception as e:
            logger.warning(f"No se pudo eliminar el archivo {file_path}: {str(e)}")

def cleanup_temp_files(local_vars):
    temp_files = ['input_path', 'output_path', 'rostro_output_path', 'manos_output_path', 'manos_numeros_output_path', 'pdf_path']
    for var_name in temp_files:
        if var_name in local_vars:
            file_path = local_vars.get(var_name)
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Archivo temporal eliminado por error: {file_path}")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar el archivo {file_path}: {str(e)}")

# ================== INFO ==================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Servidor de segmentación funcionando correctamente",
        "endpoints": {
            "rostro": "/procesar-imagen-rostro/",
            "manos": "/procesar-imagen-manos/",
            "completo": "/procesar-imagen-completa/"
        }
    }

@app.get("/info")
async def info():
    return {
        "name": "API de Segmentación Unificada",
        "version": "1.0.0",
        "description": "API para segmentación de rostro y manos con análisis de intensidades",
        "available_endpoints": [
            {"endpoint": "/procesar-imagen-rostro/", "method": "POST", "description": "Segmenta rostro y calcula intensidades"},
            {"endpoint": "/procesar-imagen-manos/", "method": "POST", "description": "Segmenta manos"},
            {"endpoint": "/procesar-imagen-completa/", "method": "POST", "description": "Segmenta rostro y manos en la misma imagen con análisis de intensidad y numeración de zonas"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
