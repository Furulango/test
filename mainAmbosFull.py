from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

def analizar_manos_por_zonas(image_path):
    """
    Analiza la imagen de manos segmentada y calcula la intensidad promedio
    de los píxeles en cada zona magenta.
    """
    try:
        imagen = cv2.imread(image_path)
        if imagen is None:
            logging.warning(f"No se pudo cargar la imagen para analizar manos: {image_path}")
            return None

        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

        lower_magenta = np.array([145, 170, 190])
        upper_magenta = np.array([155, 240, 255])

        mascara_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)
        kernel = np.ones((3, 3), np.uint8)
        mascara_magenta = cv2.morphologyEx(mascara_magenta, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mascara_magenta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        umbral_area = 28
        contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > umbral_area]
        contornos_ordenados = sorted(contornos_filtrados, key=cv2.contourArea, reverse=True)

        resultados_manos = {}
        for i, contorno in enumerate(contornos_ordenados):
            mascara_zona_actual = np.zeros_like(mascara_magenta)
            cv2.drawContours(mascara_zona_actual, [contorno], -1, 255, thickness=cv2.FILLED)
            zona_actual_pixeles = imagen_gris[np.where(mascara_zona_actual == 255)]
            promedio_gris = np.mean(zona_actual_pixeles) if len(zona_actual_pixeles) > 0 else 0
            resultados_manos[f"Zona Mano {i+1}"] = promedio_gris

        return resultados_manos
    except Exception as e:
        logging.error(f"Error al analizar las manos por zonas: {str(e)}")
        return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# ================== ENDPOINTS ==================

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

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
    Endpoint para procesar rostro + manos en una sola imagen y generar reporte con análisis de intensidad
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        rostro_output_path = f"{id_unico}_rostro_segmentado.jpg"
        manos_output_path = f"{id_unico}_manos_segmentado.jpg"
        pdf_path = f"{id_unico}_completo_segmentado.pdf"
        
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Primero, segmentamos ambas partes
        rostro_success = segmentar_rostro(input_path, rostro_output_path)
        manos_success = segmentar_manos(input_path, manos_output_path)
        
        if not rostro_success and not manos_success:
            raise HTTPException(status_code=500, detail="No se pudo procesar ni rostro ni manos")

        # Luego, hacemos el análisis de intensidad de píxeles
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
            # AHORA SÍ ANALIZAMOS LA IMAGEN DE SALIDA DE LA SEGMENTACIÓN DE MANOS
            intensidades_manos = analizar_manos_por_zonas(manos_output_path)

        pdf = FPDF()
        
        # Página 1: Segmentación visual
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Resultado de Segmentación Completa", ln=True, align='C')
        pdf.ln(10)
        
        y_position = 30
        if rostro_success and os.path.exists(rostro_output_path):
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Segmentación de Rostro:", ln=True, align='L')
            pdf.ln(5)
            pdf.image(rostro_output_path, x=10, y=y_position, w=90)
            y_position += 70
        
        if manos_success and os.path.exists(manos_output_path):
            if rostro_success:
                pdf.image(manos_output_path, x=105, y=30, w=90)
            else:
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Segmentación de Manos:", ln=True, align='L')
                pdf.ln(5)
                pdf.image(manos_output_path, x=10, y=y_position, w=90)
        
        pdf.ln(80)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Rostro procesado: {'Sí' if rostro_success else 'No'}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Manos procesadas: {'Sí' if manos_success else 'No'}", ln=True, align='L')

        # Página 2: Análisis de intensidad
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
                for zona, promedio in intensidades_manos.items():
                    pdf.cell(90, 8, zona, 1, 0, 'L')
                    pdf.cell(90, 8, f"{promedio:.2f}", 1, 1, 'C')

        pdf.output(pdf_path)
        
        cleanup_temp_files({
            'input_path': input_path,
            'rostro_output_path': rostro_output_path,
            'manos_output_path': manos_output_path
        })
        
        return FileResponse(
            pdf_path, 
            filename="resultado_completo_segmentado.pdf", 
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
    await asyncio.sleep(2)
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Archivo temporal eliminado: {file_path}")
        except Exception as e:
            logger.warning(f"No se pudo eliminar el archivo {file_path}: {str(e)}")

def cleanup_temp_files(local_vars):
    temp_files = ['input_path', 'output_path', 'rostro_output_path', 'manos_output_path', 'pdf_path']
    for var_name in temp_files:
        if var_name in local_vars:
            file_path = local_vars[var_name]
            try:
                if os.path.exists(file_path):
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
            {"endpoint": "/procesar-imagen-completa/", "method": "POST", "description": "Segmenta rostro y manos en la misma imagen"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)