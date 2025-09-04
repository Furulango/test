from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
import uuid
import os
import logging
from handsMesh import segmentar_manos  # Importar la función del archivo handsMesh.py
from faceMesh import segmentar_rostro   # Importar la función del archivo faceMesh.py

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Página principal
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/procesar-imagen-rostro/")
async def procesar_imagen_rostro(file: UploadFile = File(...)):
    """
    Endpoint para procesar solo rostro
    """
    try:
        # Validar tipo de archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Generar nombres únicos
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        output_path = f"{id_unico}_rostro_segmentado.jpg"
        pdf_path = f"{id_unico}_rostro_segmentado.pdf"
        
        logger.info(f"Procesando rostro en imagen: {file.filename}")
        
        # Guardar imagen temporal
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Llamar a la función de segmentación de rostro
        success = segmentar_rostro(input_path, output_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error al procesar el rostro")
        
        # Verificar que la imagen de salida se haya creado
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="No se pudo generar la imagen segmentada")
        
        # Crear PDF con la imagen segmentada
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Resultado de Segmentación de Rostro", ln=True, align='C')
        pdf.ln(10)
        
        # Agregar imagen al PDF
        pdf.image(output_path, x=10, y=30, w=180)
        pdf.output(pdf_path)
        
        # Limpiar archivo temporal de entrada
        if os.path.exists(input_path):
            os.remove(input_path)
        
        logger.info(f"Procesamiento de rostro completado para: {file.filename}")
        
        # Devolver el PDF y programar limpieza de archivos temporales
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
        # Validar tipo de archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Generar nombres únicos
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        output_path = f"{id_unico}_manos_segmentado.jpg"
        pdf_path = f"{id_unico}_manos_segmentado.pdf"
        
        logger.info(f"Procesando manos en imagen: {file.filename}")
        
        # Guardar imagen temporal
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Llamar a la función de segmentación de manos
        success = segmentar_manos(input_path, output_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error al procesar las manos")
        
        # Verificar que la imagen de salida se haya creado
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="No se pudo generar la imagen segmentada")
        
        # Crear PDF con la imagen segmentada
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Resultado de Segmentación de Manos", ln=True, align='C')
        pdf.ln(10)
        
        # Agregar imagen al PDF
        pdf.image(output_path, x=10, y=30, w=180)
        pdf.output(pdf_path)
        
        # Limpiar archivo temporal de entrada
        if os.path.exists(input_path):
            os.remove(input_path)
        
        logger.info(f"Procesamiento de manos completado para: {file.filename}")
        
        # Devolver el PDF y programar limpieza de archivos temporales
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
    """
    Endpoint de compatibilidad - redirige al procesamiento completo
    """
    return await procesar_imagen_completa(file)

@app.post("/procesar-imagen-completa/")
async def procesar_imagen_completa(file: UploadFile = File(...)):
    """
    Endpoint principal para procesar tanto rostro como manos en una sola imagen
    """
    try:
        # Validar tipo de archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Generar nombres únicos
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        rostro_output_path = f"{id_unico}_rostro_segmentado.jpg"
        manos_output_path = f"{id_unico}_manos_segmentado.jpg"
        pdf_path = f"{id_unico}_completo_segmentado.pdf"
        
        logger.info(f"Procesando imagen completa (rostro + manos): {file.filename}")
        
        # Guardar imagen temporal
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Procesar rostro
        rostro_success = segmentar_rostro(input_path, rostro_output_path)
        logger.info(f"Segmentación de rostro: {'exitosa' if rostro_success else 'fallida'}")
        
        # Procesar manos
        manos_success = segmentar_manos(input_path, manos_output_path)
        logger.info(f"Segmentación de manos: {'exitosa' if manos_success else 'fallida'}")
        
        # Verificar que al menos uno de los procesamientos fue exitoso
        if not rostro_success and not manos_success:
            raise HTTPException(status_code=500, detail="No se pudo procesar ni rostro ni manos")
        
        # Crear PDF con los resultados
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Resultado de Segmentación Completa", ln=True, align='C')
        pdf.ln(10)
        
        y_position = 30
        
        # Agregar resultado de rostro si fue exitoso
        if rostro_success and os.path.exists(rostro_output_path):
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Segmentación de Rostro:", ln=True, align='L')
            pdf.ln(5)
            pdf.image(rostro_output_path, x=10, y=y_position, w=90)
            y_position += 70
        
        # Agregar resultado de manos si fue exitoso
        if manos_success and os.path.exists(manos_output_path):
            if rostro_success:
                # Si ya hay rostro, poner manos al lado
                pdf.image(manos_output_path, x=105, y=30, w=90)
            else:
                # Si no hay rostro, poner manos en la posición principal
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Segmentación de Manos:", ln=True, align='L')
                pdf.ln(5)
                pdf.image(manos_output_path, x=10, y=y_position, w=90)
        
        # Agregar información del procesamiento
        pdf.ln(80)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Rostro procesado: {'Sí' if rostro_success else 'No'}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Manos procesadas: {'Sí' if manos_success else 'No'}", ln=True, align='L')
        
        pdf.output(pdf_path)
        
        # Limpiar archivos temporales
        cleanup_temp_files({
            'input_path': input_path,
            'rostro_output_path': rostro_output_path,
            'manos_output_path': manos_output_path
        })
        
        logger.info(f"Procesamiento completo exitoso para: {file.filename}")
        
        # Devolver el PDF y programar limpieza
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

async def cleanup_files(*file_paths):
    """
    Función para limpiar archivos temporales después de enviar la respuesta
    """
    import asyncio
    
    # Esperar un poco antes de limpiar para asegurar que el archivo se haya enviado
    await asyncio.sleep(2)
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Archivo temporal eliminado: {file_path}")
        except Exception as e:
            logger.warning(f"No se pudo eliminar el archivo {file_path}: {str(e)}")

def cleanup_temp_files(local_vars):
    """
    Función para limpiar archivos temporales en caso de error
    """
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

# Endpoints adicionales para información del servidor
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Servidor de segmentación unificado funcionando correctamente",
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
        "description": "API para segmentación de rostro y manos",
        "available_endpoints": [
            {
                "endpoint": "/procesar-imagen-rostro/",
                "method": "POST",
                "description": "Segmenta solo el rostro en la imagen"
            },
            {
                "endpoint": "/procesar-imagen-manos/",
                "method": "POST", 
                "description": "Segmenta solo las manos en la imagen"
            },
            {
                "endpoint": "/procesar-imagen-completa/",
                "method": "POST",
                "description": "Segmenta tanto rostro como manos en la misma imagen"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)