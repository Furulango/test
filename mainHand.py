from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
import uuid
import os
import logging
from handsMesh import segmentar_manos  # Importar la función del archivo handsMesh.py

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

@app.post("/procesar-imagen-local/")
async def procesar_imagen_local(file: UploadFile = File(...)):
    """
    Endpoint para procesar imágenes de manos y generar PDF con el resultado
    """
    try:
        # Validar tipo de archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Generar nombres únicos
        id_unico = str(uuid.uuid4())
        input_path = f"{id_unico}_input.jpg"
        output_path = f"{id_unico}_segmentado.jpg"
        pdf_path = f"{id_unico}_segmentado.pdf"
        
        logger.info(f"Procesando imagen: {file.filename}")
        
        # Guardar imagen temporal
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Llamar a la función de segmentación de manos
        success = segmentar_manos(input_path, output_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error al procesar la imagen")
        
        # Verificar que la imagen de salida se haya creado
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="No se pudo generar la imagen segmentada")
        
        # Crear PDF con la imagen segmentada
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Resultado de Segmentación de Manos", ln=True, align='C')
        pdf.ln(10)
        
        # Agregar imagen al PDF (ajustar tamaño según sea necesario)
        pdf.image(output_path, x=10, y=30, w=180)
        pdf.output(pdf_path)
        
        # Limpiar archivo temporal de entrada
        if os.path.exists(input_path):
            os.remove(input_path)
        
        logger.info(f"Procesamiento completado para: {file.filename}")
        
        # Devolver el PDF y programar limpieza de archivos temporales
        return FileResponse(
            pdf_path, 
            filename="resultado_segmentado.pdf", 
            media_type="application/pdf",
            background=cleanup_files(output_path, pdf_path)
        )
        
    except Exception as e:
        logger.error(f"Error al procesar imagen: {str(e)}")
        
        # Limpiar archivos temporales en caso de error
        for temp_file in [input_path, output_path, pdf_path]:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
        
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

# Endpoint adicional para obtener información del servidor
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Servidor de segmentación de manos funcionando correctamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)