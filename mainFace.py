from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
import uuid
import os

from faceMesh import segmentar_rostro 

app = FastAPI()

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Página principal
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")




@app.post("/procesar-imagen-local/")
async def procesar_imagen(file: UploadFile = File(...)):
    # Generar nombres únicos
    id_unico = str(uuid.uuid4())
    input_path = f"{id_unico}_input.jpg"
    output_path = f"{id_unico}_segmentado.jpg"
    pdf_path = f"{id_unico}_segmentado.pdf"

    # Guardar imagen temporal
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Llamar a la función de segmentación
    segmentar_rostro(input_path, output_path)

    # Crear PDF con la imagen segmentada
    pdf = FPDF()
    pdf.add_page()
    pdf.image(output_path, x=10, y=10, w=180)
    pdf.output(pdf_path)

    # Opcional: borrar archivos temporales si lo deseas luego

    return FileResponse(pdf_path, filename="resultado_segmentado.pdf", media_type="application/pdf")