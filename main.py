from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/procesar-imagen-local/")
async def procesar_imagen(file: UploadFile = File(...)):
    print("Recibí una imagen")
    contents = await file.read()
    with open("temp_img.png", "wb") as f:
        f.write(contents)

    image = Image.open("temp_img.png").convert("L")
    pdf_path = "imagen_procesada.pdf"
    image.save(pdf_path)

    return FileResponse(pdf_path, filename="imagen_procesada.pdf", media_type="application/pdf")
