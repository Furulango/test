# Imagen base de Python
FROM python:3.10-slim

# Instala las dependencias del sistema operativo para OpenCV Headless
RUN apt-get update && apt-get install -y libgl1 libgthread-2.0-0

# Crear directorio de trabajo
WORKDIR /app

# Copiar el archivo de dependencias de Python
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de tu código de la aplicación
COPY . .

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8080

# Comando de inicio para producción
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "mainAmbosFullTest:app", "--bind", "0.0.0.0:8080"]

