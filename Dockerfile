# Imagen base de Python
FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias si las tienes
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de tu código
COPY . .

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "mainAmbosFullTest:app", "--host", "0.0.0.0", "--port", "8080"]
