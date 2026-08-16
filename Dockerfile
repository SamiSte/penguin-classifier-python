# Python-Basisimage passend zur lokalen Entwicklungsumgebung.
FROM python:3.12.13-slim-bookworm

# Arbeitsverzeichnis innerhalb des Containers.
WORKDIR /app

# Python soll Logs direkt ausgeben und keine .pyc-Dateien erzeugen.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Abhängigkeiten zuerst kopieren und installieren.
# Dadurch kann Docker diesen Schritt bei späteren Builds cachen.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Für die Anwendung benötigte Dateien kopieren.
COPY app.py .
COPY src ./src
COPY models ./models
COPY data/penguins.csv ./data/penguins.csv

# Dash-Port dokumentieren.
EXPOSE 8050

# Anwendung starten.
# app.py wird importiert, ohne den lokalen Spyder-Startblock auszuführen.
CMD ["python", "-c", "from app import app; app.run(host='0.0.0.0', port=8050, debug=False)"]