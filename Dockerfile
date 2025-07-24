# Usa l'immagine ufficiale Python 3.11 come base.
FROM python:3.11-slim-buster

# Installa la libreria di sistema mancante: libgomp1
# Questo risolve l'errore "OSError: libgomp.so.1: cannot open shared object file: No such file or directory"
RUN apt-get update && apt-get install -y libgomp1

# Imposta la directory di lavoro all'interno del container.
WORKDIR /app

# Copia il file requirements.txt nella directory di lavoro del container.
COPY requirements.txt .

# Installa le dipendenze Python elencate in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice della tua applicazione (incluso main.py) nel container.
COPY . .

# Il comando che verrà eseguito quando il container si avvia.
# Specifica la funzione 'predict_roulette' da eseguire sulla porta 8080.
CMD ["functions-framework", "--target", "predict_roulette", "--port", "8080"]
