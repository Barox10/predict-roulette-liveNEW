# Usa l'immagine ufficiale Python 3.11 basata su Debian Bookworm (versione 12).
# 'slim-bookworm' è la versione più recente di Debian "slim".
FROM python:3.11-slim-bookworm

# Installa la libreria di sistema mancante: libgomp1
# Questo risolve l'errore "OSError: libgomp.so.1: cannot open shared object file: No such file or directory"
# Esegui apt-get clean per pulire la cache dei pacchetti e ridurre le dimensioni dell'immagine finale.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

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
