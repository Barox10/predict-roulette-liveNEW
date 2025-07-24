# Usa l'immagine ufficiale Python 3.11 come base.
# 'slim-buster' è una versione più leggera per ridurre le dimensioni dell'immagine.
FROM python:3.11-slim-buster

# Imposta la directory di lavoro all'interno del container.
# Tutti i comandi successivi verranno eseguiti da /app.
WORKDIR /app

# Copia il file requirements.txt nella directory di lavoro del container.
COPY requirements.txt .

# Installa le dipendenze Python elencate in requirements.txt.
# --no-cache-dir evita di memorizzare nella cache i pacchetti scaricati, riducendo le dimensioni finali dell'immagine.
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice della tua applicazione (incluso main.py) nel container.
COPY . .

# Il comando che verrà eseguito quando il container si avvia.
# functions-framework è la libreria che trasforma il tuo codice Python in un server HTTP.
# --target predict_roulette specifica il nome della funzione da eseguire (quella con @functions_framework.http).
# --port 8080 dice al server di ascoltare sulla porta 8080, che è la porta predefinita di Cloud Run.
CMD ["functions-framework", "--target", "predict_roulette", "--port", "8080"]
