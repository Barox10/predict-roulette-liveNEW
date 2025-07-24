# main.py aggiornato per la Cloud Function (ora servizio Cloud Run)

import functions_framework
import numpy as np
import joblib
from google.cloud import storage
import os
import logging # Aggiunto per un logging più robusto
import traceback # Aggiunto per catturare stack trace completi degli errori

# Configura il logging. Questo indirizza i log a stderr, che Cloud Logging cattura.
# Puoi impostare level=logging.DEBUG per un output ancora più dettagliato durante il debug.
logging.basicConfig(level=logging.INFO)


# --- 1. CONFIGURAZIONE (verifica che il bucket sia corretto!) ---
# Sostituisci con il nome del tuo bucket Google Cloud Storage dove sono salvati i modelli
GCS_BUCKET_NAME = 'roulette-models-2025' # <--- VERIFICA CHE SIA CORRETTO!

# Numero di risultati top da considerare per ogni modello prima della post-elaborazione
PRED_NUMERI = 15

# Definizione della sequenza dei numeri sulla ruota della roulette (francese/europea)
# Questo è FONDAMENTALE per trovare i blocchi contigui
ROULETTE_WHEEL_SEQUENCE = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
    5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]
# Creiamo un mapping per trovare velocemente l'indice di un numero nella sequenza
ROULETTE_INDEX_MAP = {num: i for i, num in enumerate(ROULETTE_WHEEL_SEQUENCE)}


# --- 2. CARICAMENTO DEI MODELLI ---
# Questa parte carica i modelli dal GCS al momento dell'inizializzazione della funzione
# (cold start), evitando di caricarli ad ogni richiesta.

models = {}
model_names = [
    'logistic_regression',
    'random_forest',
    'lightgbm',
    'catboost'
]

# Inizializza il client di Google Cloud Storage fuori dalla funzione di richiesta
# per riutilizzarlo.
storage_client = storage.Client()

logging.info(f"Caricamento modelli dal bucket GCS: {GCS_BUCKET_NAME}...") # Modificato per usare logging

for name in model_names:
    model_filename = f'model_{name}.joblib'
    gcs_path = f'models/{model_filename}' # Percorso nel bucket
    local_path = f'/tmp/{model_filename}' # Percorso temporaneo locale nella Cloud Function

    try:
        bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        models[name] = joblib.load(local_path)
        logging.info(f"Modello {name} caricato con successo.") # Modificato per usare logging
    except Exception as e:
        logging.error(f"Errore durante il caricamento del modello {name} da GCS: {e}") # Modificato per usare logging.error
        logging.error(traceback.format_exc()) # Aggiunto per stampare lo stack trace completo
        models[name] = None # Imposta a None per indicare che non è disponibile


# --- 3. NUOVA LOGICA: FUNZIONE PER TROVARE BLOCCHI CONTIGUI ---
def find_contiguous_blocks(probabilities, num_blocks=3, block_size=5):
    """
    Trova i migliori blocchi di numeri contigui sulla ruota basandosi sulle probabilità.

    Args:
        probabilities (np.array): Array delle probabilità per ogni numero (0-36).
                                  L'indice dell'array corrisponde al numero.
        num_blocks (int): Numero di blocchi contigui da trovare.
        block_size (int): Dimensione di ogni blocco.

    Returns:
        list of lists: Una lista contenente i blocchi trovati,
                       dove ogni blocco è una lista di 5 numeri contigui.
    """
    wheel_len = len(ROULETTE_WHEEL_SEQUENCE)
    block_scores = [] # Per memorizzare il punteggio di ogni possibile blocco

    # Scansiona tutte le possibili posizioni di inizio blocco sulla ruota
    for start_index in range(wheel_len):
        current_block_numbers = []
        current_block_score = 0
        
        for i in range(block_size):
            # Calcola l'indice circolare per la ruota
            wheel_index = (start_index + i) % wheel_len
            number = ROULETTE_WHEEL_SEQUENCE[wheel_index]
            current_block_numbers.append(number)
            
            # Somma le probabilità dei numeri nel blocco per ottenere un punteggio
            # Assicurati che l'indice della probabilità esista (0-36)
            if 0 <= number <= 36:
                current_block_score += probabilities[number]
            # else: Gestire numeri fuori range se necessario, ma non dovrebbe accadere per roulette

        block_scores.append({
            'numbers': current_block_numbers,
            'score': current_block_score
        })

    # Ordina i blocchi per punteggio decrescente e prendi i migliori num_blocks
    sorted_blocks = sorted(block_scores, key=lambda x: x['score'], reverse=True)

    top_blocks = [block['numbers'] for block in sorted_blocks[:num_blocks]]

    return top_blocks


# --- 4. FUNZIONE PRINCIPALE DELLA CLOUD FUNCTION ---
@functions_framework.http
def predict_roulette(request):
    """
    Funzione Cloud per prevedere i numeri della roulette.
    Accetta in input gli ultimi 5 numeri usciti e restituisce i numeri previsti.
    """
    if request.method == 'OPTIONS':
        # Permetti richieste CORS preflight
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    request_json = request.get_json(silent=True)
    
    # Estrai 'last_5_numbers' dal payload JSON
    if request_json and 'last_5_numbers' in request_json:
        last_5_numbers = request_json['last_5_numbers']
        logging.info(f"Ricevuta richiesta con ultimi 5 numeri: {last_5_numbers}") # Modificato per usare logging
    else:
        logging.error('Errore: Il campo "last_5_numbers" è richiesto nel payload JSON.') # Modificato per usare logging.error
        return ('Errore: Il campo "last_5_numbers" è richiesto nel payload JSON.', 400, headers)

    if not isinstance(last_5_numbers, list) or len(last_5_numbers) != 5:
        logging.error('Errore: "last_5_numbers" deve essere una lista di 5 numeri.') # Modificato per usare logging.error
        return ('Errore: "last_5_numbers" deve essere una lista di 5 numeri.', 400, headers)

    # Converti la lista in un array NumPy con la forma corretta per i modelli (1 riga, 5 colonne)
    input_features = np.array(last_5_numbers).reshape(1, -1)

    all_predictions = {}

    for model_name, model in models.items():
        if model is None:
            all_predictions[model_name] = "Modello non disponibile"
            logging.warning(f"Modello {model_name} non disponibile, saltato.") # Aggiunto warning
            continue

        try:
            # Prevedi le probabilità di tutti i 37 numeri
            probabilities = model.predict_proba(input_features)[0] # Ottieni le probabilità per la singola previsione

            # --- NUOVA LOGICA: Trova i blocchi contigui ---
            top_contiguous_blocks = find_contiguous_blocks(probabilities, num_blocks=3, block_size=5)
            
            # L'output ora sarà una lista di blocchi
            all_predictions[model_name] = top_contiguous_blocks
            logging.info(f"Previsione {model_name} (blocchi contigui): {top_contiguous_blocks}") # Modificato per usare logging

        except Exception as e:
            all_predictions[model_name] = f"Errore durante la previsione: {e}"
            logging.error(f"Errore durante la previsione con {model_name}: {e}") # Modificato per usare logging.error
            logging.error(traceback.format_exc()) # Aggiunto per stampare lo stack trace completo

    return (all_predictions, 200, headers)
