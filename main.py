# main.py aggiornato per la Cloud Function (ora servizio Cloud Run)

import functions_framework
import numpy as np
import joblib
from google.cloud import storage
import os
import logging
import traceback
from collections import Counter # Aggiunto per il conteggio del consenso

# Configura il logging. Questo indirizza i log a stderr, che Cloud Logging cattura.
logging.basicConfig(level=logging.INFO)


# --- 1. CONFIGURAZIONE (verifica che il bucket sia corretto!) ---
# Sostituisci con il nome del tuo bucket Google Cloud Storage dove sono salvati i modelli
GCS_BUCKET_NAME = 'roulette-models-2025' # <--- VERIFICA CHE SIA CORRETTO!

# Numero di risultati top da considerare per ogni modello prima della post-elaborazione
PRED_NUMERI = 5 # MODIFICATO: Numero massimo di numeri da restituire nel consenso finale (da 15 a 5)

# Definizione della sequenza dei numeri sulla ruota della roulette (francese/europea)
ROULETTE_WHEEL_SEQUENCE = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
    5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]
# Creiamo un mapping per trovare velocemente l'indice di un numero nella sequenza (non direttamente usato nelle nuove features)
ROULETTE_INDEX_MAP = {num: i for i, num in enumerate(ROULETTE_WHEEL_SEQUENCE)}


# --- Funzione helper per ottenere tutti i blocchi a cui un numero appartiene ---
def _get_block_ids_for_number(number, roulette_wheel_sequence, block_size=5):
    """
    Dato un numero, restituisce una lista degli ID di tutti i blocchi contigui di `block_size`
    sulla ruota della roulette che contengono questo numero.
    L'ID del blocco 'i' corrisponde al blocco che inizia a roulette_wheel_sequence[i].
    """
    block_ids = []
    wheel_len = len(roulette_wheel_sequence)
    
    # Itera su tutte le possibili posizioni di inizio blocco sulla ruota
    for block_id in range(wheel_len):
        # Costruisci il blocco corrente (avvolgendo se necessario)
        current_block = []
        for i in range(block_size):
            current_block.append(roulette_wheel_sequence[(block_id + i) % wheel_len])
        
        # Se il numero è in questo blocco, aggiungi il suo block_id
        if number in current_block:
            block_ids.append(block_id)
    return block_ids

# --- Funzione per la preparazione delle caratteristiche (AGGIORNATA per l'appartenenza ai blocchi) ---
# Questa funzione converte una lista di numeri recenti (es. 5 numeri)
# in un vettore di 37 * 5 = 185 caratteristiche, codificando l'appartenenza ai blocchi per ogni posizione.
def _prepare_features(last_n_numbers, num_possible_outcomes=37, sequence_length=5, roulette_wheel_sequence=None):
    """
    Converte una lista di numeri in un vettore di caratteristiche per i modelli,
    codificando l'appartenenza ai blocchi per ciascun numero nella sequenza.
    Il vettore avrà `num_possible_outcomes * sequence_length` elementi (37 * 5 = 185).
    Ogni elemento `features[ (posizione * 37) + block_id ]` sarà 1 se il numero
    alla `posizione` appartiene a `block_id`.
    """
    features = np.zeros(num_possible_outcomes * sequence_length) 
    
    if roulette_wheel_sequence is None:
        # Questo controllo è cruciale se la funzione viene chiamata direttamente senza la sequenza della ruota
        logging.error("roulette_wheel_sequence deve essere fornita a _prepare_features per le features basate sui blocchi.")
        raise ValueError("roulette_wheel_sequence deve essere fornita a _prepare_features per le features basate sui blocchi.")

    for i, num in enumerate(last_n_numbers):
        if 0 <= num < num_possible_outcomes: # Assicurati che il numero sia nel range valido (0-36)
            # Ottieni tutti gli ID dei blocchi a cui questo numero appartiene
            block_ids_for_this_num = _get_block_ids_for_number(num, roulette_wheel_sequence, block_size=5)
            
            for block_id in block_ids_for_this_num:
                # Calcola l'indice nel vettore complessivo delle features:
                # (posizione del numero nella sequenza * numero totale di possibili ID di blocco) + ID del blocco
                feature_index = (i * num_possible_outcomes) + block_id # num_possible_outcomes è 37
                features[feature_index] = 1 # Imposta a 1 per indicare l'appartenenza
    
    return features.reshape(1, -1)


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

logging.info(f"Caricamento modelli dal bucket GCS: {GCS_BUCKET_NAME}...")

for name in model_names:
    model_filename = f'model_{name}.joblib'
    gcs_path = f'models/{model_filename}' # Percorso nel bucket
    local_path = f'/tmp/{model_filename}' # Percorso temporaneo locale nella Cloud Function

    try:
        bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        models[name] = joblib.load(local_path)
        logging.info(f"Modello {name} caricato con successo.")
    except Exception as e:
        logging.error(f"Errore durante il caricamento del modello {name} da GCS: {e}")
        logging.error(traceback.format_exc())
        models[name] = None


# --- 3. LOGICA PER TROVARE BLOCCHI CONTIGUI (rimane la stessa, ora applicata alle probabilità dei numeri) ---
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
                       dove ogni blocco è una lista di `block_size` numeri contigui.
    """
    wheel_len = len(ROULETTE_WHEEL_SEQUENCE)
    block_scores = [] # Per memorizzare il punteggio di ogni possibile blocco

    for start_index in range(wheel_len):
        current_block_numbers = []
        current_block_score = 0 # Inizializza il punteggio per ogni nuovo blocco
        
        for i in range(block_size):
            # Calcola l'indice circolare per la ruota
            wheel_index = (start_index + i) % wheel_len
            number = ROULETTE_WHEEL_SEQUENCE[wheel_index]
            current_block_numbers.append(number)
            
            if 0 <= number <= 36:
                current_block_score += probabilities[number]

        block_scores.append({
            'numbers': current_block_numbers,
            'score': current_block_score
        })

    # Ordina i blocchi per punteggio decrescente e prendi i migliori num_blocks
    sorted_blocks = sorted(block_scores, key=lambda x: x['score'], reverse=True)

    top_blocks = [block['numbers'] for block in sorted_blocks[:num_blocks]]

    return top_blocks

# --- 4. NUOVA LOGICA: PREVISIONE DI CONSENSO ---
def _get_consensus_predictions(all_model_blocks, min_consensus_models=4, max_predictions=5): # MODIFICATO: min_consensus_models=4, max_predictions=5
    """
    Determina le previsioni di consenso basate sull'accordo tra più modelli.
    
    Args:
        all_model_blocks (dict): Dizionario in cui le chiavi sono i nomi dei modelli e i valori
                                 sono liste di blocchi previsti (es. {'model_A': [[n1,n2..], [n3,n4..]], ...}).
        min_consensus_models (int): Numero minimo di modelli che devono prevedere un numero
                                    affinché sia considerato nel consenso.
        max_predictions (int): Numero massimo di numeri di consenso da restituire.
                                Se più di `max_predictions` numeri hanno consenso, vengono prioritari per frequenza.

    Returns:
        list: Una lista di numeri unici che hanno raggiunto la soglia di consenso,
              ordinati per frequenza tra i modelli, poi per numero.
    """
    number_model_votes = Counter() # Conta quanti MODELLI diversi hanno predetto ciascun numero
    
    # Itera sui blocchi previsti da ciascun modello
    for model_name, predicted_blocks in all_model_blocks.items():
        if isinstance(predicted_blocks, str): # Salta se il modello ha restituito una stringa di errore
            continue
        
        # Raccogli tutti i numeri unici da QUESTI blocchi del modello corrente
        model_unique_numbers_in_blocks = set()
        for block in predicted_blocks:
            model_unique_numbers_in_blocks.update(block)
        
        # Per ogni numero unico predetto da questo modello, aggiungi un "voto" al numero
        for num in model_unique_numbers_in_blocks:
            number_model_votes[num] += 1
            
    consensus_numbers = []
    # Filtra i numeri che soddisfano la soglia minima di consenso
    for num, votes in number_model_votes.items():
        if votes >= min_consensus_models:
            consensus_numbers.append((num, votes)) # Memorizza (numero, voti)
            
    # Ordina i numeri di consenso: prima per voti (decrescente), poi per numero (crescente)
    consensus_numbers.sort(key=lambda x: (-x[1], x[0]))
    
    # Estrai solo i numeri e prendi i primi max_predictions
    final_predictions = [num for num, votes in consensus_numbers[:max_predictions]]
    
    return final_predictions


# --- 5. FUNZIONE PRINCIPALE DELLA CLOUD FUNCTION ---
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
        logging.info(f"Ricevuta richiesta con ultimi 5 numeri: {last_5_numbers}")
    else:
        logging.error('Errore: Il campo "last_5_numbers" è richiesto nel payload JSON.')
        return ('Errore: Il campo "last_5_numbers" è richiesto nel payload JSON.', 400, headers)

    if not isinstance(last_5_numbers, list) or len(last_5_numbers) != 5:
        logging.error('Errore: "last_5_numbers" deve essere una lista di 5 numeri.')
        return ('Errore: "last_5_numbers" deve essere una lista di 5 numeri.', 400, headers)

    # Converti la lista di 5 numeri in un array NumPy con 185 caratteristiche (nuovo formato)
    # Passa ROULETTE_WHEEL_SEQUENCE alla funzione di preparazione delle features
    try:
        input_features = _prepare_features(last_5_numbers, 
                                           num_possible_outcomes=37, 
                                           sequence_length=5, 
                                           roulette_wheel_sequence=ROULETTE_WHEEL_SEQUENCE)
    except ValueError as e:
        logging.error(f"Errore durante la preparazione delle features: {e}")
        return (f"Errore durante la preparazione delle features: {e}", 400, headers)


    all_model_predictions_blocks = {} # Memorizza le previsioni dei blocchi da ciascun modello

    for model_name, model in models.items():
        if model is None:
            all_model_predictions_blocks[model_name] = "Modello non disponibile"
            logging.warning(f"Modello {model_name} non disponibile, saltato.")
            continue

        try:
            # Prevedi le probabilità di tutti i 37 numeri
            probabilities = model.predict_proba(input_features)[0] # Ottieni le probabilità per la singola previsione

            # Trova i migliori blocchi contigui basati su queste probabilità
            top_contiguous_blocks = find_contiguous_blocks(probabilities, num_blocks=3, block_size=5)
            
            # Memorizza i blocchi previsti per questo modello
            all_model_predictions_blocks[model_name] = top_contiguous_blocks
            logging.info(f"Previsione {model_name} (blocchi contigui): {top_contiguous_blocks}")

        except Exception as e:
            all_model_predictions_blocks[model_name] = f"Errore durante la previsione: {e}"
            logging.error(f"Errore durante la previsione con {model_name}: {e}")
            logging.error(traceback.format_exc())

    # --- Applica la Logica di Consenso ---
    consensus_final_numbers = _get_consensus_predictions(all_model_predictions_blocks, 
                                                         min_consensus_models=4, # MODIFICATO: min_consensus_models=4
                                                         max_predictions=PRED_NUMERI) 

    # La Cloud Function ora restituirà un dizionario con le previsioni dei singoli modelli
    # E un nuovo campo per le previsioni di consenso.
    response_data = {
        'logistic_regression': all_model_predictions_blocks.get('logistic_regression', "Modello non disponibile"),
        'random_forest': all_model_predictions_blocks.get('random_forest', "Modello non disponibile"),
        'lightgbm': all_model_predictions_blocks.get('lightgbm', "Modello non disponibile"),
        'catboost': all_model_predictions_blocks.get('catboost', "Modello non disponibile"),
        'consensus_predictions': consensus_final_numbers # Nuovo campo per il consenso
    }

    return (response_data, 200, headers)
