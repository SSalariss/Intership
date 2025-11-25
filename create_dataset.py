
#create_dataset.py
"""
Preparazione del dataset con 20 mila chunk, da 2048 byte
"""
import random
import string
import numpy as np
from tqdm import tqdm
import os
import pickle

from sklearn.model_selection import train_test_split


# Config generica per le classi del dataset

CLASS_CONFIG = {
    'enc': {
        'dir': './data/enc',
        'ext': '.enc',
        'label': 0,
        'name': 'ENC',
    },
    'other': {
        'dir': './data/png',   # ./data/png o ./data/mp3
        'ext': '.png',         #'.png' o '.mp3'
        'label': 1,
        'name': 'PNG',         # 'PNG' o 'MP3'
    }
}

# Configurazione
CONFIG = {
    'chunk_size': 2048,
    'total_chunks': 20000, # i chunk da estrarre
    'test_size': 0.2, # 20% per testing, 80% per il training
    'seed': 42,
    'output_dir': './dataset/png'
}

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# Estrazione Chunk

def extract_chunks_from_files(
        file_paths,
        labels,
        num_chunks,
        chunk_size,
):
    """
    Estrae chunk casuali dai file mantenendo il bilanciamento delle classi
    Esempio con PDF:
    Args:
        - Lista di path ai file
        - Lista di Label (0 = ENC, 1 = PDF)
        - num_chunks: numero totale di chunk da estrarre
        - chunks_size: dimensione del chunk in byte

    Returns: 
        - lista di chunks 
        - chunk_labels: lista di label per ogni chunks
    """
    other_name = CLASS_CONFIG['other']['name']

    #separa i file in classi

    enc_files = [f for f, l in zip(file_paths, labels) if l == 0] # zip è una funzione che ritorna un iteratore di tuple
    other_files = [f for f, l in zip(file_paths, labels) if l == 1]

    #calcola quanti chunk per classe per il bilanciamento del dataset
    chunks_per_class = num_chunks // 2

    print(f"\nEstrazione {chunks_per_class} chunk per classe...")
    print(f"File ENC disponibili: {len(enc_files)}")
    print(f"File {other_name} disponibili: {len(other_files)}")

    def extract_from_class(files, num_chunks, label_value):
        chunks = []
        labels = []
        chunks_per_file = max(1, num_chunks // len(files))

        pbar = tqdm(files, desc=f"Estrazione {'ENC'if label_value == 0 else other_name}")
        for file_path in pbar:
            try:
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                file_size = len(file_data)

                if file_size < chunk_size:
                    # Vuol dire che il file è troppo piccolo, pad con zeri
                    chunk = file_data + b'\x00' * (chunk_size - file_size)
                    chunks.append(chunk)
                    labels.append(label_value)
                else:
                    # Estrai chunk casuali
                    for _ in range(chunks_per_file):
                        if len(chunks) >= num_chunks:
                            break
                        max_start = file_size - chunk_size
                        start = random.randint(0, max_start)
                        chunk = file_data[start:start + chunk_size]
                        chunks.append(chunk)
                        labels.append(label_value)
                #se abbiamo raggiunto il numero di chunk esci
                if len(chunks) >= num_chunks:
                    break
            except Exception as e: # lancia un errore se hai problemi nell'apertura
                print(f"\nErrore nel caricare {file_path}: {e}")

        return chunks[:num_chunks], labels[:num_chunks] # prendiamo i primi num_chunks dalle due liste

    # Chiamiamo la classe appena creata per l'estrazione dei chunk per entrambe le classi
    enc_chunks, enc_labels = extract_from_class(enc_files, chunks_per_class, 0)
    other_chunks, other_labels = extract_from_class(other_files, chunks_per_class, 1)

    # e comenciamo tutto insieme
    all_chunks = enc_chunks + other_chunks
    all_labels = enc_labels + other_labels

    #loggin: stamoa il numero totale di chunk per ogni classe
    print(f"\nTotale chunk estratti: {len(all_chunks)}")
    print(f"  ENC: {sum(1 for l in all_labels if l == 0)}")
    print(f"  {other_name}: {sum(1 for l in all_labels if l == 1)}")

    return all_chunks, all_labels
# Salvataggio del dataset

def save_dataset(chunks, labels, output_dir, test_size, seed):
    # Salva il dataset in train e test

    # Crea directory per l'output
    os.makedirs(output_dir, exist_ok=True)

    # Split train/test
    indices = list(range(len(chunks))) # creaiamo una lista di indici
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    # Preparazione dei dati
    train_chunks = [chunks[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_chunks = [chunks[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f"\n== Split del dataset...")
    print(f"Train: {len(train_chunks)} chunk ({sum(train_labels)} {other_name}, {len(train_labels)-sum(train_labels)} ENC)")
    print(f"Test: {len(test_chunks)} chunk ({sum(test_labels)} {other_name}, {len(test_labels)-sum(test_labels)} ENC)")

    train_data = {
        'chunks': train_chunks,
        'labels': train_labels
    }

    test_data={
        'chunks': test_chunks,
        'labels': test_labels
    }

    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    # statistiche
    stats = {
        'total_chunks': len(chunks),
        'train_size': len(train_chunks),
        'test_size': len(test_chunks),
        'chunk_size': len(chunks[0]),
        'num_classes': 2,
        'class_names': ['ENC', other_name]
    }
    with open(os.path.join(output_dir, 'dataset_info.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nDataset salvato in '{output_dir}/'")
    print(f"  - train_data.pkl ({len(train_chunks)} chunk)")
    print(f"  - test_data.pkl ({len(test_chunks)} chunk)")
    print(f"  - dataset_info.pkl (metadata)")

# Lancio del main
def main():
    
    ENC_DIR = CLASS_CONFIG['enc']['dir']
    ENC_EXT = CLASS_CONFIG['enc']['ext']

    OTHER_DIR = CLASS_CONFIG['other']['dir']
    OTHER_EXT = CLASS_CONFIG['other']['ext']

    other_name = CLASS_CONFIG['other']['name']

    # Verifica che le directory esistano
    if not os.path.exists(ENC_DIR):
        print(f"\nERRORE: Directory '{ENC_DIR}' non trovata!")
        print("Crea la cartella e inserisci i file .enc")
        return
    if not os.path.exists(OTHER_DIR):
        print(f"\nERRORE: Directory '{OTHER_DIR}' non trovata!")
        print("Crea la cartella e inserisci i file .other")
        return
    
    # 1. Carica file paths
    print("\n1. Caricamento file paths...")
    enc_files = [os.path.join(ENC_DIR, f) for f in os.listdir(ENC_DIR) if f.endswith(ENC_EXT)]
    other_files = [os.path.join(OTHER_DIR, f) for f in os.listdir(OTHER_DIR) if f.endswith(OTHER_EXT)]
    
    if len(enc_files) == 0 or len(other_files) == 0:
        print("\nERRORE: Nessun file trovato nelle directory!")
        return
    
    print(f"   Trovati {len(enc_files)} file .enc")
    print(f"   Trovati {len(other_files)} file {other_name} ({OTHER_EXT})")
    
    file_paths = enc_files + other_files
    labels = [0] * len(enc_files) + [1] * len(other_files)
    
    # 2. Estrai chunk
    print(f"\n2. Estrazione {CONFIG['total_chunks']} chunk da {len(file_paths)} file...")
    chunks, chunk_labels = extract_chunks_from_files(
        file_paths,
        labels,
        CONFIG['total_chunks'],
        CONFIG['chunk_size']
    )

    # 3. Salva dataset
    print("\n3. Salvataggio dataset...")
    save_dataset(
        chunks,
        chunk_labels,
        CONFIG['output_dir'],
        CONFIG['test_size'],
        CONFIG['seed']
    )
    
    print("\nPREPARAZIONE COMPLETATA!")

    
if __name__ == "__main__":
    main()