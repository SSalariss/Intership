
#create_dataset.py
"""
Preparazione del dataset con 20 mila chunk, da 2048 byte
"""
import random
import string
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle

from script.utils.preprocessing_utils import train_test_split



# Configurazione
CONFIG = {
    'chunk_size': 2048,
    'total_chunks': 20000, # i chunk da estrarre
    'test_size': 0.2, # 20% per testing, 80% per il training
    'seed': 42,
    'output_dir': './dataset'
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

    Args:
        - Lista di path ai file
        - Lista di Label (0 = ENC, 1 = PDF)
        - num_chunks: numero totale di chunk da estrarre
        - chunks_size: dimensione del chunk in byte

    Returns: 
        - lista di chunks 
        - chunk_labels: lista di label per ogni chunks
    """

    #separa i file in classi

    bin_files = [f for f, l in zip(file_paths, labels) if l == 0] # zip è una funzione che ritorna un iteratore di tuple
    pdf_files = [f for f, l in zip(file_paths, labels) if l == 1]

    #calcola quanti chunk per classe per il bilanciamento del dataset
    chunks_per_class = num_chunks // 2

    print(f"\nEstrazione {chunks_per_class} chunk per classe...")
    print(f"File ENC disponibili: {len(bin_files)}")
    print(f"File PDF disponibili: {len(pdf_files)}")

    def extract_from_class(files, num_chunks, label_value):
        chunks = []
        labels = []
        chunks_per_file = max(1, num_chunks // len(files))

        pbar = tqdm(files, desc=f"Estrazione {'ENC'if label_value == 0 else 'PDF'}")
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
    bin_chunks, bin_labels = extract_from_class(bin_files, chunks_per_class, 0)
    pdf_chunks, pdf_labels = extract_from_class(pdf_files, chunks_per_class, 1)

    # e combiniamo tutto insieme
    all_chunks = bin_chunks + pdf_chunks
    all_labels = bin_labels + pdf_labels

    #loggin: stamoa il numero totale di chunk per ogni classe
    print(f"\nTotale chunk estratti: {len(all_chunks)}")
    print(f"  ENC: {sum(1 for l in all_labels if l == 0)}")
    print(f"  PDF: {sum(1 for l in all_labels if l == 1)}")

    return all_chunks, all_labels

# Analisi esplorativa per investigare sull'entropia di enc e pdf

def exploratory_analysis(chunks, labels, sample_size=500):
    """Analisi esplorativa dei chunk"""
    print("\n Analisi esplorativa...\n")
    
    # Campiona chunk per l'analisi
    bin_indices = [i for i, l in enumerate(labels) if l == 0]
    pdf_indices = [i for i, l in enumerate(labels) if l == 1]
    
    # Prendiamo un sottoinsieme rappresentativo tramite gli indici con dimensione dettata da sample_size
    bin_sample = random.sample(bin_indices, min(sample_size, len(bin_indices)))
    pdf_sample = random.sample(pdf_indices, min(sample_size, len(pdf_indices)))
    
    def compute_stats(chunk_indices, label_name):
        '''
        L'entropia di Shannon misura la quantità media di informazione o "disordine" nei dati. 
        Valori più alti indicano dati più casuali o più variabilità,
        mentre valori bassi indicano dati più prevedibili.
        '''
        entropies = []
        ascii_ratios = []
        byte_distributions = np.zeros(256)
        
        for idx in chunk_indices:
            chunk = chunks[idx]
            byte_array = np.frombuffer(chunk, dtype=np.uint8)
            
            # Entropia di Shannon
            hist, _ = np.histogram(byte_array, bins=256, range=(0, 256))
            hist = hist / len(byte_array) # normalizzazione di hist: la somma degli elementi diventa 1
            hist = hist[hist > 0] # eliminazione degli zeri per evitare errori nel calcolo
            entropy = -np.sum(hist * np.log2(hist)) # calcolo dell'entropia sulla distribuzione contenuta in hist
            entropies.append(entropy) # accumulatore dell'entropia
            
            # ASCII printable ratio
            ascii_count = np.sum((byte_array >= 32) & (byte_array <= 126)) # ascii stampabili
            ascii_ratios.append(ascii_count / len(byte_array)) # rapporto tra char stampabili e lunghezza totale
            
            # Distribuzione byte aggregata
            byte_distributions += hist * 256  #riporta la distribuzione in scala assoluta
        
        byte_distributions /= len(chunk_indices) # Frequenza media normalizzata
        
        # .3f: cifre decimali
        print(f"\n{label_name}:")
        print(f"  Entropia media: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
        print(f"  ASCII ratio medio: {np.mean(ascii_ratios):.3f} ± {np.std(ascii_ratios):.3f}")
        print(f"  Min entropia: {np.min(entropies):.3f}")
        print(f"  Max entropia: {np.max(entropies):.3f}")
        
        return entropies, ascii_ratios, byte_distributions
    
    bin_stats = compute_stats(bin_sample, "ENC")
    pdf_stats = compute_stats(pdf_sample, "PDF")
    
    # Visualizzazioni: 2 righe, 2 colonne
    # axes array 2D di oggetti, per ogni grafico della griglia
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Entropia
    '''
    serve a visualizzare e confrontare la distribuzione 
    dell'entropia calcolata su due insiemi di chunk, 
    evidenziando differenze o similitudini nella variabilità informativa dei dati.
    '''
    axes[0, 0].hist(bin_stats[0], bins=30, alpha=0.6, label='ENC', color='blue', edgecolor='black')
    axes[0, 0].hist(pdf_stats[0], bins=30, alpha=0.6, label='PDF', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Entropia (bit)')
    axes[0, 0].set_ylabel('Frequenza')
    axes[0, 0].set_title('Distribuzione Entropia dei Chunk')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # ASCII ratio
    '''
    Serve a visualizzare e confrontare la distribuzione delle 
    proporzioni di caratteri ASCII stampabili tra i due gruppi di chunk, 
    per comprendere meglio la composizione testuale o binaria 
    dei dati nei due insiemi
    '''
    axes[0, 1].hist(bin_stats[1], bins=30, alpha=0.6, label='ENC', color='blue', edgecolor='black')
    axes[0, 1].hist(pdf_stats[1], bins=30, alpha=0.6, label='PDF', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('ASCII Printable Ratio')
    axes[0, 1].set_ylabel('Frequenza')
    axes[0, 1].set_title('Distribuzione ASCII Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Distribuzione byte ENC
    '''
    serve a visualizzare la distribuzione media dei valori byte nei chunk 
    appartenenti alla classe ENC, fornendo un dettaglio più granulare 
    rispetto agli istogrammi precedenti.
    '''
    axes[1, 0].bar(range(256), bin_stats[2], color='blue', alpha=0.7)
    axes[1, 0].set_xlabel('Valore Byte (0-255)')
    axes[1, 0].set_ylabel('Frequenza Media')
    axes[1, 0].set_title('Distribuzione Valori Byte - ENC')
    axes[1, 0].grid(alpha=0.3)
    
    # Distribuzione byte PDF
    '''
    permette di visualizzare la distribuzione media dei valori di byte 
    nei chunk appartenenti alla classe PDF, a complemento del grafico della classe ENC, 
    facilitando il confronto tra le due distribuzioni
    '''
    axes[1, 1].bar(range(256), pdf_stats[2], color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Valore Byte (0-255)')
    axes[1, 1].set_ylabel('Frequenza Media')
    axes[1, 1].set_title('Distribuzione Valori Byte - PDF')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout() # migliorare la disposizione
    
    return fig

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
    print(f"Train: {len(train_chunks)} chunk ({sum(train_labels)} PDF, {len(train_labels)-sum(train_labels)} ENC)")
    print(f"Test: {len(test_chunks)} chunk ({sum(test_labels)} PDF, {len(test_labels)-sum(test_labels)} ENC)")

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
        'class_names': ['ENC', 'PDF']
    }
    with open(os.path.join(output_dir, 'dataset_info.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nDataset salvato in '{output_dir}/'")
    print(f"  - train_data.pkl ({len(train_chunks)} chunk)")
    print(f"  - test_data.pkl ({len(test_chunks)} chunk)")
    print(f"  - dataset_info.pkl (metadata)")

# Lancio del main
def main():
    
    BIN_DIR = './data/enc'
    PDF_DIR = '.data/pdf'

    # Verifica che le directory esistano
    if not os.path.exists(BIN_DIR):
        print(f"\nERRORE: Directory '{BIN_DIR}' non trovata!")
        print("Crea la cartella e inserisci i file .bin")
        return
    if not os.path.exists(PDF_DIR):
        print(f"\nERRORE: Directory '{PDF_DIR}' non trovata!")
        print("Crea la cartella e inserisci i file .pdf")
        return
    
    # 1. Carica file paths
    print("\n1. Caricamento file paths...")
    bin_files = [os.path.join(BIN_DIR, f) for f in os.listdir(BIN_DIR) if f.endswith('.bin')]
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    if len(bin_files) == 0 or len(pdf_files) == 0:
        print("\nERRORE: Nessun file trovato nelle directory!")
        return
    
    print(f"   Trovati {len(bin_files)} file .bin")
    print(f"   Trovati {len(pdf_files)} file .pdf")
    
    file_paths = bin_files + pdf_files
    labels = [0] * len(bin_files) + [1] * len(pdf_files)
    
    # 2. Estrai chunk
    print(f"\n2. Estrazione {CONFIG['total_chunks']} chunk da {len(file_paths)} file...")
    chunks, chunk_labels = extract_chunks_from_files(
        file_paths,
        labels,
        CONFIG['total_chunks'],
        CONFIG['chunk_size']
    )

    # 3. Analisi esplorativa
    print("\n3. Analisi esplorativa dei chunk...")
    fig = exploratory_analysis(chunks, chunk_labels)

    # Salva grafico
    output_path = os.path.join(CONFIG['output_dir'], 'eda_analysis.png')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Grafico salvato in '{output_path}'")
    plt.close()

    # 4. Salva dataset
    print("\n4. Salvataggio dataset...")
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