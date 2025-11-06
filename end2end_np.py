import os
import json
import argparse

#import test
import numpy as np
#from sklearn.model_selection import train_test_split
from preprocessing.np_fragmentation import FileChunker, list_files
from tqdm import tqdm


def iter_dir_chunks(dir_path: str, chunker: FileChunker):
    """
    Itera tutti i file in dir_path e produce chunk di byte.
    Usa FileChunker per leggere i file a blocchi senza caricarli interamente in RAM.
    """
    files = list_files(dir_path)
    for fp in tqdm(files, desc="Processing files..."):
     # per ogni file, genera sequenzialmente i blocchi di dimensione fissa        
        for ch in chunker.iter_file_chunks(fp):
            yield ch # restituisce i chunk come stream di byte

def count_chunks(directory, chunker, ext):
    count = 0
    for _ in tqdm(iter_dir_chunks(directory, chunker),desc="counting chunks"):
        count += 1
    return count

def build_raw_bytes_streaming(pdf_dir, enc_dir, out_dir, chunk_size=2048, seed=42): #samples_per_class=20000):
    """
    Costruisce un dataset frammenti per classe (encrypted vs pdf).
    Divide 80/20 in train/test e salva i .npy risultanti.
    """
    
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)


    # Chunker che garantisce blocchi esattamente di chunk_size (droppa l’ultimo incompleto)   
    chunker = FileChunker(chunk_size=chunk_size, drop_last_incomplete=True, seed=seed)
    '''
    def collect_random_chunks(dir_path, label, max_samples):
            """Estrae casualmente fino a max_samples chunk da dir_path."""
            chunks = []
            for ch in tqdm(iter_dir_chunks(dir_path, chunker), desc=f"Reading {dir_path}"):
                chunks.append(np.frombuffer(ch, dtype=np.uint8, count=chunk_size))
                if len(chunks) >= max_samples:
                    break
            X = np.stack(chunks)
            y = np.full((len(chunks),), label, dtype=np.int64)
            return X, y

    print(f"Collecting up to {samples_per_class} chunks per class...")

    # Carica un campione bilanciato
    X_enc, y_enc = collect_random_chunks(enc_dir, label=0, max_samples=samples_per_class)
    X_pdf, y_pdf = collect_random_chunks(pdf_dir, label=1, max_samples=samples_per_class)

    # Combina le due classi
    X = np.concatenate([X_enc, X_pdf], axis=0)
    y = np.concatenate([y_enc, y_pdf], axis=0)

    # Shuffle complessivo
    idx = np.arange(len(y))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Split 80/20
    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Salva i dataset finali
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    '''
    
    enc_count = count_chunks(enc_dir, chunker, ".bin")
    pdf_count = count_chunks(pdf_dir, chunker, ".pdf")
    n_total = enc_count + pdf_count # due classi: encrypted e pdf


    # File memmap temporanei per scrivere progressivamente senza caricare tutto in RAM
    X_mm_path = os.path.join(out_dir, "X_all_mm.dat")
    y_mm_path = os.path.join(out_dir, "y_all_mm.dat")
    X_mm = np.memmap(X_mm_path, dtype=np.uint8, mode="w+", shape=(n_total, chunk_size))
    y_mm = np.memmap(y_mm_path, dtype=np.int64, mode="w+", shape=(n_total,))


    # Riempie i memmap leggendo a chunk dalle due directory e assegnando le etichette
    write_ptr = 0
    for label, (dir_path, ext) in enumerate([(enc_dir, ".bin"), (pdf_dir, ".pdf")]):
        taken = 0 # numero di chunk presi
        for ch in tqdm(iter_dir_chunks(dir_path, chunker), desc=f"Reading chunks from {dir_path}"):
            # da buffer di byte in array uint8 di lunghezza |chunk|
            X_mm[write_ptr, :] = np.frombuffer(ch, dtype=np.uint8, count=chunk_size)
            y_mm[write_ptr] = label # etichetta: 0=encrypted, 1=pdf
            write_ptr += 1
            taken += 1
        if taken == 0:
            raise ValueError(f"Classe {label}: nessun chunk trovato nei file")

    # Randomizer
    idx = np.arange(n_total)
    rng.shuffle(idx)

    test_size = int(n_total * 0.2)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    def save_split(indices, x_path, y_path, block=8192):
        """
        Copia dal memmap gli esempi selezionati dagli indici in array compatti
        e salva su disco come .npy, elaborando a blocchi per contenere la RAM.
         """
        # array NumPy non inizializzati
        X_out = np.empty((len(indices), chunk_size), dtype=np.uint8) # i dati dei chunk selezionati
        y_out = np.empty((len(indices),), dtype=np.int64) # e etichette corrispondenti ai chunk selezionati
        start = 0
        for i in tqdm(range(0, len(indices), block), desc="splitting.."):
            sl = indices[i:i+block] # sottoinsieme di indici
            X_out[start:start+len(sl)] = X_mm[sl]
            y_out[start:start+len(sl)] = y_mm[sl]
            start += len(sl)
        np.save(x_path, X_out)
        np.save(y_path, y_out)


    save_split(train_idx, os.path.join(out_dir, "X_train.npy"), os.path.join(out_dir, "y_train.npy"))
    save_split(test_idx, os.path.join(out_dir, "X_test.npy"), os.path.join(out_dir, "y_test.npy"))


    del X_mm, y_mm
    os.remove(X_mm_path)
    os.remove(y_mm_path)



def parse_args():
    """
    Definisce e parsa gli argomenti da riga di comando.
    """
    p = argparse.ArgumentParser(description="End-to-end (raw bytes): chunking, split, save")
    p.add_argument("--pdf_dir", required=True, type=str)
    p.add_argument("--enc_dir", required=True, type=str)
    p.add_argument("--chunk_size", default=2048, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--out_dir", default="./artifacts", type=str)


    return p.parse_args()


def main():
    """
    Entry point: esegue la pipeline di costruzione del dataset e salva metadati.
    """
    args = parse_args()


    print("Building raw-bytes dataset (streaming, low RAM)...")
    build_raw_bytes_streaming(
        pdf_dir=args.pdf_dir,
        enc_dir=args.enc_dir,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    print(f"Saved raw-bytes dataset to {args.out_dir}")

    # Metadata
    meta = {
        "pdf_dir": args.pdf_dir,
        "enc_dir": args.enc_dir,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "dtype": "uint8",
        "split": {"train": 0.8, "test": 0.2},
        "class_names": ["encrypted", "pdf"]
    }
    
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
