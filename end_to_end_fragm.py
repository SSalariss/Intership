import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.fragmentation import FileChunker, list_files

def iter_dir_chunks(dir_path: str, chunker: FileChunker, exts):
    """
    Itera tutti i file in dir_path (filtrati per estensioni) e produce chunk di byte.
    Usa FileChunker per leggere i file a blocchi senza caricarli interamente in RAM.
    """
    files = list_files(dir_path, exts=exts)
    for fp in files:
        # per ogni file, genera sequenzialmente i blocchi di dimensione fissa        
        for ch in chunker.iter_file_chunks(fp):
            yield ch # restituisce i chunk come stream di byte

def build_raw_bytes_streaming(pdf_dir, enc_dir, per_class_samples, out_dir,
                              chunk_size=2048, seed=42, pdf_ext=".pdf", enc_ext=".bin"):
    """
    Costruisce un dataset bilanciato di frammenti di byte (encrypted vs pdf) in modalità streaming.
    - Legge chunk di dimensione fissa dai file delle due directory.
    - Scrive i campioni in memmap su disco per minimizzare l'uso di RAM.
    - Esegue shuffle e split 80/20 e salva .npy finali più metadata.json.
    """
    import numpy as np, os
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Chunker che garantisce blocchi esattamente di chunk_size (droppa l’ultimo incompleto)   
    chunker = FileChunker(chunk_size=chunk_size, drop_last_incomplete=True, seed=seed)

    n_per_cls = per_class_samples   # numero di campioni per ciascuna classe
    n_total = n_per_cls * 2         # due classi: encrypted e pdf

    # File memmap temporanei per scrivere progressivamente senza caricare tutto in RAM
    X_mm_path = os.path.join(out_dir, "X_all_mm.dat")
    y_mm_path = os.path.join(out_dir, "y_all_mm.dat")
    X_mm = np.memmap(X_mm_path, dtype=np.uint8, mode="w+", shape=(n_total, chunk_size))
    y_mm = np.memmap(y_mm_path, dtype=np.int64, mode="w+", shape=(n_total,))

    # Riempie i memmap leggendo a chunk dalle due directory e assegnando le etichette
    write_ptr = 0
    for label, (dir_path, ext) in enumerate([(enc_dir, enc_ext), (pdf_dir, pdf_ext)]):
        taken = 0 # numero di chunk presi
        for ch in iter_dir_chunks(dir_path, chunker, exts=[ext] if ext else None):
            # da buffer di byte in array uint8 di lunghezza |chunk|
            X_mm[write_ptr, :] = np.frombuffer(ch, dtype=np.uint8, count=chunk_size)
            y_mm[write_ptr] = label # etichetta: 0=encrypted, 1=pdf
            write_ptr += 1
            taken += 1
            if taken >= n_per_cls:
                break
        if taken < n_per_cls:
            raise ValueError(f"Classe {label}: trovati solo {taken} chunk, servono {n_per_cls}")

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
        X_out = np.empty((len(indices), chunk_size), dtype=np.uint8)
        y_out = np.empty((len(indices),), dtype=np.int64)
        start = 0
        for i in range(0, len(indices), block):
            sl = indices[i:i+block] # sottoinsieme di indici
            X_out[start:start+len(sl)] = X_mm[sl]
            y_out[start:start+len(sl)] = y_mm[sl]
            start += len(sl)
        np.save(x_path, X_out)
        np.save(y_path, y_out)

    save_split(train_idx, os.path.join(out_dir, "X_train.npy"), os.path.join(out_dir, "y_train.npy"))
    save_split(test_idx,  os.path.join(out_dir, "X_test.npy"),  os.path.join(out_dir, "y_test.npy"))

    del X_mm, y_mm
    os.remove(X_mm_path)
    os.remove(y_mm_path)


def parse_args():
    """
    Definisce e parse-a gli argomenti da riga di comando.
    Nota: max_files_per_class e --streaming sono definiti ma non utilizzati nella funzione principale.
    """
    p = argparse.ArgumentParser(description="End-to-end (raw bytes): chunking, split, save")
    p.add_argument("--pdf_dir", required=True, type=str)
    p.add_argument("--enc_dir", required=True, type=str)
    p.add_argument("--pdf_ext", default=".pdf", type=str)
    p.add_argument("--enc_ext", default=".bin", type=str)
    p.add_argument("--samples", default=20000, type=int, help="Samples per class")
    p.add_argument("--chunk_size", default=2048, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--max_files_per_class", default=None, type=int)
    p.add_argument("--out_dir", default="./artifacts", type=str)
    p.add_argument("--streaming", action="store_true")

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
        per_class_samples=args.samples,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        seed=args.seed,
        pdf_ext=args.pdf_ext,
        enc_ext=args.enc_ext,
    )
    print(f"Saved raw-bytes dataset to {args.out_dir}")

    # Scrivi metadata
    meta = {
        "pdf_dir": args.pdf_dir,
        "enc_dir": args.enc_dir,
        "samples_per_class": args.samples,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "pdf_ext": args.pdf_ext,
        "enc_ext": args.enc_ext,
        "dtype": "uint8",
        "split": {"test_size": 0.2, "stratified": True},
        "class_names": ["encrypted", "pdf"]
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()

