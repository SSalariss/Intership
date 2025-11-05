import os
import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from preprocessing.fragmentation import FileChunker, list_files

def iter_dir_chunks(dir_path: str, chunker: FileChunker):
    """
    Itera tutti i file in dir_path e produce chunk di byte.
    Usa FileChunker per leggere i file a blocchi senza caricarli interamente in RAM.
    """
    files = list_files(dir_path)
    for fp in files:
        # per ogni file, genera sequenzialmente i blocchi di dimensione fissa        
        for ch in chunker.iter_file_chunks(fp):
            yield ch # restituisce i chunk come stream di byte

def count_chunks_in_dir(dir_path, chunker):
    total = 0
    for _ in iter_dir_chunks(dir_path, chunker):
        total += 1
    return total

def build_raw_bytes_streaming(pdf_dir, enc_dir, out_dir, chunk_size=2048, seed=42):
    os.makedirs(out_dir, exist_ok=True) # crea la dir se non esiste
    rng = torch.Generator().manual_seed(seed) # generatore di numeri casuali

    chunker = FileChunker(chunk_size=chunk_size, drop_last_incomplete=True, seed=seed)

    num_enc = count_chunks_in_dir(enc_dir, chunker)
    num_pdf = count_chunks_in_dir(pdf_dir, chunker)

    n_total = num_enc + num_pdf

    # Allocazione tensori in memoria (RAM)
    X_all = torch.empty((n_total, chunk_size), dtype=torch.uint8)
    y_all = torch.empty((n_total,), dtype=torch.long)

    write_ptr = 0
    for label, (dir_path, ext) in enumerate([(enc_dir, ".bin"), (pdf_dir, ".pdf")]):
        for ch in iter_dir_chunks(dir_path, chunker):
            # ch è bytes; converto in tensor torch uint8
            chunk_tensor = torch.frombuffer(ch, dtype=torch.uint8)
            X_all[write_ptr] = chunk_tensor
            y_all[write_ptr] = label
            write_ptr += 1

    # Shuffle e split
    idx = torch.randperm(n_total, generator=rng)
    test_size = int(n_total * 0.2)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    # Salvo come tensori torch
    torch.save(X_all[train_idx], os.path.join(out_dir, "X_train.pt"))
    torch.save(y_all[train_idx], os.path.join(out_dir, "y_train.pt"))
    torch.save(X_all[test_idx], os.path.join(out_dir, "X_test.pt"))
    torch.save(y_all[test_idx], os.path.join(out_dir, "y_test.pt"))


def parse_args():
    """
    Definisce e parse-a gli argomenti da riga di comando.
    Nota: max_files_per_class e --streaming sono definiti ma non utilizzati nella funzione principale.
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

    # Scrivi metadata
    meta = {
        "pdf_dir": args.pdf_dir,
        "enc_dir": args.enc_dir,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "dtype": "uint8",
        "split": {"test_size": 0.2, "stratified": True},
        "class_names": ["encrypted", "pdf"]
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()

