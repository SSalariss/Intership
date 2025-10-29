import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.fragmentation import FileChunker, list_files

def iter_dir_chunks(dir_path: str, chunker: FileChunker, exts):
    files = list_files(dir_path, exts=exts)
    for fp in files:
        for ch in chunker.iter_file_chunks(fp):
            yield ch

def build_raw_bytes_streaming(pdf_dir, enc_dir, per_class_samples, out_dir,
                              chunk_size=2048, seed=42, pdf_ext=".pdf", enc_ext=".bin"):
    import numpy as np, os
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    chunker = FileChunker(chunk_size=chunk_size, drop_last_incomplete=True, seed=seed)

    n_per_cls = per_class_samples
    n_total = n_per_cls * 2

    X_mm_path = os.path.join(out_dir, "X_all_mm.dat")
    y_mm_path = os.path.join(out_dir, "y_all_mm.dat")
    X_mm = np.memmap(X_mm_path, dtype=np.uint8, mode="w+", shape=(n_total, chunk_size))
    y_mm = np.memmap(y_mm_path, dtype=np.int64, mode="w+", shape=(n_total,))

    write_ptr = 0
    for label, (dir_path, ext) in enumerate([(enc_dir, enc_ext), (pdf_dir, pdf_ext)]):
        taken = 0
        for ch in iter_dir_chunks(dir_path, chunker, exts=[ext] if ext else None):
            X_mm[write_ptr, :] = np.frombuffer(ch, dtype=np.uint8, count=chunk_size)
            y_mm[write_ptr] = label
            write_ptr += 1
            taken += 1
            if taken >= n_per_cls:
                break
        if taken < n_per_cls:
            raise ValueError(f"Classe {label}: trovati solo {taken} chunk, servono {n_per_cls}")

    idx = np.arange(n_total)
    rng.shuffle(idx)

    test_size = int(n_total * 0.2)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    def save_split(indices, x_path, y_path, block=8192):
        X_out = np.empty((len(indices), chunk_size), dtype=np.uint8)
        y_out = np.empty((len(indices),), dtype=np.int64)
        start = 0
        for i in range(0, len(indices), block):
            sl = indices[i:i+block]
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

