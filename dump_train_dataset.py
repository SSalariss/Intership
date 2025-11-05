import os, json, argparse
import numpy as np
import torch
from transformers import AutoTokenizer
from train_byt5 import ByteChunksDataset  # riusa la classe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./artifacts")
    ap.add_argument("--model_dir", type=str, default="google/byt5-small")
    ap.add_argument("--out_path", type=str, default="./artifacts/train_dump.jsonl")
    ap.add_argument("--max_items", type=int, default=100)   # limita la dimensione del dump
    ap.add_argument("--max_length", type=int, default=2048)
    args = ap.parse_args()

    # Carica artifacts train
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy")).astype(np.uint8, copy=False)
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy")).astype(np.int64, copy=False)

    # Tokenizer ByT5
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)

    # Dataset
    ds = ByteChunksDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), tok, max_length=args.max_length)

    # Scrivi un JSONL con alcuni esempi
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for i in range(min(args.max_items, len(ds))):
            ex = ds[i]

            # Campi base già utili per debug
            record = {
                "idx": i,
                "label": int(ex["labels"].item()),
                "mask_sum": int(ex["attention_mask"].sum().item()),
                "input_ids_head": ex["input_ids"][:32].tolist(),
                "input_ids_tail": ex["input_ids"][-32:].tolist(),
            }

            # Aggiunta: anteprima dei byte grezzi e una vista in latin-1
            bytes_i = ds.x[i].tolist()  # uint8 lungo 2048
            text_preview = bytes(bytes_i[:64]).decode("latin-1", errors="ignore")
            record["bytes_head"] = bytes_i[:32]          # primi 32 byte
            record["text_preview"] = text_preview        # anteprima decodificata (1:1 via latin-1)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote dump to {args.out_path}")

if __name__ == "__main__":
    main()
