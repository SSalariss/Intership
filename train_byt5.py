# train_byt5.py
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, get_linear_schedule_with_warmup
from tqdm import tqdm

#Dataset su byte grezzi per ByT5
class ByteChunksDataset(Dataset):
    """
    Dataset che lavora direttamente su byte (uint8).
    Ogni esempio viene shiftato di +3 per evitare collisioni
    con i token riservati 0, 1, 2 (pad, eos, unk).
    """
    def __init__(self, x_bytes: torch.Tensor, y: torch.Tensor, chunk_size: int = 2048) -> None:
        assert len(x_bytes) == len(y)
        self.x_bytes = x_bytes
        self.y = y
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x_bytes[idx]
        if len(x) > self.chunk_size:
            x = x[:self.chunk_size]

        # Shift di +3 per ByT5 (token 0,1,2 riservati)
        input_ids = torch.tensor(x, dtype=torch.long) + 3
        attention_mask = torch.ones_like(input_ids)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }
    
# Caricamento del data Loader
def make_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 0
):
    """
    Carica i file .npy e restituisce DataLoader train/test
    """
    X_train = torch.from_numpy(np.load(os.path.join(data_dir, "X_train.npy")).astype(np.uint8))[:2000]
    y_train = torch.from_numpy(np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64))[:2000]
    X_test  = torch.from_numpy(np.load(os.path.join(data_dir, "X_test.npy")).astype(np.uint8))[:1000]
    y_test  = torch.from_numpy(np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64))[:1000]

    print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"[INFO] Label dist train: {y_train.sum().item()} / {len(y_train)} positivi")

    train_ds = ByteChunksDataset(X_train, y_train)
    test_ds  = ByteChunksDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# Encoder ByT5 + pooling + testa lineare
class ByT5EncoderForClassification(nn.Module):
    def __init__(self, encoder, hidden_size, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, L, H]

        # Pooling semplice (mean o max)
        if self.pooling == "mean":
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        elif self.pooling == "max":
            pooled = hidden_states.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9).max(1).values
        else:
            raise ValueError("Pooling non valido: usa 'mean' o 'max'")

        logits = self.classifier(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return logits, loss

# Costruzione del modello
def build_byt5_classifier(model_dir="google/byt5-small", freeze_encoder=False, pooling="mean"):
    print(f"[INFO] Carico modello da {model_dir} ...")
    encoder = T5EncoderModel.from_pretrained(model_dir)
    model = ByT5EncoderForClassification(encoder, encoder.config.d_model, pooling)

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("[INFO] Encoder congelato")
    else:
        print("[INFO] Encoder allenabile")

    return model

# Training e validazione
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Uso dispositivo: {device}")

    train_loader, test_loader = make_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_byt5_classifier(args.model_dir, args.freeze_encoder, args.pooling).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))  # evita 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            #with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and not args.no_amp)):
            logits, loss = model(input_ids, attn_mask, labels)

            loss.backward()
            scaler.scale(loss).backward()
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = total_correct / total
        print(f"Epoch {epoch}: Train loss {train_loss:.4f}, acc {train_acc:.3f}")

       
        # VALIDAZIONE
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, loss = model(input_ids, attn_mask, labels)
                val_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: Val loss {val_loss:.4f}, acc {val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print("Miglior modello salvato.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping attivato a epoch {epoch}")
                break

# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="google/byt5-small")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./runs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(args)