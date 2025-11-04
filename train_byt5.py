# file: train_byt5.py 

import os, json, time, argparse
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from gpu_selector import get_device

class ByteChunksDataset(Dataset):
    """
    Dataset PyTorch per frammenti di byte destinati a ByT5.

    Input atteso:
    - x_bytes: array NumPy di dtype uint8 e shape (N, 2048).
      Ogni riga rappresenta un frammento da 2048 byte preso dai file originali.
      Manteniamo i byte grezzi per essere coerenti con l'approccio token-free di ByT5. 

    - y: array NumPy di dtype int64 e shape (N,).
      Etichetta binaria per ciascun frammento: 0 = encrypted (.bin), 1 = pdf (.pdf). 

    - tokenizer: tokenizer associato al checkpoint di ByT5 (es. google/byt5-small).
      Questo tokenizer lavora a livello di byte UTF-8, senza vocabolario di subword,
      quindi mappa direttamente i caratteri/byte in input_ids compatibili con l'encoder. 

    - max_length: lunghezza massima della sequenza per ByT5.
      Usiamo 2048 per mantenere esattamente un frammento da 2KB per input.
      Se add_special_tokens=True, il tokenizer aggiunge i token speciali necessari e
      troncando/padding si garantisce una lunghezza fissa. 
    """
    def __init__(self, x_bytes: torch.Tensor, y: torch.Tensor, tokenizer:AutoTokenizer, max_length: int = 2050):
        # Controlli di integrità su tipo e dimensionalità dei dati
        assert isinstance(x_bytes,  torch.Tensor) and x_bytes.dtype == torch.uint8 and x_bytes.ndim == 2, \
            "x_bytes deve essere  torch.Tensor uint8 di shape (N, 2048)"
        assert isinstance(y,  torch.Tensor) and y.dtype == torch.int64 and y.ndim == 1, \
            "y deve essere  torch.Tensor int64 di shape (N,)"
        assert x_bytes.shape[0] == y.shape[0], \
            "X e y devono avere lo stesso numero di campioni"
        self.x:torch.Tensor = x_bytes
        self.y:torch.Tensor = y
        self.tok:AutoTokenizer = tokenizer
        self.max_length:int = max_length

    def __len__(self):
        # Numero di campioni totali
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Restituisce un singolo esempio trasformato per ByT5:
        - input_ids: tensore Long [max_length]
        - attention_mask: tensore Long [max_length]
        - labels: tensore Long scalare (0 o 1)

        Passi chiave:
        1) Recupera il frammento grezzo (2048 byte) da X.
        2) Converte i byte in una stringa "latin-1" per un mapping 1:1 byte->char,
           evitando perdita/alterazione. Usare latin-1 significa che i valori 0..255
           restano invariati come singoli caratteri Unicode, che il tokenizer ByT5
           mapperà poi a byte-level input_ids. 
        3) Tokenizza con padding/troncamento a lunghezza fissa max_length.
        4) Converte l'etichetta in tensore Long per PyTorch.
        """
        chunk = self.x[idx]  # shape (2048,), dtype uint8

        # Conversione bytes -> stringa 1:1 via latin-1.
        # Nota: bytes(chunk) creerebbe una copia; bytes(chunk.tolist()) è più esplicito.
        # errors="ignore" scarta eventuali anomalie, ma con latin-1 non dovrebbero emergere.
        text = bytes(chunk.tolist()).decode("latin-1", errors="ignore")

        # Tokenizzazione per ByT5:
        # - add_special_tokens: permette di includere eventuali token speciali richiesti dal modello.
        # - padding="max_length": garantisce batch con sequenze di lunghezza uniforme (essenziale per GPU).
        # - truncation=True: se per qualche motivo la stringa superasse max_length, viene tronca (non dovrebbe in questo caso).
        # - return_tensors="pt": ritorna tensori PyTorch pronti per il DataLoader.
        enc = self.tok(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),       # LongTensor [L]
            "attention_mask": enc["attention_mask"].squeeze(0),  # LongTensor [L]
            "labels": torch.as_tensor(self.y[idx], dtype=torch.long)  # LongTensor []
        }
        return item

# Esempio di costruzione DataLoader (da usare nel tuo train loop):
def make_dataloaders(data_dir, tokenizer, batch_size=32, max_length=2048, num_workers=2):
    """
    Carica gli artifact salvati (X_train.npy, y_train.npy, X_test.npy, y_test.npy),
    istanzia i Dataset e crea DataLoader ottimizzati per GPU.

    - pin_memory=True velocizza il trasferimento CPU->GPU.
    - shuffle=True sul train per mescolare gli esempi ad ogni epoca.
    """
    X_train:torch.Tensor = torch.from_numpy(np.load(os.path.join(data_dir, "X_train.npy")).astype(np.uint8, copy=False))
    y_train:torch.Tensor = torch.from_numpy(np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64, copy=False))
    X_test:torch.Tensor  = torch.from_numpy(np.load(os.path.join(data_dir, "X_test.npy")).astype(np.uint8, copy=False))
    y_test:torch.Tensor  = torch.from_numpy(np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64, copy=False))

    train_ds = ByteChunksDataset(X_train, y_train, tokenizer, max_length=max_length)
    val_ds   = ByteChunksDataset(X_test,  y_test,  tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    # Sanity check input
    b = next(iter(train_loader))
    print("ids range:", b["input_ids"].min().item(), b["input_ids"].max().item())
    print("mask sum (first 8):", b["attention_mask"].sum(dim=1)[:8])
    print("labels dist:", torch.bincount(b["labels"]).tolist())
    del b
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader

class ByT5EncoderForClassification(nn.Module):
    """
    Wrapper che usa l'encoder di ByT5 per classificazione binaria su sequenze di byte.

    Idea:
    - Passiamo gli input_ids e attention_mask nell'encoder ByT5 per ottenere le hidden states
      di dimensione d_model per ciascuna posizione della sequenza. 
    - Riduciamo le hidden states a una singola rappresentazione vettoriale per esempio
      tramite un'operazione di pooling (mean pooling sulle posizioni valide secondo la mask,
      oppure 'cls' usando la prima posizione). 
    - Applichiamo una testa lineare (Linear) per proiettare d_model -> 1 logit.
      Usiamo BCEWithLogitsLoss per classificazione binaria. 
    """

    def __init__(self, encoder_model, hidden_size: int, num_labels: int = 1, pooling: str = "mean"):
        """
        Parametri:
        - encoder_model: istanza di AutoModel caricata da un checkpoint ByT5 (es. google/byt5-small).
          AutoModel restituisce last_hidden_state dell'encoder (non serve il decoder per classificazione). [file:2]
        - hidden_size: dimensione d_model del ByT5 (presa da encoder.config.d_model).
        - num_labels: 1 per binario (logit singolo); se multi-classe, impostare al numero di classi e usare CrossEntropy. [file:2]
        - pooling: 'mean' (consigliato) oppure 'cls'. 
        """
        super().__init__()
        self.encoder = encoder_model # t5encoderModel
        self.pooling = pooling
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Input:
        - input_ids: LongTensor [B, L] prodotto dal tokenizer ByT5 a partire dai byte. [file:2]
        - attention_mask: LongTensor [B, L] con 1 dove il token è valido (non pad). [file:2]
        - labels: LongTensor [B] con 0/1. Se fornito, calcoliamo la loss. [file:2]

        Output:
        - dict con:
          - 'loss' (se labels presenti): BCEWithLogitsLoss su logit singolo.
          - 'logits': Tensor [B] con i logit grezzi (prima di sigmoid). [file:2]
        """
        # Esegue il forward dell'encoder ByT5 e ottiene le hidden states per token: [B, L, H]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, H]

        # Pooling: comprimiamo [L, H] -> [H] per ciascun esempio.
        if self.pooling == "cls":
            # Usa la prima posizione (convenzionalmente tipo [BOS]); semplice ma meno robusta in assenza di token dedicati.
            pooled = last_hidden[:, 0, :]  # [B, H]
        else:
            # Mean pooling pesata dalla mask per ignorare padding.
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            summed = (last_hidden * mask).sum(dim=1)     # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-6)      # [B, 1] evita divisione per zero
            pooled = summed / denom                      # [B, H]

        # Testa di classificazione: proiezione lineare verso 1 logit per esempio.
        logits = self.classifier(pooled).squeeze(-1)  # [B]

        loss = None
        if labels is not None:
            # Per binario usiamo BCEWithLogitsLoss: applica internamente la sigmoid ai logit.
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return {"logits": logits}

# Esempio di istanziazione del modello a partire da un checkpoint locale:
def build_byt5_classifier(model_dir: str, pooling: str = "mean"):
    """
    Carica AutoModel (encoder) e restituisce il wrapper con classification head.
    """
    encoder = T5EncoderModel.from_pretrained(model_dir)           # carica l'encoder ByT5
    hidden_size = encoder.config.d_model                     # dimensione d_model
    
    # Riusa la tua testa di classificazione 
    model = ByT5EncoderForClassification(encoder, hidden_size, num_labels=1, pooling=pooling)
    
    # Inizializza la testa per stabilità
    nn.init.trunc_normal_(model.classifier.weight, std=0.02)
    nn.init.zeros_(model.classifier.bias)


    # Tokenizer: per il byte-level di byt5
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)  # tokenizer byte-level
    return model, tokenizer


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    Calcola metriche standard per classificazione binaria.

    Parametri:
    - logits: Tensor [N] con i logit grezzi (uscita della testa prima della sigmoid). [file:2]
    - labels: Tensor [N] con etichette 0/1 (dtype long o int). [file:2]
    - threshold: soglia di decisione sulla probabilità della classe positiva. [file:2]

    Passi:
    1) Applica sigmoid ai logit per ottenere probabilità p = P(y=1 | x).
    2) Confronta con la soglia per produrre predizioni binarie 0/1.
    3) Calcola:
       - accuracy = fractione di predizioni corrette.
       - precision = TP / (TP + FP) (quanto sono “pulite” le positive). 
       - recall = TP / (TP + FN) (quante positive vere vengono recuperate).
       - F1 = media armonica tra precision e recall (equilibrio tra le due). [file:2]

    Note:
    - Aggiungiamo un piccolo epsilon (1e-9) al denominatore per evitare divisioni per zero
      in casi limite (es. nessuna predizione positiva). [file:2]
    """
    with torch.no_grad():
        # 1) Probabilità dalla sigmoid
        probs = torch.sigmoid(logits)              # [N]
        # 2) Predizioni binarie
        preds = (probs >= threshold).long()        # [N]
        labels = labels.long()                     # [N]

        # 3) Metriche
        correct = (preds == labels).sum().item()
        acc = correct / max(1, labels.numel())

        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        # Facoltativo: anche la specificità (TN rate)
        # tn = ((preds == 0) & (labels == 0)).sum().item()
        # specificity = tn / (tn + fp + 1e-9)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            # "specificity": specificity
        }

def train(args):
    """
    Esegue il fine-tuning di ByT5 per classificazione binaria su frammenti di 2048 byte.

    Caratteristiche:
    - Selezione automatica della GPU libera sul cluster (get_device). [file:2]
    - Mixed precision (autocast + GradScaler) per ridurre memoria VRAM e accelerare. [file:2]
    - Scheduler lineare con warmup per stabilità dell'ottimizzazione. [file:2]
    - Early stopping su validation loss, salvataggio del best checkpoint. [file:2]
    """
    
    # 1) Scelta del dispositivo (GPU del cluster o fallback CPU)
    try:
        device_str = get_device()  # e.g. 'cuda:0'; termina se non disponibile
    except SystemExit:
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    # 2) Costruzione modello e tokenizer dal checkpoint locale ByT5 (solo encoder)
    model, tokenizer = build_byt5_classifier(args.model_dir, pooling=args.pooling)
    model.to(device)

    # Facoltativo: congela l'encoder per le prime epoche per stabilizzare/risparmiare VRAM
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # 3) DataLoader per train/val
    train_loader, val_loader = make_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )

    # 4) Ottimizzatore e scheduler
    if args.freeze_encoder:
    # Solo head
        optimizer = AdamW([
            {"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.0},
        ])
    else:
        # Fine-tuning completo
        head_params = list(model.classifier.parameters())
        base_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
        optimizer = AdamW([
            {"params": base_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": head_params, "lr": 5e-4, "weight_decay": 0.0},
        ])


    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)  # più conservativo
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 5) Mixed precision (attivo solo su CUDA)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler(device="cuda", enabled=use_amp)
    # 6) Early stopping
    best_val = float("inf")
    patience = 0
    os.makedirs(args.out_dir, exist_ok=True)

    # Estrai tutte le label train dal dataset per calcolare il peso
    y_train_all = train_loader.dataset.y  # è un torch.Tensor int64
    N_pos = int((y_train_all == 1).sum().item())
    N_neg = int((y_train_all == 0).sum().item())
    pos_w = (N_neg / max(1, N_pos)) if N_pos > 0 else 1.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    print(f"pos_weight={pos_w:.3f}")


    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        start_t = time.time()

        # Loop di training per batch
        for batch in train_loader:
            # Trasferimento tensori su GPU con non_blocking per overlap CPU->GPU
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward in fp16 (autocast) se su CUDA
            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = out["logits"].clamp(-20, 20)  # stabilità numerica
                loss = bce(logits, labels.float())

            # Backprop con GradScaler per stabilità numerica in fp16
            scaler.scale(loss).backward()
            # Clipping per evitare esplosioni di gradiente
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        dur = time.time() - start_t

        # 7) Validazione
        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out["loss"].item()
                all_logits.append(out["logits"].detach().cpu())
                all_labels.append(labels.detach().cpu())

        val_loss /= max(1, len(val_loader))
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(logits, labels)

        print(f"Epoch {epoch:02d} | time {dur:.1f}s | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"acc {metrics['accuracy']:.4f} f1 {metrics['f1']:.4f} P {metrics['precision']:.4f} R {metrics['recall']:.4f}")

        # 8) Early stopping & checkpoint
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "byt5_cls_best.pt"))
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping triggered.")
                break

    # 9) Report finale
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({"best_val_loss": best_val, "metrics": metrics, "config": vars(args)}, f, indent=2)

# file: train_byt5.py (estratto: argparse + main)

import argparse
import os

def parse_args():
    """
    Parametri principali:
    - data_dir: cartella con X_train.npy, y_train.npy, X_test.npy, y_test.npy (artifact generati dallo streaming). [file:2]
    - model_dir: checkpoint locale di ByT5 (es. google/byt5-small scaricato). [file:2]
    - out_dir: cartella per checkpoint e report del training. [file:2]
    - batch_size: esempi per batch (aumenta su GPU con più VRAM, riduci se OOM). [file:2]
    - epochs: numero di epoche. [file:2]
    - lr: learning rate (5e-5 per fine-tuning encoder; se alleni solo la head, 1e-3 può essere ok). [file:2]
    - warmup_ratio: frazione dei passi totali usata per warmup del LR. [file:2]
    - weight_decay: regolarizzazione L2. [file:2]
    - max_grad_norm: clipping dei gradienti. [file:2]
    - max_length: lunghezza sequenza ByT5 (2048 per un chunk da 2KB). [file:2]
    - pooling: 'mean' o 'cls' per la riduzione [L,H] -> [H]. [file:2]
    - freeze_encoder: se presente, allena solo la testa di classificazione nella prima run. [file:2]
    - num_workers: processi di prefetch dei DataLoader; 2-4 su cluster è spesso un buon compromesso. [file:2]
    """
    ap = argparse.ArgumentParser(description="Fine-tuning ByT5 (byte-level) per classificazione binaria")
    ap.add_argument("--data_dir", type=str, default="./artifacts")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./runs/byt5_small_cls")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no_amp", action="store_true")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
