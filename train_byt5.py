# train_byt5.py
from typing import Tuple, Dict, Any
import os, json, time, argparse
import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from gpu_selector import get_device

class ByteChunksDataset(Dataset):
    """
    Dataset su byte grezzi per ByT5
    """
    def __init__(self, x_bytes: torch.Tensor, y: torch.Tensor, chunk_size: int = 2048) -> None:
        assert x_bytes.dtype == torch.uint8 and x_bytes.ndim == 2 and x_bytes.shape[1] == chunk_size
        assert y.dtype == torch.int64 and y.ndim == 1 and y.shape[0] == x_bytes.shape[0]
        self.x: torch.Tensor = x_bytes
        self.y: torch.Tensor = y
        self.L: int = chunk_size

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ids: 0..255 -> 3..258 per non usare 0/1/2 riservati a pad/eos/unk
        ids = self.x[idx].to(torch.long) + 3                  
        attn = torch.ones(self.L, dtype=torch.long)           
        label = torch.as_tensor(self.y[idx], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": label}

def make_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Carica X/y train e test come uint8/int64, crea Dataset byte-level e DataLoader.  
    """
    X_train = torch.from_numpy(np.load(os.path.join(data_dir, "X_train.npy")).astype(np.uint8, copy=False))
    y_train = torch.from_numpy(np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64, copy=False))
    X_test  = torch.from_numpy(np.load(os.path.join(data_dir, "X_test.npy" )).astype(np.uint8, copy=False))
    y_test  = torch.from_numpy(np.load(os.path.join(data_dir, "y_test.npy" )).astype(np.int64, copy=False))

    train_ds = ByteChunksDataset(X_train, y_train, chunk_size=X_train.shape[1])
    val_ds   = ByteChunksDataset(X_test,  y_test,  chunk_size=X_test.shape[1])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    b = next(iter(train_loader))
    print("ids range:", b["input_ids"].min().item(), b["input_ids"].max().item())
    print("mask sum (first 8):", b["attention_mask"].sum(dim=1)[:8])
    print("labels dist:", torch.bincount(b["labels"]).tolist())
    del b
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

class ByT5EncoderForClassification(nn.Module):
    """
    Encoder ByT5 + pooling + head lineare; mean pooling con mask e dropout per stabilità.  
    """
    def __init__(self, encoder_model: T5EncoderModel, hidden_size: int, num_labels: int = 1, pooling: str = "mean", dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder_model
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, L, H]
        if self.pooling == "cls":
            pooled = last_hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        return {"logits": logits}

def build_byt5_classifier(model_dir: str, pooling: str = "mean") -> ByT5EncoderForClassification:
    """
    Carica T5EncoderModel (ByT5) e costruisce la testa di classificazione. 
    """
    encoder = T5EncoderModel.from_pretrained("google/byt5-small")
    hidden_size = encoder.config.d_model
    model = ByT5EncoderForClassification(encoder, hidden_size, num_labels=1, pooling=pooling)
    return model

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calcola accuracy/precision/recall/F1 da logit e soglia.  
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    labels = labels.long()
    correct = (preds == labels).sum().item()
    acc = correct / max(1, labels.numel())
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def pick_best_threshold(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """
    Sweep su soglie in [0.01,0.99] per massimizzare F1 su validation. 
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy()
    ths = np.linspace(0.01, 0.99, 200)
    best_f1, best_th = 0.0, 0.5
    for t in ths:
        preds = (probs >= t).astype(np.int32)
        tp = ((preds==1)&(y==1)).sum()
        fp = ((preds==1)&(y==0)).sum()
        fn = ((preds==0)&(y==1)).sum()
        precision = tp / max(1, tp+fp)
        recall    = tp / max(1, tp+fn)
        f1 = 2*precision*recall / max(1e-9, precision+recall)
        if f1 > best_f1:
            best_f1, best_th = f1, t
    return float(best_th), float(best_f1)

def train(args: argparse.Namespace) -> None:
    # device
    try:
        device_str = get_device()
    except SystemExit:
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    # modello
    model = build_byt5_classifier(args.model_dir, pooling=args.pooling)
    model.to(device)

    # (opzionale) congelamento iniziale
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # dataloader
    train_loader, val_loader = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # optimizer & scheduler
    if args.freeze_encoder:
        optimizer = AdamW([{"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.0}])
    else:
        head_params = list(model.classifier.parameters())
        base_params = [p for n,p in model.named_parameters() if p.requires_grad and "classifier" not in n]
        optimizer = AdamW([
            {"params": base_params, "lr": args.lr,   "weight_decay": 0.01},
            {"params": head_params, "lr": 5e-4,      "weight_decay": 0.0},
        ])

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # BCE con pos_weight
    y_train_all = train_loader.dataset.y
    N_pos = int((y_train_all == 1).sum().item())
    N_neg = int((y_train_all == 0).sum().item())
    pos_w = float(N_neg / max(1, N_pos)) if N_pos > 0 else 1.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    print(f"pos_weight={pos_w:.3f}")

    best_val = float("inf")
    patience = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        start_t = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"].clamp(-20, 20)
                loss = bce(logits, labels.float())

            # guardrail numerico
            if not torch.isfinite(loss):
                print("Non-finite loss:", loss.item(), "logits:", logits.min().item(), logits.max().item())
                raise RuntimeError("Loss is NaN/Inf")

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        dur = time.time() - start_t

        # validazione
        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out["logits"].clamp(-20, 20)
                    loss = bce(logits, labels.float())
                if torch.isfinite(loss):
                    val_loss += loss.item()
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if len(val_loader) > 0:
            val_loss /= max(1, len(val_loader))
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        best_th, _ = pick_best_threshold(logits, labels)
        metrics = compute_metrics(logits, labels, threshold=best_th)

        print(f"Epoch {epoch:02d} | time {dur:.1f}s | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"th {best_th:.3f} | acc {metrics['accuracy']:.4f} f1 {metrics['f1']:.4f} "
              f"P {metrics['precision']:.4f} R {metrics['recall']:.4f}")

        if torch.isfinite(torch.tensor(val_loss)) and (val_loss < best_val - 1e-6):
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "byt5_cls_best.pt"))
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping triggered.")
                break

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({"best_val_loss": best_val, "metrics": metrics, "config": vars(args)}, f, indent=2)
