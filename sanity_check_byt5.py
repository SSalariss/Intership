import torch, numpy as np, os
from transformers import T5EncoderModel, AutoTokenizer
from train_byt5 import ByteChunksDataset, ByT5EncoderForClassification, compute_metrics
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

# === CONFIGURAZIONE DI BASE ===
DATA_DIR = "./artifacts"
MODEL_DIR = "google/byt5-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512      
BATCH_SIZE = 32
EPOCHS = 10  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using {DEVICE}")

# === CARICA I DATI ===
X_train = torch.from_numpy(np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.uint8))
y_train = torch.from_numpy(np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int64))

# Seleziona solo 100 campioni
N = min(100, len(X_train))
X_train = X_train[:N]
y_train = y_train[:N]

print(f"Sanity check su {N} esempi. Distribuzione label:", torch.bincount(y_train))

# === COSTRUISCI MODELLO E TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
encoder = T5EncoderModel.from_pretrained(MODEL_DIR)
hidden_size = encoder.config.d_model
model = ByT5EncoderForClassification(encoder, hidden_size, num_labels=1, pooling="mean")
print("Encoder grad status:", next(model.encoder.parameters()).requires_grad)
print("Classifier grad status:", next(model.classifier.parameters()).requires_grad)

model.to(DEVICE)

# === DATASET + DATALOADER ===
ds = ByteChunksDataset(X_train, y_train, tokenizer, max_length=MAX_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
sample = ds[0]
print("Esempio tokenizzato:", sample["input_ids"][:20])
print("Label:", sample["labels"])

# === OPTIMIZER ===
optimizer = AdamW(model.parameters(), lr=5e-4)  # più alto
loss_fn = nn.BCEWithLogitsLoss()

# === TRAIN LOOP ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in dl:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out["logits"]
        loss = loss_fn(logits, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dl)

    # === Valutazione su train stesso (overfitting) ===
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attn)["logits"]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(logits, labels)

    print(f"Epoch {epoch} | loss {avg_loss:.4f} | acc {metrics['accuracy']:.3f} | f1 {metrics['f1']:.3f}")

print("Sanity check completato ")
