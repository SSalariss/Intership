import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import random


CONFIG = {
    'chunk_size': 2048,                         # Dimensione in byte di ogni chunk
    'model_name': 'google/byt5-small',          # Modello ByT5 da usare
    'max_length': 1024,                         # Lunghezza della sequenza (token)
    'batch_size': 16,                           # Dimensione del batch
    'learning_rate': 5e-5,                      # Learning rate per AdamW
    'num_epochs': 5,                            # Numero massimo di epoche
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Usa GPU se disponibile
    'seed': 42,                                 # Seed per la riproducibilità
    'patience': 3                               # Patience per early stopping
}


# Imposta i seed per rendere i risultati riproducibili
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

print(f"Device: {CONFIG['device']}")


class FileChunkDataset(Dataset):
    """Dataset per chunk di file (ENC vs PDF)."""
    
    def __init__(self, file_paths, labels, chunk_size=2048, chunks_per_file=10):
        # Liste in cui salveremo i chunk e le relative etichette
        self.chunks = []
        self.labels = []
        
        print("Caricamento ed estrazione chunk...")
        # Itera su tutti i file e le loro etichette
        for file_path, label in tqdm(zip(file_paths, labels), total=len(file_paths)):
            try:
                # Legge l'intero contenuto del file in modalità binaria
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                file_size = len(file_data)  # Dimensione del file in byte
                
                # Scarta file troppo piccoli (poco informativi)
                if file_size < 512:
                    print(f"Skipping {file_path} (too small: {file_size} bytes)")
                    continue
                
                if file_size < chunk_size:
                    # Se il file è più piccolo del chunk, fa padding con byte 0
                    chunk = file_data + b'\x00' * (chunk_size - file_size)
                    self.chunks.append(chunk)
                    self.labels.append(label)
                else:
                    # Estrae più chunk per file in modo deterministico (no random)
                    max_start = file_size - chunk_size
                    for i in range(chunks_per_file):
                        # Calcola uno start deterministico in base all'indice i
                        start = (max_start * i) // chunks_per_file
                        # Estrae il chunk di dimensione fissa
                        chunk = file_data[start:start + chunk_size]
                        self.chunks.append(chunk)
                        self.labels.append(label)
            except Exception as e:
                # In caso di errore nella lettura del file, lo segnala ma continua
                print(f"Errore nel caricare {file_path}: {e}")
        
        # Stampa numero totale di chunk e distribuzione delle etichette (ENC vs PDF)
        print(f"Totale chunk estratti: {len(self.chunks)}")
        # Calcola quanti PDF (label 1) e quanti ENC (label 0) ci sono
        n_pdf = sum(self.labels)
        n_enc = len(self.labels) - n_pdf
        print(f"Distribuzione classi: PDF={n_pdf}, ENC={n_enc}")
    
    def __len__(self):
        # Numero totale di esempi nel dataset
        return len(self.chunks)
    
    def __getitem__(self, idx):
        # Restituisce il chunk e la label corrispondente
        chunk = self.chunks[idx]
        label = self.labels[idx]
        return chunk, label


def load_file_paths(enc_dir, pdf_dir):
    """Carica i path dei file ENC e PDF e crea le etichette."""
    # Cerca file che finiscono con .bin
    enc_files = [os.path.join(enc_dir, f) for f in os.listdir(enc_dir) if f.endswith('.bin')]
    # Prende tutti i file .pdf nella cartella pdf_dir
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    # Concatenazione dei path
    file_paths = enc_files + pdf_files
    # Etichette: 0 per ENC, 1 per PDF
    labels = [0] * len(enc_files) + [1] * len(pdf_files)
    
    print(f"File ENC: {len(enc_files)}")
    print(f"File PDF: {len(pdf_files)}")
    
    return file_paths, labels


class ByT5Classifier(torch.nn.Module):
    """ByT5 adattato per classificazione binaria."""
    
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        # Carica il modello T5 per generazione condizionale (include encoder+decoder)
        self.byt5 = T5ForConditionalGeneration.from_pretrained(model_name)
        # Usa solo l'encoder per la classificazione
        self.encoder = self.byt5.encoder
        
        # Congela i parametri del decoder (non servono per classificazione)
        for param in self.byt5.decoder.parameters():
            param.requires_grad = False
        
        # Testa di classificazione: LayerNorm + MLP
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.encoder.config.d_model),   # Normalizzazione per stabilizzare il training
            torch.nn.Linear(self.encoder.config.d_model, 256), # Proiezione nello spazio intermedio
            torch.nn.ReLU(),                                   # Attivazione non lineare
            torch.nn.Dropout(0.3),                             # Dropout per regolarizzazione
            torch.nn.Linear(256, num_labels)                   # Layer finale per 2 classi
        )
    
    def forward(self, input_ids, attention_mask):
        # Passa gli input all'encoder di ByT5
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Prende gli hidden states (shape: batch_size x seq_len x d_model)
        hidden_states = encoder_outputs.last_hidden_state
        # Mean pooling pesato dall'attenzione per ottenere un vettore per sequenza
        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        
        # Applica la testa di classificazione al vettore pooled
        logits = self.classifier(pooled)
        return logits


def prepare_batch(chunks, labels, tokenizer, max_length, device):
    """
    Prepara un batch per il training:
    - converte i bytes in stringhe UTF-8
    - tokenizza con il tokenizer di ByT5
    - restituisce tensori su device
    """
    
    texts = []  # Lista di stringhe da dare al tokenizer
    for chunk in chunks:
        try:
            # Decodifica i bytes in stringa UTF-8
            # errors="ignore" scarta i byte non validi per UTF-8 invece di lanciare eccezioni
            text = chunk.decode('utf-8', errors='ignore')
            texts.append(text)
        except Exception as e:
            # In caso di errore inatteso, usa stringa vuota come fallback
            print(f"Errore decodifica: {e}")
            texts.append('')
    
    # Usa il tokenizer ufficiale di ByT5 (lavora su UTF-8 raw bytes internamente)
    encoded = tokenizer(
        texts,
        padding='max_length',   # Pad fino a max_length
        max_length=max_length,  # Trunca se più lungo di max_length
        truncation=True,
        return_tensors='pt'     # Restituisce tensori PyTorch
    )
    
    # Sposta input_ids e attention_mask sul device scelto (CPU/GPU)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Converte le label in tensore Long su device
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    return input_ids, attention_mask, labels_tensor


def train_model(model, train_loader, val_loader, config, tokenizer):
    """Esegue il training del modello con validazione e early stopping."""
    
    # Ottimizzatore AdamW per modelli Transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    # Loss per classificazione multi-classe (qui 2 classi)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0.0        # Migliore accuratezza di validazione vista finora
    patience_counter = 0      # Contatore per l'early stopping
    
    for epoch in range(config['num_epochs']):
        # Modalità training (abilita dropout, ecc.)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Barra di progresso per il training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for chunks, labels in pbar:
            # Prepara il batch (tokenizzazione + spostamento su device)
            input_ids, attention_mask, labels = prepare_batch(
                chunks, labels, tokenizer, config['max_length'], config['device']
            )
            
            # Azzera i gradienti accumulati
            optimizer.zero_grad()
            # Forward pass: ottiene i logits dal modello
            logits = model(input_ids, attention_mask)
            # Calcola la loss di cross-entropy
            loss = criterion(logits, labels)
            # Backpropagation: calcolo dei gradienti
            loss.backward()
            # Aggiorna i pesi
            optimizer.step()
            
            # Accumula loss e accuratezza
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Aggiorna la barra di progresso con la loss corrente
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Loss media e accuratezza sul training set
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Modalità evaluation (disabilita dropout, ecc.)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Nessun gradiente durante la validazione
        with torch.no_grad():
            for chunks, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]"):
                # Prepara il batch di validazione
                input_ids, attention_mask, labels = prepare_batch(
                    chunks, labels, tokenizer, config['max_length'], config['device']
                )
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                # Loss di validazione
                loss = criterion(logits, labels)
                
                # Accumula loss e accuratezza
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Loss media e accuratezza sul validation set
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Log dei risultati per l'epoca corrente
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        
        # Salva il modello se migliora l'accuratezza di validazione
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # Reset del contatore di patience
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ Salvato miglior modello (val_acc: {val_acc:.4f})")
        else:
            # Nessun miglioramento: aumenta il contatore di patience
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['patience']}")
            
            # Se non migliora per 'patience' epoche consecutive, ferma il training
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping attivato dopo {epoch+1} epoche")
                break
    
    return best_val_acc


def evaluate_model(model, test_loader, config, tokenizer):
    """Valutazione finale del modello sul test set (accuracy, report, confusion matrix)."""
    
    # Modalità evaluation
    model.eval()
    
    all_predictions = []  # Lista di tutte le predizioni
    all_labels = []       # Lista di tutte le etichette reali
    all_probs = []        # Probabilità per la classe positiva (PDF)
    
    # Nessun gradiente durante la valutazione
    with torch.no_grad():
        for chunks, labels in tqdm(test_loader, desc="Evaluating"):
            # Prepara il batch di test
            input_ids, attention_mask, labels = prepare_batch(
                chunks, labels, tokenizer, config['max_length'], config['device']
            )
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            # Probabilità softmax sulle 2 classi
            probs = torch.softmax(logits, dim=1)
            # Predizione come argmax
            predictions = torch.argmax(logits, dim=1)
            
            # Salva predizioni, etichette e probabilità
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilità della classe 1 (PDF)
    
    # Stampa metriche di classificazione dettagliate
    print("\n" + "="*50)
    print("RISULTATI TEST SET")
    print("="*50)
  
    # Calcola e stampa l'accuracy globale
    test_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    return test_acc


def main():
    # Directory aggiornata per file ENC
    ENC_DIR = './data/enc'
    PDF_DIR = './data/pdf'
    
    # Carica i path dei file e le etichette
    print("Caricamento file paths...")
    # Chiamata alla funzione aggiornata
    file_paths, labels = load_file_paths(ENC_DIR, PDF_DIR)
    
    # Split train/val/test a livello di FILE per evitare data leakage
    print("\nSplit train/val/test...")
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=CONFIG['seed']
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.125, stratify=train_labels, random_state=CONFIG['seed']
    )
    
    print(f"  Train: {len(train_files)} file")
    print(f"  Val:   {len(val_files)} file")
    print(f"  Test:  {len(test_files)} file")
    
    # Crea i dataset di chunk a partire dai file
    print("\nCreazione datasets...")
    train_dataset = FileChunkDataset(train_files, train_labels, CONFIG['chunk_size'])
    val_dataset = FileChunkDataset(val_files, val_labels, CONFIG['chunk_size'])
    test_dataset = FileChunkDataset(test_files, test_labels, CONFIG['chunk_size'])
    
    # Crea i DataLoader per batching ed iterazione
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Carica il tokenizer ByT5 (lavora su UTF-8 raw bytes)
    print("\nCaricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Carica il modello ByT5 adattato alla classificazione
    print("\nCaricamento ByT5-small...")
    model = ByT5Classifier(CONFIG['model_name']).to(CONFIG['device'])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri trainable: {trainable_params:,}/{total_params:,}")
    
    # Avvia il training
    print("\n" + "="*50)
    print("INIZIO TRAINING")
    print("="*50 + "\n")
    train_model(model, train_loader, val_loader, CONFIG, tokenizer)
    
    # Carica il miglior modello salvato e valuta sul test set
    print("\nCaricamento miglior modello per test finale...")
    model.load_state_dict(torch.load('best_model.pt'))
    evaluate_model(model, test_loader, CONFIG, tokenizer)
    
    print("\nTraining completato!")


if __name__ == "__main__":
    main()
