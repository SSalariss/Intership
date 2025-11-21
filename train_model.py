
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm

# Importa gpu_selector
try:
    from gpu_selector import get_device
    DEVICE = get_device()
    print(f"GPU assegnata: {DEVICE}")
except ImportError:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"gpu_selector non trovato, uso: {DEVICE}")


CONFIG = {
    'dataset_dir': './dataset',
    'model_name': 'google/byt5-small',
    'max_length': 3072,
    'batch_size': 2,
    'accum_steps': 8,
    'learning_rate': 5e-5,
    'num_epochs': 15,
    'device': DEVICE,
    'seed': 42,
    'save_dir': './models',
    'debug_mode': False,      # True = usa subset, False = dataset completo
    'debug_train_size': 12000,
    'debug_test_size': 2400
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

print(f"Device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponibile: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# Dataset

class ChunkDataset(Dataset):

    def __init__(self, chunks, labels):
        self.chunks = chunks
        self.labels = labels

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx], self.labels[idx]
    
def load_dataset(dataset_dir):
    # carica del dataset 

    train_path = os.path.join(dataset_dir, 'train_data.pkl')
    test_path = os.path.join(dataset_dir, 'test_data.pkl')
    info_path = os.path.join(dataset_dir, 'dataset_info.pkl')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset non trovato in {dataset_dir}.")
    
    print("\nCaricamento dataset...")

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    with open(info_path, 'rb') as f:
        info = pickle.load(f)

    print(f" Train: {len(train_data['chunks'])} chunk")
    print(f" Test: {len(test_data['chunks'])} chunk")
    print(f" Chunk size: {info['chunk_size']} byte")
    print(f" Classi: {info['class_names']}")

    train_dataset = ChunkDataset(train_data['chunks'], train_data['labels'])
    test_dataset = ChunkDataset(test_data['chunks'], test_data['labels'])

    return train_dataset, test_dataset, info

#Modello byt5 per la classificazione
class ByT5Classifier(torch.nn.Module):

    def __init__(self, model_name, num_labels=2):
        super().__init__()
        print(f"\nCaricamento modello {model_name}...")

        # carica tokenizer e encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        #d_model:  dimensione encoder output nel nostro caso small 1472
        hidden_size = self.encoder.config.d_model # dimensione del vettore di rappresentazione

        # Classification head: layer da eseguire
        '''
        tre layer permettono al modello di imparare rappresentazioni intermedie che 
        trasformano lo spazio originale di 1472 dimensioni in uno spazio di decisione per le classificazioni.

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 512),   # piu neuroni
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),          # comprime le feature in trasformazione lineare
            torch.nn.ReLU(),                    # introduce la non-linearità ReLU(x) = max(0,x)
            torch.nn.Dropout(0.2),              # disattiva dei neuroni per prevenire overfitting
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),              # 20% dei neuroni disattivati
            torch.nn.Linear(128, num_labels)    # riduce progressivamente la dimensionalità fino a ottenere le previsioni finali
        )
        '''

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.LayerNorm(256),  # Stabilizzatore
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels)
        )

    def forward(self,input_ids,attention_mask):
        #L'encoder di ByT5 processa i 2048 byte attraverso 12 layer di transformer
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # convertire una sequenza variabile di token in una singola rappresentazione fissa pronta per la classificazione.
        # mean pooling
        hidden_states = outputs.last_hidden_state                                           # last_hidden_state è un tensore di dimensione [batch_size, sequence_length, hidden_size].
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()   # espande l'attention mask per far corrispondere la forma degli hidden states, permettendo di applicare il mask a ogni dimensione
        sum_hidden = torch.sum(hidden_states * mask_expanded, 1)                            # somma i contributi di tutti i token reali (escludendo il padding) per ogni hidden state [1 è la dimensione]
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)                              # somma il mask e applica un limite minimo per evitare divisioni per zero
        pooled = sum_hidden / sum_mask                                                      # Questa riga calcola la media degli hidden states, dividendo la somma per il numero di token reali

        # Classifica
        logits = self.classifier(pooled)
        return logits # ritorniamo il logit da passare
    

def prepare_batch(chunks, labels, tokenizer, max_length, device):
    # Prepariamo il batch pcon il tokenizer byt5

    batch_labels = []
    texts = []

    for chunk, label in zip(chunks, labels):
        # il tokenizer gestisce i byte
        # dobbiamo converirli in stringhe utf8
        # text = chunk.decode('utf-8', errors='ignore')

        text = chunk.decode('latin-1')
        texts.append(text)
        batch_labels.append(label)

    #Tokenizza
    encoded = tokenizer(
        texts,                  # stringhe da tokenizzare
        padding='max_length',   # aggiungiamo padding
        truncation=True,        # tronchiamo le sequenze maggiori di 1024 byte
        max_length=max_length,
        return_tensors='pt'      # ritorna un tensore
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    labels_tensor = torch.tensor(batch_labels,dtype=torch.long).to(device)

    return input_ids, attention_mask, labels_tensor

# 1 step training
def train_epoch(model, dataloader, optimizer, criterion, config):
    # un epoch
    model.train()       # modalità training
    
    total_loss = 0      # accumula il loss
    correct = 0         # conta il numero di predizioni corrette
    total = 0           # numero di campioni per l'accuracy

    # Accumulation gradient
    accumulation_steps = config.get('accump_steps', 8)
    optimizer.zero_grad()  # IMPORTANTE: Azzerare PRIMA del loop

    pbar = tqdm(dataloader, desc="Training")
    #for chunks, labels in pbar:
    for i, (chunks, labels) in enumerate(pbar):
        '''
        La barra mostra:
            - Percentuale completata
            - Numero di batch processati
            - ETA (tempo stimato)
            - Velocità (batch/sec)
        '''
        input_ids, attention_mask, labels = prepare_batch(
            chunks, labels, model.tokenizer, config['max_length'], config['device']
        ) # ritorna input_ids, attention_mask e labels

        # rimettere qua l'opitmizer se togli accumulation
        #optimizer.zero_grad()                       # azzera i gradienti (pytorch li accumula di default)
        
        # Forward pass
        logits = model(input_ids, attention_mask)   # byte -> hidden states, mean pooling, classification head
        loss = criterion(logits, labels)            # confronta i logits con le etichette vere (errore)
        
        # Scale loss
        loss = loss / accumulation_steps
        
        # Backward (accumula i gradienti)
        loss.backward()                             # calcola i gradienti della loss

        # Step (Solo ogni N batch)
        if (i + 1) % accumulation_steps == 0:
            # Gradient Clipping (opzionale ma raccomandato)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()      # Aggiorna i pesi
            optimizer.zero_grad() # Reset dei gradienti


        '''
        da rimettere se togli accumulation, e togliere l'if sopra

        I gradienti possono diventare troppo grandi, causando aggiornamenti instabili e oscillazioni della loss
        Il gradient clipping scala verso il basso tutti i gradienti se la loro norma complessiva supera la soglia
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()                                # Aggiorna i pesi
        
        total_loss += loss.item()                       # accumula la loss del batch corrente

        '''
        # --- Metriche per logging ---
        # Moltiplichiamo di nuovo per mostrare la loss "vera" del singolo batch
        current_loss = loss.item() * accumulation_steps 
        total_loss += current_loss

        predictions = torch.argmax(logits, dim=1)       # ottiene la classe predetta per ogni campione del batch
        correct += (predictions == labels).sum().item() # Conta quanti campioni sono stati predetti correttamente: prodotti vs etichette
        total += labels.size(0)                         # Conta il numero totale di campioni
        
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{correct/total:.4f}'})

        '''
        # progress bar
        current_acc = correct / total                   # Accuratezza parziale sui batch fino a questo punto
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',               # loss dell'ultimo batch
            'acc': f'{current_acc:.4f}'                 # accuracy cumulativa
        })
        '''
    # Gestione dell'ultimo batch se il dataset non è perfettamente divisibile
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy
    

def evaluate(model, dataloader, criterion, config):
    # Valutazione del modello

    model.eval()

    total_loss=0
    correct = 0
    total = 0

    # Disabilitiamo il calcolo dei gradienti per risparmiare memoria e velocizzare l'esecuzione
    with torch.no_grad():
        for chunks, labels in tqdm(dataloader, desc="Evaulating"):
            input_ids, attention_mask, labels = prepare_batch(
                chunks, labels, model.tokenizer, config['max_length'], config['device']
            ) # tensori pronti per il modello

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    #calcolo delle metriche finali
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, config):
    # loop completo

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.05
    )

    criterion = torch.nn.CrossEntropyLoss() # Crea la funzione di loss

    
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    ) 
    

    best_test_acc = 0

    print("\nINIZIO TRAINING\n")

    for epoch in range(config['num_epochs']):
        print(f"\n Epoch {epoch+1}/{config['num_epochs']}")

        # Training
        train_loss,train_acc = train_epoch(model, train_loader, optimizer, criterion, config)

        # Evaluation
        test_loss, test_acc = evaluate(model, test_loader, criterion, config)
        
        # Update scheduler
        scheduler.step(test_loss)

        # Print risultati
        print(f"\n{'─'*60}")
        print(f"Risultati Epoch {epoch + 1}:")
        print(f"  Train → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Test  → Loss: {test_loss:.4f}  | Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"{'─'*60}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(config['save_dir'], exist_ok=True)
            model_path = os.path.join(config['save_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc:': test_acc,
                'test_loss': test_loss,
                'config': config
            }, model_path)
            print(f" Salvato il miglior modello (test_acc: {test_acc:.4f}({test_acc*100:.2f}%)")

    return best_test_acc

def main():

    print(" With latin-1 & accumulation Gradient")
    # carichiamo il dataset
    try:
        train_dataset, test_dataset, info = load_dataset(CONFIG['dataset_dir'])
    except FileNotFoundError as e:
        print(f"\n Errore nel caricamento del dataset: {e}")
        return
    
    if CONFIG['debug_mode']:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(CONFIG['debug_train_size']))
        test_dataset = Subset(test_dataset, range(CONFIG['debug_test_size']))
        print(f"\n DEBUG MODE: {CONFIG['debug_train_size']} train, {CONFIG['debug_test_size']} test")
    
    # Crea DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True, 
        num_workers = 0 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f"\nBatch size: {CONFIG['batch_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print(f"\n Learning Rate: {CONFIG['learning_rate']}")

    # Inizializzazione del modello
    model = ByT5Classifier(CONFIG['model_name'], num_labels=2).to(CONFIG['device'])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    #print(f"\nParametri totali: {total_params:,}")
    #print(f"Parametri trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Fase di Training 
    train_model(model, train_loader, test_loader, CONFIG)

     # Carica best model
    print("\nCaricamento del miglior modello per valutazione finale...")
    checkpoint = torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(" TRAINING COMPLETATO")

if __name__ == "__main__":
    main()
