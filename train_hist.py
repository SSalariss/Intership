import torch
from torch.utils.data import Dataset, DataLoader
import os
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
    'dataset_dir': './hist_dataset/pdf', # DA CAMBIARE A SECONDA DEL TEST CHE SI SVOLGE !!
    'batch_size': 64,
    'input_size': 256,
    'learning_rate': 1e-3,
    'num_epochs': 30,
    'device': DEVICE,
    'seed': 42,
    'save_dir': './hist_models/pdf',     # DA CAMBIARE A SECONDA DEL TEST CHE SI SVOLGE !!
}

torch.manual_seed(CONFIG['seed'])

print(f"Device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponibile: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# Dataset
class TensorDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features    # tensor [N, 256]
        self.labels = labels        # tensor [N]

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = self.features[idx]              # shape [256]
        y = self.labels[idx].long()         # label 0/1
        return x, y
    
def load_dataset(dataset_dir):
    # carica del dataset 

    train_x = torch.load(os.path.join(dataset_dir, "train_features.pt"))  # [N_train, 256]
    train_y = torch.load(os.path.join(dataset_dir, "train_labels.pt"))    # [N_train]

    test_x  = torch.load(os.path.join(dataset_dir, "test_features.pt"))   # [N_test, 256]
    test_y  = torch.load(os.path.join(dataset_dir, "test_labels.pt"))     # [N_test]

    train_ds = TensorDataset(train_x, train_y)
    test_ds  = TensorDataset(test_x, test_y)

    return train_ds, test_ds

# Modello per la classificazione con istogramma
class NnClassifier(torch.nn.Module):

    def __init__(self, input_size=256, num_labels=2):
        super().__init__()
        print(f"\nCaricamento modello MLP... \nInput: {input_size}")

    
        # Classification head: layer da eseguire
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_labels)
        )

    def forward(self,x):
        # Classifica
        logits = self.classifier(x)
        return logits # ritorniamo il logit da passare

# 1 step training
def train_epoch(model, dataloader, optimizer, criterion, config):
    # un epoch
    model.train()       # modalità training
    
    total_loss = 0      # accumula il loss
    correct = 0         # conta il numero di predizioni corrette
    total = 0           # numero di campioni per l'accuracy

    
    pbar = tqdm(dataloader, desc="Training")
    for x, y in pbar:
        x = x.to(config['device'])
        y = y.to(config['device'])

        optimizer.zero_grad()                           # azzera i gradienti (pytorch li accumula di default)
        
        # Forward pass
        logits = model(x)   
        loss = criterion(logits, y)                     # confronta i logits con le etichette vere (errore)

        # Backward (accumula i gradienti)
        loss.backward()                                 # calcola i gradienti della loss

        optimizer.step()                                # Aggiorna i pesi
        
        total_loss += loss.item()                       # accumula la loss del batch corrente

        predictions = torch.argmax(logits, dim=1)       # ottiene la classe predetta per ogni campione del batch
        correct += (predictions == y).sum().item()      # Conta quanti campioni sono stati predetti correttamente: prodotti vs etichette
        total += y.size(0)                              # Conta il numero totale di campioni
                
        # progress bar
        current_acc = correct / total                   # Accuratezza parziale sui batch fino a questo punto
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',               # loss dell'ultimo batch
            'acc': f'{current_acc:.4f}'                 # accuracy cumulativa
        })
        
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
        for x, y in tqdm(dataloader, desc="Evaulating"):
            x = x.to(config['device'])
            y = y.to(config['device'])
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    #calcolo delle metriche finali
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, config):
    # loop completo
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
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
                'test_acc': test_acc,
                'test_loss': test_loss,
                'config': config
            }, model_path)
            print(f" Salvato il miglior modello (test_acc: {test_acc:.4f}({test_acc*100:.2f}%)")

    return best_test_acc

def main():
    print(f"Input dir: {CONFIG['dataset_dir']}\nOutput dir: {CONFIG['save_dir']}")

    # Carica i tuoi tensori
    train_ds, test_ds = load_dataset(CONFIG['dataset_dir'])
    train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, CONFIG['batch_size'], shuffle=False)

    print(f"\nBatch size: {CONFIG['batch_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print(f"\n Learning Rate: {CONFIG['learning_rate']}")

    # Inizializzazione del modello
    model = NnClassifier(CONFIG['input_size'], num_labels=2).to(CONFIG['device'])

    # Fase di Training 
    train_model(model, train_loader, test_loader, CONFIG)

     # Carica best model
    checkpoint = torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(" TRAINING COMPLETATO")

if __name__ == "__main__":
    main()
