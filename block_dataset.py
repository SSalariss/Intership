import pickle
import os
import torch
from torch.utils.data import Dataset

def chunk_to_histogram(block):
    # block: i bytes da 0-255 del chunk
    x = {i: 0 for i in range(256)} # i contatori inizializzati a 0

    for val in block:
        x[val] += 1 

    # Normalizzazione
    total = len(block)
    hist = [x[i] / total for i in range(256)]

    return hist # list di 256 float (somma = 1.0)

def preprocess_all_chunks(chunks, labels):
    # Convertire i chunk in istogrammi
    features = []
    new_labels = []

    for i, chunk in enumerate(chunks):
        hist = chunk_to_histogram(chunk)
        features.append(hist)
        new_labels.append(labels[i])
    
    return features, new_labels 

class HistogramDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features    # list di [256 floats]
        self.labels = labels        # list di 0/1

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def load_histogram_dataset(dataset_dir):
    train_path = os.path.join(dataset_dir, "train_data.pkl")
    test_path = os.path.join(dataset_dir, "test_data.pkl")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    # calcolo delle feature su tutti i chunk
    all_features_train, all_labels_train = preprocess_all_chunks(
        train_data['chunks'],
        train_data['labels']
    )

    all_features_test, all_labels_test = preprocess_all_chunks(
        test_data['chunks'],
        test_data['labels']
    )
    
    # PyTorch dataset
    hist_trainset = HistogramDataset(all_features_train, all_labels_train)
    hist_testset = HistogramDataset(all_features_test, all_labels_test)


    return hist_trainset, hist_testset

def main():
    # La cartella da cui prendere i pickle
    dataset_dir = "./dataset/pdf"
    print(f"input dataset: {dataset_dir}")
    # La cartella su cui salveremo i tensori
    save_dir = "./hist_dataset/pdf"
    os.makedirs(save_dir, exist_ok=True)
    print(f"output dataset: {save_dir}")

    hist_trainset, hist_testset = load_histogram_dataset(dataset_dir)

    # Estrazione dei tensori
    all_train_x = torch.stack([x for x, y in hist_trainset])
    all_train_y = torch.tensor([y for x, y in hist_trainset])

    all_test_x = torch.stack([x for x, y in hist_testset])
    all_test_y = torch.tensor([y for x, y in hist_testset])

    # Salva in file
    torch.save(all_train_x, os.path.join(save_dir, "train_features.pt"))
    torch.save(all_train_y, os.path.join(save_dir, "train_labels.pt"))

    torch.save(all_test_x, os.path.join(save_dir, "test_features.pt"))
    torch.save(all_test_y, os.path.join(save_dir, "test_labels.pt"))
    print("Dati salvati: ", save_dir)


if __name__ == "__main__":
    main()

    

    