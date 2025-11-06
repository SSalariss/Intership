# preprocessing/fragmentation.py

import random
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Dict

import numpy as np


# -------------------------------
# Utility
# -------------------------------

def list_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    return [p for p in root.rglob("*") if p.is_file()]


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


# -------------------------------
# FileChunker: frammentazione e sampling
# -------------------------------

class FileChunker:
    """
    Legge file binari e li suddivide in chunk di dimensione fissa (default 2048 B).
    Può raccogliere chunk da una cartella per classe e poi campionarne un numero target per classe.
    Produce chunk come bytes (o come np.ndarray uint8) per uso diretto con modelli byte-level.
    """

    def __init__(self, chunk_size: int = 2048, drop_last_incomplete: bool = True, seed: int = 42):
        # default a 2048
        self.chunk_size = chunk_size
        # flag booleano che indica se scartare o meno
        self.drop_last_incomplete = drop_last_incomplete
        # operazioni randomiche sui chunk 
        self.rng = random.Random(seed)

    # serve a leggere un file binario a pezzi (chunk) di dimensione fissa
    def iter_file_chunks(self, filepath: Path) -> Iterable[bytes]:
        with open(filepath, "rb") as f: # rb: modalià binaria
            while True:
                buf = f.read(self.chunk_size)
                if not buf: # se non ci sono byte da leggere esci
                    break 
                if len(buf) < self.chunk_size and self.drop_last_incomplete: # se è piu piccolo del chunk size scartalo
                    break
                yield buf # restituisce un valore ma permette alla funzione di riprendere l'esecuzione successivamente.

    def collect_chunks_from_dir(
        self,
        dir_path: str, # il path della directory da passare 
    ) -> List[bytes]:
        
        files = list_files(dir_path)
        chunks: List[bytes] = []
        # per ogni file va a prendersi tutti i chunk possibili
        for fp in files:
            for ch in self.iter_file_chunks(fp):
                chunks.append(ch)
        return chunks

    # campionamento di un numero specifico di chunk di byte da directory associate a classi
    # restituisce i chunk, le etichette numeriche e i nomi delle classi.
    def sample_class_chunks(
        self,
        class_dirs: Dict[str, str],
ù    ) -> Tuple[List[bytes], List[int], List[str]]:
        """
        class_dirs: mappa {class_name: directory}
        Ritorna: chunks(list di bytes), labels (0..C-1), class_names_by_label
        """
        class_names = list(class_dirs.keys()) # serve a iterare sulle classi con un indice numerico e un nome.
        chunks_all: List[bytes] = []          # raccoglitore dei chunk di tutte le classi.
        labels: List[int] = []                # memorizzare le etichette numeriche corrispondenti ai chunk

        # iteriamo le classi per ottenere l'indice numerico e il nome
        for label, cname in enumerate(class_names):
            
            class_chunks = self.collect_chunks_from_dir(class_dirs[cname]) # Raccoglie tutti i chunk dalla directory
            # self.rng.shuffle(class_chunks) # mescoliamo per migliorare la varietà nei batch 
            chunks_all.extend(class_chunks) # byte raccolti per la classe corrente
            labels.extend([label] * len(class_chunks)) # label corrispondenti per ogni chunk aggiunto

        # shuffle complessivo mantenendo allineate etichette
        indice = list(range(len(chunks_all)))
        self.rng.shuffle(indice)
        chunks_all = [chunks_all[i] for i in indice]
        labels = [labels[i] for i in indice]

        return chunks_all, labels, class_names

    # ---------------------------
    # Helper per dataset raw bytes
    # ---------------------------

    def to_numpy_uint8(self, chunks: List[bytes]) -> np.ndarray:
        """
        Converte list[bytes] in un array np.uint8 shape (N, chunk_size).
        """
        if not chunks:
            return np.empty((0, self.chunk_size), dtype=np.uint8) #fallback per chunk vuoti
        
        arr = np.frombuffer(b"".join(chunks), dtype=np.uint8) # crea un unico buffer in un vettore di uint8
        N = len(chunks) # numero di chunk

        return arr.reshape(N, self.chunk_size) # matrice N righe, chunk size colonne
    
    

    def build_raw_dataset(
        self,
        pdf_dir: str, # pdf directory
        enc_dir: str, # bin directory
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Costruisce direttamente X (uint8, N x chunk_size) e y (int64) sui byte grezzi.
        """
        class_dirs = {"encrypted": enc_dir, "pdf": pdf_dir} # dizionario nome classe : nome directory
        '''
        Leggere i chunk di byte grezzi da ciascuna directory di classe
        - Ottenere la lista dei chunk (chunks)
        - Ottenere le etichette numeriche corrispondenti (labels)
        - Ottenere i nomi delle classi (class_names)
        '''
        chunks, labels, class_names = self.sample_class_chunks(class_dirs=class_dirs)
        
        X = self.to_numpy_uint8(chunks) # Converte la lista di chunk in un array NumPy uint8
        y = np.array(labels, dtype=np.int64) 
        # classname serve per interpretare le etichette numeriche
        return X, y, class_names
