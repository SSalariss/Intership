# preprocessing/fragmentation.py

import random
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Dict

import numpy as np


# -------------------------------
# Utility
# -------------------------------

def list_files(root_dir: str, exts: Optional[Iterable[str]] = None) -> List[Path]:
    root = Path(root_dir)
    if exts:
        exts = set(e.lower() for e in exts)
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
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
        self.chunk_size = chunk_size
        self.drop_last_incomplete = drop_last_incomplete
        self.rng = random.Random(seed)

    def iter_file_chunks(self, filepath: Path) -> Iterable[bytes]:
        with open(filepath, "rb") as f:
            while True:
                buf = f.read(self.chunk_size)
                if not buf:
                    break
                if len(buf) < self.chunk_size and self.drop_last_incomplete:
                    break
                yield buf

    def collect_chunks_from_dir(
        self,
        dir_path: str,
        exts: Optional[Iterable[str]] = None,
        max_files: Optional[int] = None
    ) -> List[bytes]:
        files = list_files(dir_path, exts=exts)
        if max_files is not None:
            files = files[:max_files]
        chunks: List[bytes] = []
        for fp in files:
            for ch in self.iter_file_chunks(fp):
                chunks.append(ch)
        return chunks

    def sample_class_chunks(
        self,
        class_dirs: Dict[str, str],
        per_class_samples: int,
        exts_map: Optional[Dict[str, Iterable[str]]] = None,
        max_files_per_class: Optional[int] = None,
    ) -> Tuple[List[bytes], List[int], List[str]]:
        """
        class_dirs: mappa {class_name: directory}
        exts_map: mappa facoltativa {class_name: estensioni consentite}
        Ritorna: chunks(list di bytes), labels (0..C-1), class_names_by_label
        """
        class_names = list(class_dirs.keys())
        chunks_all: List[bytes] = []
        labels: List[int] = []

        for label, cname in enumerate(class_names):
            exts = exts_map.get(cname) if exts_map else None
            class_chunks = self.collect_chunks_from_dir(
                class_dirs[cname],
                exts=exts,
                max_files=max_files_per_class
            )
            if len(class_chunks) < per_class_samples:
                raise ValueError(f"Classe {cname}: trovati {len(class_chunks)} chunk, servono {per_class_samples}")
            self.rng.shuffle(class_chunks)
            class_sample = class_chunks[:per_class_samples]
            chunks_all.extend(class_sample)
            labels.extend([label] * len(class_sample))

        # shuffle complessivo mantenendo allineate etichette
        idx = list(range(len(chunks_all)))
        self.rng.shuffle(idx)
        chunks_all = [chunks_all[i] for i in idx]
        labels = [labels[i] for i in idx]
        return chunks_all, labels, class_names

    # ---------------------------
    # Helper per dataset raw bytes
    # ---------------------------

    def to_numpy_uint8(self, chunks: List[bytes]) -> np.ndarray:
        """
        Converte list[bytes] in un array np.uint8 shape (N, chunk_size).
        """
        if not chunks:
            return np.empty((0, self.chunk_size), dtype=np.uint8)
        arr = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        N = len(chunks)
        return arr.reshape(N, self.chunk_size)

    def build_raw_dataset(
        self,
        pdf_dir: str,
        enc_dir: str,
        per_class_samples: int = 20000,
        pdf_exts: Optional[Iterable[str]] = (".pdf",),
        enc_exts: Optional[Iterable[str]] = (".bin",),
        max_files_per_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Costruisce direttamente X (uint8, N x chunk_size) e y (int64) sui byte grezzi.
        """
        class_dirs = {"encrypted": enc_dir, "pdf": pdf_dir}
        exts_map = {"pdf": pdf_exts, "encrypted": enc_exts}
        chunks, labels, class_names = self.sample_class_chunks(
            class_dirs=class_dirs,
            per_class_samples=per_class_samples,
            exts_map=exts_map,
            max_files_per_class=max_files_per_class,
        )
        X = self.to_numpy_uint8(chunks)  # shape (N, chunk_size), dtype uint8
        y = np.array(labels, dtype=np.int64)
        return X, y, class_names
