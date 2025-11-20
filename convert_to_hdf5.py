import h5py
import pickle
import numpy as np
import os

def convert_pickle_to_hdf5(pickle_path, hdf5_path):
    # Converte dataset pickle in hdf5 per lazy loading

    print(f" Caricamento {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    labels = data['labels']

    print(f" Conversione {len(chunks)} chunk in HDF5...")

    with h5py.File(hdf5_path, 'w') as f:
        # Convertiamo il chunks in array numpy di byte
        max_len = max(len(c) for c in chunks)
        chunks_array = np.zeros((len(chunks), max_len), dtype=np.uint8)

        for i, chunk in enumerate(chunks):
            chunks_array[i, :len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
        
        # Salva in hdf5 
        f.create_dataset('chunks', data=chunks_array, 
                         compression='gzip', compression_opts=1)
        f.create_dataset('labels', data=np.array(labels, dtype=np.int32))
        f.attrs['max_chunk_size'] = max_len
    
    print(f" Salvato in {hdf5_path}\n")

if __name__ == "__main__":
    dataset_dir = './dataset'
    # Converti train e test
    convert_pickle_to_hdf5(
        os.path.join(dataset_dir, 'train_data.pkl'),
        os.path.join(dataset_dir, 'train_data.h5')
    )
    
    convert_pickle_to_hdf5(
        os.path.join(dataset_dir, 'test_data.pkl'),
        os.path.join(dataset_dir, 'test_data.h5')
    )

    print("-- compressione completata --\n")