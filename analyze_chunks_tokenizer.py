import torch
from transformers import AutoTokenizer
import numpy as np
import pickle
import os

CONFIG = {
    'model_name': 'google/byt5-small',
    'max_length': 1024,
    'dataset_dir': './dataset',
    'num_samples': 10
}

def analyze_tokenization(chunk, tokenizer, max_length):
    # Analizziamo come il tokenizer va a processare un chunk

    # Convertiamo il chunk
    text = chunk.decode('utf-8', errors='ignore')

    # Tokenizza
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'][0]
    attention_mask = encoded['attention_mask'][0]

    # Conta token reali senza padding
    num_real_tokens = attention_mask.sum().item()
    num_padding = (attention_mask == 0).sum().item()

    # Decodifica per vedere cosa vede il modello
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # Analisi dei byte
    original_bytes = len(chunk)
    decoded_bytes = len(decoded_text.encode('utf-8'))

    return{
        'original_chunk_bytes': original_bytes,
        'decoded_text_bytes': decoded_bytes,
        'num_real_tokens': num_real_tokens,
        'num_padding_tokens': num_padding,
        'total_tokens': len(input_ids),
        'text_length_chars': len(text),
        'decoded_length_chars': len(decoded_text),
        'coverage_ratio': decoded_bytes / original_bytes if original_bytes > 0 else 0,
        'original_text': text,
        'decoded_text': decoded_text
    }

def compare_max_lengths(chunk, tokenizer):
    # Confronta diverse max_length per vedere l'impatto

    results = {}
    max_lengths = [512,1024,1536,2048,3072]

    for max_len in max_lengths:
        try:
            result = analyze_tokenization(chunk, tokenizer, max_len)
            results[max_len] = {
                'real_tokens': result['num_real_tokens'],
                'coverage_ratio': result['coverage_ratio'],
                'decoded_bytes': result['decoded_text_bytes']
            }
        except Exception as e:
            results[max_len] = {'error': str(e)}

    return results

def visual_comparison(chunk,tokenizer, max_length):
    
    result = analyze_tokenization(chunk, tokenizer, max_length)

    print("\nConfronto tra il chunk Originale e quello Processato: \n")

    orig = result['original_text'][:2048]
    decoded = result['decoded_text'][:2048]

    print("\n i char dell'originale:\n", repr(orig))

    print("\n i char del modello:\n", repr(decoded))

    print("\n statistiche: ")
    print(f"Byte originali:           {result['original_chunk_bytes']:,}")
    print(f"Byte dopo decode:         {result['decoded_text_bytes']:,}")
    print(f"Token reali processati:   {result['num_real_tokens']:,}")
    print(f"Token di padding:         {result['num_padding_tokens']:,}")
    print(f"Coverage ratio:           {result['coverage_ratio']:.2%}")

    if result['coverage_ratio'] < 0.95:
        print(f"\n Il modello vede solo {result['coverage_ratio']:.1%} dei byte originali!")
    elif result['coverage_ratio'] >= 0.99:
        print(f"\n Il modello vede praticamente tutto il chunk ({result['coverage_ratio']:.1%})")
    else:
        print(f"\n Il modello vede {result['coverage_ratio']:.1%} dei byte originali")
    
    return result

def main():
    
    print("Analizzatore chunk byt5")

    # Carica tokenizer
    print(f"\n Caricamento del tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    print(f"   Tokenizer max_length configurato: {tokenizer.model_max_length:,}")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")

     # Carica alcuni chunk dal dataset
    print(f"\n Caricamento chunk dal dataset...")
    
    train_path = os.path.join(CONFIG['dataset_dir'], 'train_data.pkl')
    if not os.path.exists(train_path):
        print(f" Dataset non trovato in {CONFIG['dataset_dir']}")
        return
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    chunks = train_data['chunks'][:CONFIG['num_samples']]
    labels = train_data['labels'][:CONFIG['num_samples']]
    
    print(f"   Caricati {len(chunks)} chunk per l'analisi")
    
    # Analisi dettagliata di alcuni chunk
    print(f"\n ANALISI DEI CHUNK")
    
    all_results = []
    
    for i, (chunk, label) in enumerate(zip(chunks, labels)):
        class_name = "PDF" if label == 1 else "ENC"
        print(f"CHUNK #{i+1} - Classe: {class_name}")
        
        result = visual_comparison(chunk, tokenizer, CONFIG['max_length'])
        all_results.append(result)
    
    # Statistiche aggregate
    print("\n\nSTATISTICHE AGGREGATE")
    
    avg_coverage = np.mean([r['coverage_ratio'] for r in all_results])
    avg_real_tokens = np.mean([r['num_real_tokens'] for r in all_results])
    avg_decoded_bytes = np.mean([r['decoded_text_bytes'] for r in all_results])
    
    print(f"\nMedia su {len(all_results)} chunk:")
    print(f"  Coverage medio:        {avg_coverage:.2%}")
    print(f"  Token reali medi:      {avg_real_tokens:.0f}")
    print(f"  Byte processati medi:  {avg_decoded_bytes:.0f}")
    
    # Test con diverse max_length
    print("\nCONFRONTO TRA DIVERSE MAX_LENGTH")
    
    sample_chunk = chunks[0]
    comparison = compare_max_lengths(sample_chunk, tokenizer)
    
    print(f"\nChunk di test: {len(sample_chunk)} byte\n")
    print(f"{'Max Length':<12} {'Token Reali':<15} {'Byte Processati':<18} {'Coverage':<10}")
    
    for max_len, data in sorted(comparison.items()):
        if 'error' not in data:
            print(f"{max_len:<12} {data['real_tokens']:<15} {data['decoded_bytes']:<18} {data['coverage_ratio']:.1%}")
        else:
            print(f"{max_len:<12} ERROR: {data['error']}")
    
    # Raccomandazioni
    
    if avg_coverage < 0.80:
        print("\n  Il modello vede meno dell'80%\\dei byte!")
        print(f"   Coverage attuale: {avg_coverage:.1%}")
        print(f"   Max length attuale: {CONFIG['max_length']}")
    
    elif avg_coverage < 0.95:
        print("\n  Il modello vede circa il 90%\\dei byte")
        print(f"   Coverage attuale: {avg_coverage:.1%}")
        print(f"   Max length attuale: {CONFIG['max_length']}")
    
    else:
        print("\n  Il modello vede praticamente tutti i byte!")
        print(f"   Coverage attuale: {avg_coverage:.1%}")
        print(f"   Max length attuale: {CONFIG['max_length']}")
        print("\n  Il modello ha accesso a quasi tutto il contesto del chunk.")
    
    # Info su token speciali
    print("INFO TOKEN SPECIALI")
    
    print(f"\nToken speciali di ByT5:")
    print(f"  PAD token:  {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token:  {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  UNK token:  {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

     # Test token speciali
    sample_text = chunks[0].decode('utf-8', errors='ignore')[:100]
    encoded = tokenizer(sample_text, return_tensors='pt')
    
    num_pad = (encoded['input_ids'] == tokenizer.pad_token_id).sum().item()
    num_eos = (encoded['input_ids'] == tokenizer.eos_token_id).sum().item()
    num_unk = (encoded['input_ids'] == tokenizer.unk_token_id).sum().item()
    
    print(f"\nIn un sample encoding:")
    print(f"  PAD tokens: {num_pad}")
    print(f"  EOS tokens: {num_eos}")
    print(f"  UNK tokens: {num_unk}")


if __name__ == "__main__":
    main()