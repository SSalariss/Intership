import torch
from transformers import AutoTokenizer
import numpy as np
import pickle
import os

# CONFIGURAZIONE

CONFIG = {
    'model_name': 'google/byt5-small',
    'max_length': 3072,
    'dataset_dir': './dataset',
    'num_samples': 10  # Numero di chunk da analizzare
}

# FUNZIONI 

def analyze_tokenization(chunk, tokenizer, max_length):
    """Analizza come il tokenizer processa un chunk"""
    
    # Converti chunk in testo, usa LATIN-1 e tenta di preservare tutti i byte!
    text = chunk.decode('latin-1')  
    
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
    
    # Conta token reali (non padding)
    num_real_tokens = attention_mask.sum().item()
    num_padding = (attention_mask == 0).sum().item()
    
    # Decodifica per vedere cosa "vede" il modello
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    # Analisi byte
    original_bytes = len(chunk)
    decoded_bytes = len(decoded_text.encode('latin-1'))  
    
    return {
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
    """Confronta diverse max_length per vedere l'impatto"""
    
    results = {}
    max_lengths = [512, 1024, 1536, 2048, 3072]
    
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


def visual_comparison(chunk, tokenizer, max_length):
    """Visualizza chunk originale vs quello che vede il modello"""
    
    result = analyze_tokenization(chunk, tokenizer, max_length)
    
    print("\nChunk Originale vs Processato")
    
    orig = result['original_text'][:2048]
    decoded = result['decoded_text'][:2048]
    
    print("\nCHUNK ORIGINALE: ")
    print(repr(orig))
    
    print("\nCHUNK VISTO DAL MODELLO: ")
    print(repr(decoded))
    
    print("\n STATISTICHE:")
    print(f"Byte originali:           {result['original_chunk_bytes']:,}")
    print(f"Byte dopo decode:         {result['decoded_text_bytes']:,}")
    print(f"Token reali processati:   {result['num_real_tokens']:,}")
    print(f"Token di padding:         {result['num_padding_tokens']:,}")
    print(f"Coverage ratio:           {result['coverage_ratio']:.2%}")
    
    if result['coverage_ratio'] < 0.95:
        print(f"\n  ATTENZIONE: Il modello vede solo {result['coverage_ratio']:.1%} dei byte originali!")
    elif result['coverage_ratio'] >= 0.99:
        print(f"\n OK: Il modello vede praticamente tutto il chunk ({result['coverage_ratio']:.1%})")
    else:
        print(f"\n Il modello vede {result['coverage_ratio']:.1%} dei byte originali")
    
    return result


# ==================== MAIN ====================

def main():
    print("\nANALIZZATORE CHUNK ByT5")
    
    # 1. Carica tokenizer
    print(f"\n1. Caricamento tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    print(f"   Tokenizer max_length configurato: {tokenizer.model_max_length:,}")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")
    
    # 2. Carica alcuni chunk dal dataset
    print(f"\n2. Caricamento chunk dal dataset...")
    
    train_path = os.path.join(CONFIG['dataset_dir'], 'train_data.pkl')
    if not os.path.exists(train_path):
        print(f" Dataset non trovato in {CONFIG['dataset_dir']}")
        return
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    chunks = train_data['chunks'][:CONFIG['num_samples']]
    labels = train_data['labels'][:CONFIG['num_samples']]
    
    print(f"   Caricati {len(chunks)} chunk per l'analisi")
    
    # 3. Analisi dettagliata di alcuni chunk
    print(f"\n3. ANALISI DETTAGLIATA")
    
    all_results = []
    
    for i, (chunk, label) in enumerate(zip(chunks, labels)):
        class_name = "PDF" if label == 1 else "ENC"
        print(f"CHUNK #{i+1} - Classe: {class_name}")
        
        result = visual_comparison(chunk, tokenizer, CONFIG['max_length'])
        all_results.append(result)
    
    # 4. Statistiche aggregate
    print("\n\nSTATISTICHE AGGREGATE")
    
    avg_coverage = np.mean([r['coverage_ratio'] for r in all_results])
    avg_real_tokens = np.mean([r['num_real_tokens'] for r in all_results])
    avg_decoded_bytes = np.mean([r['decoded_text_bytes'] for r in all_results])
    
    print(f"\nMedia su {len(all_results)} chunk:")
    print(f"  Coverage medio:        {avg_coverage:.2%}")
    print(f"  Token reali medi:      {avg_real_tokens:.0f}")
    print(f"  Byte processati medi:  {avg_decoded_bytes:.0f}")
    
    # 5. Test con diverse max_length
    print("\n\nCONFRONTO TRA DIVERSE MAX_LENGTH")
    
    sample_chunk = chunks[0]
    comparison = compare_max_lengths(sample_chunk, tokenizer)
    
    print(f"\nChunk di test: {len(sample_chunk)} byte\n")
    print(f"{'Max Length':<12} {'Token Reali':<15} {'Byte Processati':<18} {'Coverage':<10}")
    print("-" * 80)
    
    for max_len, data in sorted(comparison.items()):
        if 'error' not in data:
            print(f"{max_len:<12} {data['real_tokens']:<15} {data['decoded_bytes']:<18} {data['coverage_ratio']:.1%}")
        else:
            print(f"{max_len:<12} ERROR: {data['error']}")
    
    # 6. Raccomandazioni
    
    if avg_coverage < 0.80:
        print("\n   Il modello vede meno dell'80%\\dei byte!")
        print(f"    Coverage attuale: {avg_coverage:.1%}")
        print(f"    Max length attuale: {CONFIG['max_length']}")
    
    
    elif avg_coverage < 0.95:
        print("\n   Il modello vede circa il 90%\\dei byte")
        print(f"    Coverage attuale: {avg_coverage:.1%}")
        print(f"    Max length attuale: {CONFIG['max_length']}")
    
    else:
        print("\n   Il modello vede praticamente tutti i byte!")
        print(f"    Coverage attuale: {avg_coverage:.1%}")
        print(f"    Max length attuale: {CONFIG['max_length']}")
        print("\n   Il modello ha accesso a quasi tutto il contesto del chunk.")
    
    # 7. Info su token speciali
    print("\n\nINFO TOKEN SPECIALI")
    
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