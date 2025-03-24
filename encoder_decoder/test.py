import torch
from dataset import load_multi30k_dataset, load_tokenizers, tokenize_en, tokenize_de
from encoder_decoder import Transformer, create_mask
import os

device = torch.device("cpu")

def translate_sentence(model, sentence, src_tokenizer, src_vocab, tgt_vocab, max_len=50):
    model.eval()
    
    # Tokenize and convert to lowercase
    src_tokens = [token.lower() for token in src_tokenizer(sentence)]
    print(f"Tokens: {src_tokens}")  # Debug tokenization
    
    # Convert tokens to indices and add BOS/EOS
    src_indices = [2] + [src_vocab.get(token, 1) for token in src_tokens] + [3]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    tgt_indices = [2]  # Start with BOS token
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        
        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_tensor, tgt_tensor, 0)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask,
                         src_padding_mask, tgt_padding_mask)
            
            # Apply temperature scaling to reduce overconfidence
            output = output / 1.5  # Add temperature scaling
            
            # Get top k predictions instead of just the top one
            probs = torch.softmax(output[0, -1], dim=-1)
            top_k = torch.topk(probs, k=5)
            next_word_idx = top_k.indices[0].item()  # Take the most likely token
            
        if next_word_idx == 3 or len(tgt_indices) >= max_len:  # EOS token or max length
            break
            
        tgt_indices.append(next_word_idx)
    
    # Convert indices back to words using reverse lookup
    result = []
    reverse_vocab = {idx: word for word, idx in tgt_vocab.items()}
    for idx in tgt_indices[1:]:  # Skip BOS
        if idx in reverse_vocab and reverse_vocab[idx] != '<unk>':
            result.append(reverse_vocab[idx])
    
    return ' '.join(result) if result else '<translation failed>'

def main():
    checkpoint_dir = "checkpoints"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('transformer_checkpoint_epoch_')]
    
    if not checkpoints:
        print("No checkpoints found! Please train the model first.")
        return
        
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(checkpoint_path)
    
    # Print training information
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Training Loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Load dataset to get vocabularies
    _, _, _, src_vocab, tgt_vocab = load_multi30k_dataset()
    
    # Initialize model with the same configuration as training
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,    # Match training configuration
        nhead=8,        # Match training configuration
        num_encoder_layer=2,  # Match training configuration
        num_decoder_layer=2,  # Match training configuration
        dim_feedforward=512,  # Match training configuration
        dropout=0.3
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    spacy_en, _ = load_tokenizers()
    en_tokenizer = lambda text: tokenize_en(text, spacy_en)
    
    # Test sentences
    test_sentences = [
        "Hello, how are you?",
        "I love programming.",
        "The weather is nice today.",
        "What time is it?",
        "See you tomorrow!"
    ]
    
    print("\nTesting translations:")
    print("-" * 50)
    for sentence in test_sentences:
        translation = translate_sentence(model, sentence, en_tokenizer, src_vocab, tgt_vocab)
        print(f"Input:  {sentence}")
        print(f"Output: {translation}")
        print("-" * 50)

if __name__ == "__main__":
    main()