import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.data import DataLoader
from dataset import load_multi30k_dataset, collate_batch
from encoder_decoder import Transformer, create_mask

# Force CPU usage for stability
device = torch.device("cpu")
print("Using CPU device for training")

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using {device} device for training")

def train_epoch(model, dataloader, optimizer, criterion, pad_idx):
    model.train()
    total_loss = 0
    
    try:
        for i, (src, tgt) in enumerate(dataloader):
            try:
                # Move data to device
                src = src.to(device)
                tgt = tgt.to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)
                
                # Move masks to device if they exist
                if src_mask is not None:
                    src_mask = src_mask.to(device)
                if tgt_mask is not None:
                    tgt_mask = tgt_mask.to(device)
                if src_padding_mask is not None:
                    src_padding_mask = src_padding_mask.to(device)
                if tgt_padding_mask is not None:
                    tgt_padding_mask = tgt_padding_mask.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(src, tgt_input, src_mask, tgt_mask, 
                             src_padding_mask, tgt_padding_mask)
                loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                if i % 5 == 0:  # Print even more frequently
                    print(f"Batch {i}, Loss: {loss.item():.4f}")
                
            except RuntimeError as e:
                print(f"Error in batch {i}: {e}")
                raise e
                
    except Exception as e:
        print(f"Error during training: {e}")
        raise e
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, pad_idx):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)
            
            if src_mask is not None:
                src_mask = src_mask.to(device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(device)
            if src_padding_mask is not None:
                src_padding_mask = src_padding_mask.to(device)
            if tgt_padding_mask is not None:
                tgt_padding_mask = tgt_padding_mask.to(device)
            
            output = model(src, tgt_input, src_mask, tgt_mask,
                         src_padding_mask, tgt_padding_mask)
            
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    # Enhanced training parameters for longer training
    BATCH_SIZE = 32     # Smaller batch size for better generalization
    d_model = 256      # Increased model capacity
    nhead = 8          # More attention heads
    num_encoder_layers = 2  # More layers
    num_decoder_layers = 2
    dim_feedforward = 512   # Larger feedforward network
    dropout = 0.3    # Increased dropout to prevent overfitting
    num_epochs = 3  # Much longer training time
    checkpoint_dir = "/Users/uddeshyabarnwal/Dev/Transformer/encoder_decoder/checkpoints"
    
    # Create checkpoint directory
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load dataset first to get vocabularies
    print("Loading dataset...")
    train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k_dataset()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, pad_idx=src_vocab['<pad>'])
    )
    
    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate_batch(x, pad_idx=src_vocab['<pad>'])
    )
    
    # Initialize model
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layer=num_encoder_layers,
        num_decoder_layer=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # Setup criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,    # 10% warm-up
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        try:
            # Training
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion, src_vocab['<pad>'])
            
            # Validation
            val_loss = evaluate(model, val_dataloader, criterion, src_vocab['<pad>'])
            
            end_time = time.time()
            
            print(f"Epoch: {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Time: {end_time - start_time:.2f}s")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'transformer_checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {e}")
            error_checkpoint_path = os.path.join(checkpoint_dir, 'transformer_checkpoint_error.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss if 'train_loss' in locals() else None,
            }, error_checkpoint_path)
            raise e

if __name__ == "__main__":
    main()