"""
Fast Training Script for MJAVE Price Prediction Model (Using Pre-computed Embeddings)

This script uses pre-computed BERT and ResNet features for 5-10x faster training!

Requirements:
    1. Run precompute_embeddings.py first to generate embeddings
    2. Embeddings should be in ./embeddings/ directory

Usage:
    python train_mjave_price_fast.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from mjave_price_model import MJAVEPriceModel, compute_smape


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    EMBEDDINGS_DIR = './embeddings'
    
    # Model parameters
    TXT_HIDDEN_SIZE = 768       # BERT-base (DeBERTa)
    IMG_HIDDEN_SIZE = 2048      # ResNet50
    IMG_GLOBAL_SIZE = 2048
    IMG_NUM_REGIONS = 49        # 7×7 grid
    ATTN_SIZE = 200             # Same as JAVE paper
    TEXT_SEQ_LEN = 254          # Max length 256 - [CLS] - [SEP] = 254
    
    # Training parameters
    BATCH_SIZE = 512  # Much larger batch size possible now! (was 256)
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.02
    
    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs
    MIN_DELTA = 0.1  # Minimum improvement in SMAPE (0.1%)
    
    # Data parameters
    TRAIN_SPLIT = 0.9
    
    # Model options
    USE_GLOBAL_GATE = False
    USE_REGIONAL_GATE = False
    DROPOUT = 0


# ============================================================================
# Dataset with Pre-computed Embeddings
# ============================================================================

class PrecomputedEmbeddingsDataset(Dataset):
    """Dataset that loads pre-computed embeddings from disk"""
    def __init__(self, embeddings_dir):
        print(f"Loading pre-computed embeddings from {embeddings_dir}...")
        
        # Load all embeddings
        self.text_features = np.load(os.path.join(embeddings_dir, 'text_features.npy'))
        self.text_global = np.load(os.path.join(embeddings_dir, 'text_global.npy'))
        self.attention_masks = np.load(os.path.join(embeddings_dir, 'attention_masks.npy'))
        self.img_regional = np.load(os.path.join(embeddings_dir, 'img_regional.npy'))
        self.img_global = np.load(os.path.join(embeddings_dir, 'img_global.npy'))
        self.targets = np.load(os.path.join(embeddings_dir, 'targets.npy'))
        
        print(f"Loaded {len(self.targets)} samples")
        print(f"  text_features: {self.text_features.shape}")
        print(f"  text_global: {self.text_global.shape}")
        print(f"  attention_masks: {self.attention_masks.shape}")
        print(f"  img_regional: {self.img_regional.shape}")
        print(f"  img_global: {self.img_global.shape}")
        print(f"  targets: {self.targets.shape}")
        
        # Validate shapes
        assert self.text_features.shape[1] == 254, f"Expected text_features with 254 tokens, got {self.text_features.shape[1]}"
        assert self.img_regional.shape[1] == 49, f"Expected 49 image regions, got {self.img_regional.shape[1]}"
        print("✓ Shape validation passed")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'text_features': torch.from_numpy(self.text_features[idx]).float(),
            'text_global': torch.from_numpy(self.text_global[idx]).float(),
            'attention_mask': torch.from_numpy(self.attention_masks[idx]).float(),
            'img_regional': torch.from_numpy(self.img_regional[idx]).float(),
            'img_global': torch.from_numpy(self.img_global[idx]).float(),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # All features are pre-computed, just move to device
        text_features = batch['text_features'].to(device)
        text_global = batch['text_global'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        img_regional = batch['img_regional'].to(device)
        img_global = batch['img_global'].to(device)
        batch_targets = batch['target'].to(device)
        
        # Forward pass (no feature extraction needed!)
        outputs = model(
            text_features=text_features,
            text_global_features=text_global,
            img_features=img_regional,
            img_global_features=img_global,
            attention_mask=attention_mask
        )
        
        loss = criterion(outputs, batch_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(batch_targets.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    smape = compute_smape(predictions, targets)
    
    return avg_loss, smape


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # All features are pre-computed, just move to device
            text_features = batch['text_features'].to(device)
            text_global = batch['text_global'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            img_regional = batch['img_regional'].to(device)
            img_global = batch['img_global'].to(device)
            batch_targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(
                text_features=text_features,
                text_global_features=text_global,
                img_features=img_regional,
                img_global_features=img_global,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs, batch_targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    smape = compute_smape(predictions, targets)
    
    return avg_loss, smape


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("="*70)
    print("MJAVE Fast Training with Pre-computed Embeddings")
    print("="*70)
    
    # Check if embeddings exist
    if not os.path.exists(Config.EMBEDDINGS_DIR):
        print(f"\n❌ ERROR: Embeddings directory not found: {Config.EMBEDDINGS_DIR}")
        print("Please run: python precompute_embeddings.py first!")
        return
    
    required_files = [
        'text_features.npy', 'text_global.npy', 'attention_masks.npy',
        'img_regional.npy', 'img_global.npy', 'targets.npy'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(Config.EMBEDDINGS_DIR, file)):
            print(f"\n❌ ERROR: Missing file: {file}")
            print("Please run: python precompute_embeddings.py first!")
            return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = PrecomputedEmbeddingsDataset(Config.EMBEDDINGS_DIR)
    
    # Split dataset
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders (no custom collate needed, all tensors!)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True  # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nBatch size: {Config.BATCH_SIZE} (much larger than before!)")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    print("\nInitializing MJAVE model...")
    model = MJAVEPriceModel(
        txt_hidden_size=Config.TXT_HIDDEN_SIZE,
        img_hidden_size=Config.IMG_HIDDEN_SIZE,
        img_global_size=Config.IMG_GLOBAL_SIZE,
        img_num_regions=Config.IMG_NUM_REGIONS,
        attn_size=Config.ATTN_SIZE,
        use_global_gate=Config.USE_GLOBAL_GATE,
        use_regional_gate=Config.USE_REGIONAL_GATE,
        dropout=Config.DROPOUT
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    # Training loop with early stopping
    best_val_smape = float('inf')
    patience_counter = 0
    early_stop = False
    
    print(f"\n{'='*70}")
    print(f"Starting training for {Config.NUM_EPOCHS} epochs (with early stopping)...")
    print(f"Early stopping patience: {Config.EARLY_STOPPING_PATIENCE} epochs")
    print(f"{'='*70}\n")
    
    for epoch in range(Config.NUM_EPOCHS):
        if early_stop:
            print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
            print(f"No improvement for {Config.EARLY_STOPPING_PATIENCE} consecutive epochs")
            break
            
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_smape = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_smape = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train SMAPE: {train_smape:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val SMAPE:   {val_smape:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Calculate overfitting gap
        overfitting_gap = val_smape - train_smape
        print(f"  Overfitting Gap: {overfitting_gap:.2f}%", end="")
        if overfitting_gap > 5:
            print(" ⚠️ (High!)")
        elif overfitting_gap > 2:
            print(" ⚠️")
        else:
            print(" ✅")
        
        # Early stopping logic
        if val_smape < best_val_smape - Config.MIN_DELTA:
            # Significant improvement
            best_val_smape = val_smape
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mjave_price_model_fast_false.pth')
            print(f"  ✅ New best model saved! Val SMAPE: {val_smape:.2f}%")
        else:
            # No significant improvement
            patience_counter += 1
            print(f"  ⏳ No improvement for {patience_counter}/{Config.EARLY_STOPPING_PATIENCE} epochs")
            
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                early_stop = True
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_smape': best_val_smape,
        }, 'checkpoint_latest_fast.pth')
    
    print(f"\n{'='*70}")
    print(f"✅ Training completed!")
    print(f"{'='*70}")
    print(f"Best validation SMAPE: {best_val_smape:.2f}%")
    print(f"Best model saved to: best_mjave_price_model_fast_false.pth")


if __name__ == "__main__":
    main()
