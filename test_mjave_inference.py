"""
Test inference script for MJAVE price prediction
Loads pre-computed test embeddings and generates predictions for paper submission
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import the model
from mjave_price_model import MJAVEPriceModel


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    TEST_EMBEDDINGS_DIR = "./embeddings"
    TEST_CSV = "/media/KB/Segformer_ML/student_resource/dataset/train_filtered.csv"
    MODEL_PATH = "./best_mjave_price_model_fast.pth"  # Path to trained model
    OUTPUT_CSV = "papersubmissiontrain.csv"
    
    # Model parameters (must match training)
    TXT_HIDDEN_SIZE = 768
    IMG_HIDDEN_SIZE = 2048
    IMG_GLOBAL_SIZE = 2048
    IMG_NUM_REGIONS = 49
    ATTN_SIZE = 200
    TEXT_SEQ_LEN = 254
    DROPOUT = 0.5
    
    # Inference settings
    BATCH_SIZE = 512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Dataset
# ============================================================================

class TestEmbeddingsDataset(Dataset):
    """Dataset that loads pre-computed test embeddings"""
    def __init__(self, embeddings_dir):
        print(f"Loading test embeddings from {embeddings_dir}...")
        
        # Load all embeddings
        self.text_features = np.load(os.path.join(embeddings_dir, 'text_features.npy'))
        self.text_global = np.load(os.path.join(embeddings_dir, 'text_global.npy'))
        self.attention_masks = np.load(os.path.join(embeddings_dir, 'attention_masks.npy'))
        self.img_regional = np.load(os.path.join(embeddings_dir, 'img_regional.npy'))
        self.img_global = np.load(os.path.join(embeddings_dir, 'img_global.npy'))
        
        print(f"Loaded {len(self.text_features)} test samples")
        print(f"  text_features: {self.text_features.shape}")
        print(f"  text_global: {self.text_global.shape}")
        print(f"  img_regional: {self.img_regional.shape}")
        print(f"  img_global: {self.img_global.shape}")
        
        # Validate shapes
        assert self.text_features.shape[1] == 254, f"Expected 254 tokens, got {self.text_features.shape[1]}"
        assert self.img_regional.shape[1] == 49, f"Expected 49 regions, got {self.img_regional.shape[1]}"
        print("✓ Shape validation passed")
    
    def __len__(self):
        return len(self.text_features)
    
    def __getitem__(self, idx):
        return {
            'text_features': torch.from_numpy(self.text_features[idx]).float(),
            'text_global': torch.from_numpy(self.text_global[idx]).float(),
            'attention_mask': torch.from_numpy(self.attention_masks[idx]).float(),
            'img_regional': torch.from_numpy(self.img_regional[idx]).float(),
            'img_global': torch.from_numpy(self.img_global[idx]).float(),
        }


# ============================================================================
# Inference Function
# ============================================================================

def run_inference():
    """Run inference on test set and save predictions"""
    
    print("="*70)
    print("MJAVE Test Inference")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ Error: Model not found at {Config.MODEL_PATH}")
        print("Please train the model first using: python train_mjave_price_fast.py")
        return
    
    # Check if test embeddings exist
    if not os.path.exists(Config.TEST_EMBEDDINGS_DIR):
        print(f"❌ Error: Test embeddings not found at {Config.TEST_EMBEDDINGS_DIR}")
        print("Please compute test embeddings first using: python precompute_test_embeddings.py")
        return
    
    # Create dataset and dataloader (load embeddings first)
    test_dataset = TestEmbeddingsDataset(Config.TEST_EMBEDDINGS_DIR)
    num_embeddings = len(test_dataset)
    
    # Load test CSV to get sample IDs (must match embeddings count)
    print(f"\nLoading test CSV from: {Config.TEST_CSV}")
    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"CSV has {len(test_df)} samples, embeddings have {num_embeddings} samples")
    
    # Make sure they match
    if len(test_df) != num_embeddings:
        print(f"⚠️ WARNING: Mismatch! Taking first {num_embeddings} rows from CSV")
        test_df = test_df.head(num_embeddings)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must not shuffle to maintain order
        num_workers=4
    )
    
    # Load model
    print(f"\nLoading trained model from: {Config.MODEL_PATH}")
    model = MJAVEPriceModel(
        txt_hidden_size=Config.TXT_HIDDEN_SIZE,
        img_hidden_size=Config.IMG_HIDDEN_SIZE,
        img_global_size=Config.IMG_GLOBAL_SIZE,
        img_num_regions=Config.IMG_NUM_REGIONS,
        attn_size=Config.ATTN_SIZE,
        dropout=Config.DROPOUT,
        use_regional_gate=False  # Trained without regional gate
    ).to(Config.DEVICE)
    
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    # Handle both formats: dict with 'model_state_dict' or direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    if isinstance(checkpoint, dict):
        print(f"  - Best validation SMAPE: {checkpoint.get('best_smape', 'N/A')}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    else:
        print(f"  - Loaded direct state_dict (no metadata)")
    
    # Run inference
    print("\n" + "="*70)
    print("Running inference...")
    print("="*70)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Move to device
            text_features = batch['text_features'].to(Config.DEVICE)
            text_global = batch['text_global'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            img_regional = batch['img_regional'].to(Config.DEVICE)
            img_global = batch['img_global'].to(Config.DEVICE)
            
            # Forward pass
            log_price_pred = model(
                text_features=text_features,
                text_global_features=text_global,
                img_features=img_regional,
                img_global_features=img_global,
                attention_mask=attention_mask
            )
            
            # Keep predictions in LOG SPACE (same as training)
            # Convert to actual prices later for CSV output
            all_predictions.extend(log_price_pred.cpu().numpy().flatten().tolist())
    
    # Create submission dataframe
    print("\n" + "="*70)
    print("Creating submission file...")
    print("="*70)
    
    # Convert log predictions to actual prices for CSV
    import numpy as np
    actual_prices = np.expm1(np.array(all_predictions))
    
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'price': actual_prices
    })
    
    # Save to CSV
    submission_df.to_csv(Config.OUTPUT_CSV, index=False)
    
    print(f"✅ Predictions saved to: {Config.OUTPUT_CSV}")
    print(f"   Total samples: {len(submission_df)}")
    print(f"   Price range: ${submission_df['price'].min():.2f} - ${submission_df['price'].max():.2f}")
    print(f"   Mean price: ${submission_df['price'].mean():.2f}")
    print(f"   Median price: ${submission_df['price'].median():.2f}")
    
    # Show first few predictions
    print("\nFirst 10 predictions:")
    print(submission_df.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("🎉 Inference complete!")
    print("="*70)


if __name__ == "__main__":
    run_inference()
