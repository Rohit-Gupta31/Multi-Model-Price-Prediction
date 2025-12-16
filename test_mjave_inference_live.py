"""
Test inference script for MJAVE price prediction - LIVE MODE (no pre-computed embeddings)
Computes embeddings on-the-fly to save disk space
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DebertaV2Tokenizer, DebertaV2Model
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# Import the model
from mjave_price_model import MJAVEPriceModel

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    DATA_DIR = "/media/KB/Segformer_ML/student_resource/dataset"
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    IMAGE_DIR = "/media/KB/Segformer_ML/student_resource/images_test"
    MODEL_PATH = "./best_mjave_price_model_fast.pth"  # Path to trained MJAVE model
    DEBERTA_MODEL_PATH = "/media/KB/Segformer_ML/student_resource/src/price_predictor_model_deberta_10epochs"  # Path to trained DeBERTa model
    OUTPUT_CSV = "livetestsubmission.csv"
    print(DEBERTA_MODEL_PATH)
    # Model parameters (must match training)
    TXT_HIDDEN_SIZE = 768
    IMG_HIDDEN_SIZE = 2048
    IMG_GLOBAL_SIZE = 2048
    IMG_NUM_REGIONS = 49
    ATTN_SIZE = 200
    TEXT_SEQ_LEN = 254
    DROPOUT = 0.5
    MAX_LENGTH = 256  # Text tokenization length
    
    # Inference settings
    BATCH_SIZE = 16  # Smaller batch size since we're computing features on-the-fly
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Feature Extractors
# ============================================================================

class DeBERTaFeatureExtractor:
    """Extract text features using trained DeBERTa model"""
    def __init__(self, model_path, max_length=256, device='cuda'):
        print(f"Loading trained DeBERTa model from: {model_path}")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        self.model = DebertaV2Model.from_pretrained(model_path).to(device)
        self.model.eval()
        self.max_length = max_length
        self.device = device
        print(f"✓ DeBERTa model loaded on {device}")
    
    @torch.no_grad()
    def extract_features(self, texts):
        """
        Extract features from text
        Returns:
            token_features: [B, 254, 768] - all token embeddings (excluding [CLS] and [SEP])
            global_feature: [B, 768] - [CLS] token embedding
            attention_mask: [B, 254] - mask for valid tokens
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get DeBERTa outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [B, seq_len, 768]
        
        # Extract [CLS] token (position 0)
        global_feature = last_hidden_state[:, 0, :]  # [B, 768]
        
        # Remove [CLS] (position 0) and [SEP] (position -1)
        # Keep only the content tokens
        token_features = last_hidden_state[:, 1:-1, :]  # [B, seq_len-2, 768]
        token_attention_mask = attention_mask[:, 1:-1]  # [B, seq_len-2]
        
        return token_features, global_feature, token_attention_mask


class ResNetFeatureExtractor:
    """Extract image features using ResNet50"""
    def __init__(self, device='cuda'):
        print("Loading ResNet50...")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final FC layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        print(f"✓ ResNet50 loaded on {device}")
    
    @torch.no_grad()
    def extract_features(self, images):
        """
        Extract features from images
        Args:
            images: List of PIL Images
        Returns:
            regional_features: [B, 49, 2048] - 7x7 grid of regional features
            global_feature: [B, 2048] - global image feature
        """
        # Preprocess images
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Extract features
        features = self.model(batch)  # [B, 2048, 7, 7]
        
        # Regional features: reshape to [B, 2048, 49] then transpose to [B, 49, 2048]
        regional_features = features.view(features.size(0), features.size(1), -1)  # [B, 2048, 49]
        regional_features = regional_features.transpose(1, 2)  # [B, 49, 2048]
        
        # Global feature: average pooling over spatial dimensions
        global_feature = features.mean(dim=[2, 3])  # [B, 2048]
        
        return regional_features, global_feature


# ============================================================================
# Dataset
# ============================================================================

class TestDataset(Dataset):
    """Dataset for test data with images and text"""
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        print(f"Loaded {len(self.df)} test samples from {csv_path}")
        print(f"Loading images from: {image_dir}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Extract image filename from URL
        image_url = row['image_link']
        image_filename = image_url.split('/')[-1]
        image_path = os.path.join(self.image_dir, image_filename)
        
        # Try to load image, use black placeholder if corrupted/missing
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Create a black image as placeholder (224x224 RGB)
            print(f"Warning: Could not load {image_filename}, using black placeholder")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get text
        text = str(row['catalog_content'])
        
        # Get sample_id
        sample_id = row['sample_id']
        
        return {
            'image': image,
            'text': text,
            'sample_id': sample_id,
            'index': idx
        }


def collate_fn(batch):
    """Custom collate function to keep PIL images and strings"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    indices = [item['index'] for item in batch]
    
    return {
        'images': images,
        'texts': texts,
        'sample_ids': sample_ids,
        'indices': indices
    }


# ============================================================================
# Inference Function
# ============================================================================

def run_inference():
    """Run inference on test set with live feature extraction"""
    
    print("="*70)
    print("MJAVE Test Inference - LIVE MODE (No Pre-computed Embeddings)")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ Error: Model not found at {Config.MODEL_PATH}")
        print("Please train the model first using: python train_mjave_price_fast.py")
        return
    
    # Check if test CSV exists
    if not os.path.exists(Config.TEST_CSV):
        print(f"❌ Error: Test CSV not found at {Config.TEST_CSV}")
        return
    
    # Initialize feature extractors
    print("\nInitializing feature extractors...")
    deberta_extractor = DeBERTaFeatureExtractor(
        Config.DEBERTA_MODEL_PATH,
        max_length=Config.MAX_LENGTH,
        device=Config.DEVICE
    )
    resnet_extractor = ResNetFeatureExtractor(device=Config.DEVICE)
    
    # Create dataset and dataloader
    print(f"\nLoading test dataset...")
    test_dataset = TestDataset(Config.TEST_CSV, Config.IMAGE_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must not shuffle to maintain order
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # Load MJAVE model (without regional_gate since it wasn't used during training)
    print(f"\nLoading trained MJAVE model from: {Config.MODEL_PATH}")
    mjave_model = MJAVEPriceModel(
        txt_hidden_size=Config.TXT_HIDDEN_SIZE,
        img_hidden_size=Config.IMG_HIDDEN_SIZE,
        img_global_size=Config.IMG_GLOBAL_SIZE,
        img_num_regions=Config.IMG_NUM_REGIONS,
        attn_size=Config.ATTN_SIZE,
        dropout=Config.DROPOUT,
        use_regional_gate=False  # Model was trained without regional gate
    ).to(Config.DEVICE)
    
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            mjave_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ MJAVE model loaded successfully!")
            print(f"  - Best validation SMAPE: {checkpoint.get('best_smape', 'N/A')}")
            print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        else:
            # Checkpoint is a dict with other keys, try to load it directly
            mjave_model.load_state_dict(checkpoint)
            print(f"✓ MJAVE model loaded successfully!")
    else:
        # Checkpoint is the state dict itself
        mjave_model.load_state_dict(checkpoint)
        print(f"✓ MJAVE model loaded successfully!")
    
    mjave_model.eval()
    
    # Run inference
    print("\n" + "="*70)
    print("Running inference with live feature extraction...")
    print("="*70)
    
    all_predictions = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['images']
            texts = batch['texts']
            sample_ids = batch['sample_ids']
            
            # Extract features on-the-fly
            text_features, text_global, attention_mask = deberta_extractor.extract_features(texts)
            img_regional, img_global = resnet_extractor.extract_features(images)
            
            # Forward pass through MJAVE model
            log_price_pred = mjave_model(
                text_features=text_features,
                text_global_features=text_global,
                img_features=img_regional,
                img_global_features=img_global,
                attention_mask=attention_mask
            )
            
            # Keep predictions in LOG SPACE (same as training)
            # Convert to actual prices later for CSV output
            all_predictions.extend(log_price_pred.cpu().numpy().flatten().tolist())
            all_sample_ids.extend(sample_ids)
    
    # Create submission dataframe
    print("\n" + "="*70)
    print("Creating submission file...")
    print("="*70)
    
    # Convert log predictions to actual prices for CSV
    import numpy as np
    actual_prices = np.expm1(np.array(all_predictions))
    
    submission_df = pd.DataFrame({
        'sample_id': all_sample_ids,
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
