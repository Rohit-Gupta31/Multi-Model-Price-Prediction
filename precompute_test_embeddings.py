"""
Pre-compute embeddings for TEST dataset using trained DeBERTa + ResNet50
EXACT SAME extraction methods as training to ensure consistency

This script uses:
- Text: EXACT same DeBERTaFeatureExtractor from replace_text_embeddings.py
- Image: EXACT same ResNetFeatureExtractor from precompute_embeddings.py
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    OUTPUT_DIR = "./embeddings_test"
    
    # Model paths (EXACT SAME as training)
    DEBERTA_MODEL_PATH = "/media/KB/Segformer_ML/student_resource/src/price_predictor_model_deberta_10epochs"
    
    # Parameters (EXACT SAME as training)
    MAX_LENGTH = 256  # Will result in 254 tokens after removing [CLS] and [SEP]
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# DeBERTa Feature Extractor (EXACT COPY from replace_text_embeddings.py)
# ============================================================================

class DeBERTaFeatureExtractor:
    """Extract features from text using trained DeBERTa"""
    def __init__(self, model_path, device='cuda'):
        self.device = device
        print(f"Loading trained DeBERTa from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"✅ DeBERTa loaded successfully!")
    
    def extract(self, texts, max_length=77):
        """
        Extract DeBERTa features from texts
        
        Returns:
            text_features: [batch_size, seq_len, 768] - Token features
            text_global: [batch_size, 768] - [CLS] token
            attention_mask: [batch_size, seq_len]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding='max_length',  # Pad to max_length for consistent shapes
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Token features (remove [CLS] and [SEP])
            text_features = outputs.last_hidden_state[:, 1:-1, :]  # [B, S-2, 768]
            
            # Global feature ([CLS] token)
            text_global = outputs.last_hidden_state[:, 0, :]  # [B, 768]
            
            # Adjust attention mask (remove [CLS] and [SEP])
            attention_mask = attention_mask[:, 1:-1]  # [B, S-2]
        
        return text_features, text_global, attention_mask


# ============================================================================
# ResNet Feature Extractor (EXACT COPY from precompute_embeddings.py)
# ============================================================================

class ResNetFeatureExtractor:
    """Extract regional and global features from images using ResNet"""
    def __init__(self, model_name='resnet50', device='cuda'):
        self.device = device
        
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=True).to(device)
        resnet.eval()
        
        # Freeze ResNet
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Extract layers for regional features (before final pooling)
        # ResNet50 layer4 output: [B, 2048, 7, 7]
        self.regional_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Extract layer for global features (after pooling)
        # ResNet50 avgpool + flatten: [B, 2048]
        self.global_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, images):
        """
        Extract ResNet features from images
        
        Args:
            images: List of PIL Images
        
        Returns:
            img_regional: [batch_size, 49, 2048] - Regional features (7×7 flattened)
            img_global: [batch_size, 2048] - Global features
        """
        # Convert PIL images to tensor
        images = torch.stack([self.transform(img) for img in images])
        images = images.to(self.device)
        
        with torch.no_grad():
            # Regional features: [B, 2048, 7, 7]
            regional_features = self.regional_extractor(images)
            
            # Reshape to [B, 2048, 49] then transpose to [B, 49, 2048]
            batch_size = regional_features.size(0)
            img_regional = regional_features.view(batch_size, 2048, -1).transpose(1, 2)
            
            # Global features: [B, 2048, 1, 1] -> [B, 2048]
            global_features = self.global_extractor(images)
            img_global = global_features.view(batch_size, -1)
        
        return img_regional, img_global


# ============================================================================
# Dataset
# ============================================================================

class TestDataset(Dataset):
    """Dataset for test data"""
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
            # Create a black image as placeholder
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get text
        text = str(row['catalog_content'])
        
        return {
            'image': image,
            'text': text,
            'index': idx
        }


def collate_fn(batch):
    """Custom collate function"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    indices = [item['index'] for item in batch]
    
    return {
        'images': images,
        'texts': texts,
        'indices': indices
    }


# ============================================================================
# Main Pre-computation Function
# ============================================================================

def precompute_test_embeddings():
    """Pre-compute all embeddings for test dataset"""
    
    print("="*70)
    print("PRE-COMPUTING TEST EMBEDDINGS")
    print("Using EXACT SAME extraction methods as training")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Test CSV: {Config.TEST_CSV}")
    print(f"Output dir: {Config.OUTPUT_DIR}")
    print(f"Trained DeBERTa: {Config.DEBERTA_MODEL_PATH}")
    print(f"Max text length: {Config.MAX_LENGTH} (254 tokens after [CLS]/[SEP] removal)")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print("="*70)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize feature extractors (EXACT SAME as training)
    deberta_extractor = DeBERTaFeatureExtractor(
        Config.DEBERTA_MODEL_PATH,
        device=Config.DEVICE
    )
    resnet_extractor = ResNetFeatureExtractor(
        model_name='resnet50',
        device=Config.DEVICE
    )
    
    print("\n✅ Feature extractors initialized")
    
    # Create dataset and dataloader
    test_dataset = TestDataset(Config.TEST_CSV, Config.IMAGE_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # Pre-allocate arrays
    num_samples = len(test_dataset)
    text_features_all = np.zeros((num_samples, 254, 768), dtype=np.float32)  # 256 - [CLS] - [SEP] = 254
    text_global_all = np.zeros((num_samples, 768), dtype=np.float32)
    attention_masks_all = np.zeros((num_samples, 254), dtype=np.float32)
    img_regional_all = np.zeros((num_samples, 49, 2048), dtype=np.float32)
    img_global_all = np.zeros((num_samples, 2048), dtype=np.float32)
    targets_all = np.full(num_samples, np.nan, dtype=np.float32)  # Test has no targets
    indices_all = []
    
    print(f"\nProcessing {num_samples} test samples...")
    print("="*70)
    
    # Extract features
    for batch in tqdm(test_loader, desc="Extracting features"):
        images = batch['images']
        texts = batch['texts']
        indices = batch['indices']
        
        # Extract text features with DeBERTa (EXACT SAME as training)
        text_features, text_global, attention_mask = deberta_extractor.extract(
            texts, 
            max_length=Config.MAX_LENGTH
        )
        
        # Extract image features with ResNet (EXACT SAME as training)
        img_regional, img_global = resnet_extractor.extract(images)
        
        # Store results
        for i, idx in enumerate(indices):
            text_features_all[idx] = text_features[i].cpu().numpy()
            text_global_all[idx] = text_global[i].cpu().numpy()
            attention_masks_all[idx] = attention_mask[i].cpu().numpy()
            img_regional_all[idx] = img_regional[i].cpu().numpy()
            img_global_all[idx] = img_global[i].cpu().numpy()
            indices_all.append(idx)
    
    # Save embeddings
    print("\n" + "="*70)
    print("Saving embeddings to disk...")
    print("="*70)
    np.save(os.path.join(Config.OUTPUT_DIR, 'text_features.npy'), text_features_all)
    np.save(os.path.join(Config.OUTPUT_DIR, 'text_global.npy'), text_global_all)
    np.save(os.path.join(Config.OUTPUT_DIR, 'attention_masks.npy'), attention_masks_all)
    np.save(os.path.join(Config.OUTPUT_DIR, 'img_regional.npy'), img_regional_all)
    np.save(os.path.join(Config.OUTPUT_DIR, 'img_global.npy'), img_global_all)
    np.save(os.path.join(Config.OUTPUT_DIR, 'targets.npy'), targets_all)
    
    # Save indices
    with open(os.path.join(Config.OUTPUT_DIR, 'indices.txt'), 'w') as f:
        for idx in indices_all:
            f.write(f"{idx}\n")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ TEST Pre-computation Complete!")
    print("="*70)
    print(f"Using Trained DeBERTa from: {Config.DEBERTA_MODEL_PATH}")
    print(f"Total samples processed: {num_samples}")
    print(f"\nSaved files:")
    print(f"  - text_features.npy:    [{text_features_all.shape}] = {text_features_all.nbytes / 1024**2:.2f} MB")
    print(f"  - text_global.npy:      [{text_global_all.shape}] = {text_global_all.nbytes / 1024**2:.2f} MB")
    print(f"  - attention_masks.npy:  [{attention_masks_all.shape}] = {attention_masks_all.nbytes / 1024**2:.2f} MB")
    print(f"  - img_regional.npy:     [{img_regional_all.shape}] = {img_regional_all.nbytes / 1024**2:.2f} MB")
    print(f"  - img_global.npy:       [{img_global_all.shape}] = {img_global_all.nbytes / 1024**2:.2f} MB")
    print(f"  - targets.npy:          [{targets_all.shape}] = {targets_all.nbytes / 1024**2:.2f} MB")
    print(f"  - indices.txt:          {len(indices_all)} lines")
    
    total_size = (text_features_all.nbytes + text_global_all.nbytes + 
                  attention_masks_all.nbytes + img_regional_all.nbytes + 
                  img_global_all.nbytes + targets_all.nbytes) / 1024**3
    print(f"\nTotal disk space: {total_size:.2f} GB")
    print(f"\n🚀 Now you can run inference with: python inference_mjave_exact.py")
    print("="*70)


if __name__ == "__main__":
    precompute_test_embeddings()
