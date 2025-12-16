"""
Pre-compute BERT and ResNet embeddings for faster training

This script extracts all text and image features once and saves them to disk.
After running this, training will be 5-10x faster!

Usage:
    python precompute_embeddings.py

Output:
    ./embeddings/text_features.npy       - [N, 75, 768] BERT token features
    ./embeddings/text_global.npy         - [N, 768] BERT [CLS] features
    ./embeddings/attention_masks.npy     - [N, 75] Attention masks
    ./embeddings/img_regional.npy        - [N, 49, 2048] ResNet regional features
    ./embeddings/img_global.npy          - [N, 2048] ResNet global features
    ./embeddings/targets.npy             - [N] Log prices
    ./embeddings/indices.txt             - Valid sample indices
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    CSV_FILE = '/media/KB/Segformer_ML/student_resource/dataset/test.csv'
    IMAGE_DIR = '/media/KB/Segformer_ML/student_resource/images_test'
    OUTPUT_DIR = './embeddings_bert_image'
    
    # Parameters
    MAX_PRICE = 100
    MAX_LENGTH = 77
    BATCH_SIZE = 32  # Batch size for feature extraction
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Feature Extractors (Same as training)
# ============================================================================

class BERTFeatureExtractor:
    """Extract features from text using BERT/DeBERTa"""
    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        self.device = device
        # Use AutoTokenizer and AutoModel to support both BERT and DeBERTa
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def extract(self, texts, max_length=77):
        """
        Extract BERT features from texts
        
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
# Main Pre-computation Script
# ============================================================================

def precompute_embeddings():
    """Pre-compute all embeddings and save to disk"""
    
    print("="*70)
    print("MJAVE Pre-compute Embeddings Script")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print()
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(Config.CSV_FILE)
    df = df[df['price'] < Config.MAX_PRICE]
    df['log_price'] = np.log1p(df['price'])
    print(f"Total samples: {len(df)}")
    
    # Filter valid images and prepare data
    print("Filtering valid images...")
    valid_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        image_url = str(row['image_link'])
        image_name = image_url.split('/')[-1]
        image_path = os.path.join(Config.IMAGE_DIR, image_name)
        
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    img.verify()
                
                # Get text
                text = str(row['catalog_content']).strip()
                if not text or text.lower() in ['nan', 'none', '']:
                    text = "product image"
                
                valid_data.append({
                    'idx': idx,
                    'image_path': image_path,
                    'text': text,
                    'log_price': row['log_price']
                })
            except:
                continue
    
    print(f"Valid samples: {len(valid_data)}")
    
    if len(valid_data) == 0:
        print("ERROR: No valid data found!")
        return
    
    # Initialize feature extractors
    print("\nLoading DeBERTa (trained) and ResNet models...")
    # Use your trained DeBERTa model instead of base BERT
    deberta_model_path = "/media/KB/Segformer_ML/student_resource/src/price_predictor_model_deberta"
    bert_extractor = BERTFeatureExtractor(model_name=deberta_model_path, device=Config.DEVICE)
    resnet_extractor = ResNetFeatureExtractor(device=Config.DEVICE)
    
    # Pre-allocate arrays
    num_samples = len(valid_data)
    text_features_all = np.zeros((num_samples, 75, 768), dtype=np.float32)
    text_global_all = np.zeros((num_samples, 768), dtype=np.float32)
    attention_masks_all = np.zeros((num_samples, 75), dtype=np.float32)
    img_regional_all = np.zeros((num_samples, 49, 2048), dtype=np.float32)
    img_global_all = np.zeros((num_samples, 2048), dtype=np.float32)
    targets_all = np.zeros(num_samples, dtype=np.float32)
    indices_all = []
    
    # Extract features in batches
    print("\nExtracting features...")
    batch_size = Config.BATCH_SIZE
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = valid_data[start_idx:end_idx]
        
        # Prepare batch
        texts = [item['text'] for item in batch_data]
        images = []
        for item in batch_data:
            try:
                img = Image.open(item['image_path']).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224), color='black')
            images.append(img)
        
        # Extract text features
        text_features, text_global, attention_mask = bert_extractor.extract(texts, Config.MAX_LENGTH)
        
        # Extract image features
        img_regional, img_global = resnet_extractor.extract(images)
        
        # Store in arrays
        batch_len = end_idx - start_idx
        text_features_all[start_idx:end_idx] = text_features.cpu().numpy()
        text_global_all[start_idx:end_idx] = text_global.cpu().numpy()
        attention_masks_all[start_idx:end_idx] = attention_mask.cpu().numpy()
        img_regional_all[start_idx:end_idx] = img_regional.cpu().numpy()
        img_global_all[start_idx:end_idx] = img_global.cpu().numpy()
        
        # Store targets and indices
        for i, item in enumerate(batch_data):
            targets_all[start_idx + i] = item['log_price']
            indices_all.append(item['idx'])
    
    # Save to disk
    print("\nSaving embeddings to disk...")
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
    print("✅ Pre-computation Complete!")
    print("="*70)
    print(f"Using Trained DeBERTa from: {deberta_model_path}")
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
    print(f"\n🚀 Now you can run: python train_mjave_price_fast.py")
    print("="*70)


if __name__ == "__main__":
    precompute_embeddings()
