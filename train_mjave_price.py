"""
Training Script for MJAVE Price Prediction Model

This script shows how to integrate the MJAVE model with your price prediction task.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

from mjave_price_model import MJAVEPriceModel, compute_smape


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    CSV_FILE = '/media/KB/Segformer_ML/student_resource/dataset/train_filtered.csv'
    IMAGE_DIR = '/media/KB/Segformer_ML/student_resource/images'
    
    # Model parameters
    TXT_HIDDEN_SIZE = 768       # BERT-base
    IMG_HIDDEN_SIZE = 2048      # ResNet50
    IMG_GLOBAL_SIZE = 2048
    IMG_NUM_REGIONS = 49        # 7×7 grid
    ATTN_SIZE = 512
    
    # Training parameters
    BATCH_SIZE = 256  # Reduced from 16 to fit in remaining GPU memory
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    
    # Data parameters
    MAX_PRICE = 100
    TRAIN_SPLIT = 0.9
    MAX_LENGTH = 77
    
    # Model options
    USE_GLOBAL_GATE = True
    USE_REGIONAL_GATE = True
    DROPOUT = 0.3


# ============================================================================
# Feature Extractors
# ============================================================================

class BERTFeatureExtractor:
    """Extract features from text using BERT"""
    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Freeze BERT
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
            padding=True,
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
            images: List of PIL Images or tensor [B, 3, 224, 224]
        
        Returns:
            img_regional: [batch_size, 49, 2048] - Regional features (7×7 flattened)
            img_global: [batch_size, 2048] - Global features
        """
        # Convert PIL images to tensor if needed
        if isinstance(images, list):
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

class MJAVEPriceDataset(Dataset):
    """Dataset for MJAVE price prediction"""
    def __init__(self, csv_file, image_dir, max_price=100):
        # Load and filter data
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['price'] < max_price]
        self.df['log_price'] = np.log1p(self.df['price'])
        
        # Filter valid images
        valid_indices = []
        for idx, row in self.df.iterrows():
            image_url = str(row['image_link'])
            image_name = image_url.split('/')[-1]
            image_path = os.path.join(image_dir, image_name)
            
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    valid_indices.append(idx)
                except:
                    continue
        
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        self.image_dir = image_dir
        
        print(f"Loaded {len(self.df)} valid samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_url = str(row['image_link'])
        image_name = image_url.split('/')[-1]
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get text
        text = str(row['catalog_content']).strip()
        if not text or text.lower() in ['nan', 'none', '']:
            text = "product image"
        
        # Get target
        target = torch.tensor(row['log_price'], dtype=torch.float32)
        
        return {
            'image': image,
            'text': text,
            'target': target
        }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, bert_extractor, resnet_extractor, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image']
        texts = batch['text']
        batch_targets = batch['target'].to(device)
        
        # Extract features
        text_features, text_global, attention_mask = bert_extractor.extract(texts)
        img_regional, img_global = resnet_extractor.extract(images)
        
        # Forward pass
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


def validate_epoch(model, dataloader, criterion, bert_extractor, resnet_extractor, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image']
            texts = batch['text']
            batch_targets = batch['target'].to(device)
            
            # Extract features
            text_features, text_global, attention_mask = bert_extractor.extract(texts)
            img_regional, img_global = resnet_extractor.extract(images)
            
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
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize feature extractors
    print("Loading feature extractors...")
    bert_extractor = BERTFeatureExtractor(device=device)
    resnet_extractor = ResNetFeatureExtractor(device=device)
    
    # Initialize model
    print("Initializing MJAVE model...")
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
    
    # Create dataset
    print("Loading dataset...")
    dataset = MJAVEPriceDataset(Config.CSV_FILE, Config.IMAGE_DIR, Config.MAX_PRICE)
    
    # Split dataset
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Custom collate function to handle PIL images and text
    def custom_collate_fn(batch):
        """Custom collate to handle PIL images and text strings"""
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        targets = torch.stack([item['target'] for item in batch])
        
        return {
            'image': images,  # Keep as list of PIL images
            'text': texts,    # Keep as list of strings
            'target': targets
        }
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=8,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    # Training loop
    best_val_smape = float('inf')
    
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_smape = train_epoch(
            model, train_loader, optimizer, criterion,
            bert_extractor, resnet_extractor, device
        )
        
        # Validate
        val_loss, val_smape = validate_epoch(
            model, val_loader, criterion,
            bert_extractor, resnet_extractor, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train SMAPE: {train_smape:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val SMAPE: {val_smape:.2f}%")
        
        # Save best model
        if val_smape < best_val_smape:
            best_val_smape = val_smape
            torch.save(model.state_dict(), 'best_mjave_price_model.pth')
            print(f"✅ New best model saved! Val SMAPE: {val_smape:.2f}%")
    
    print(f"\nTraining completed!")
    print(f"Best validation SMAPE: {best_val_smape:.2f}%")


if __name__ == "__main__":
    main()
