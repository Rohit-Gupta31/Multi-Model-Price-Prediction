"""
MJAVE-inspired Multimodal Model for Price Prediction (PyTorch)

Adapted from JAVE (Joint Attribute-Value Extraction) model:
Original: Attribute extraction from product text+images
Modified: Price regression from product text+images

Architecture:
1. Text Self-Attention: Text tokens attend to each other
2. Cross-Modal Attention: Text attends to image regions (with global gate)
3. Regional Gate: Filters relevant image regions based on text features
4. Regression Head: Predicts price from fused multimodal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TextSelfAttention(nn.Module):
    """Self-attention layer for text features"""
    def __init__(self, txt_hidden_size, attn_size, dropout=0.1):
        super().__init__()
        self.attn_size = attn_size
        
        self.W_q = nn.Linear(txt_hidden_size, attn_size, bias=False)
        self.W_k = nn.Linear(txt_hidden_size, attn_size, bias=False)
        self.W_v = nn.Linear(txt_hidden_size, attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features, attention_mask=None):
        """
        Args:
            text_features: [batch_size, seq_len, txt_hidden_size]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for padding)
        Returns:
            attended_text: [batch_size, seq_len, attn_size]
        """
        Q = self.W_q(text_features)  # [B, S1, D]
        K = self.W_k(text_features)  # [B, S1, D]
        V = self.W_v(text_features)  # [B, S1, D]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_size)  # [B, S1, S1]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to attention mask (1 -> 0, 0 -> -inf)
            mask_expanded = attention_mask.unsqueeze(1)  # [B, 1, S1]
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # [B, S1, S1]
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, V)  # [B, S1, D]
        
        return attended


class CrossModalAttention(nn.Module):
    """Cross-modal attention: Text (query) attends to Image (key, value)"""
    def __init__(self, txt_hidden_size, img_hidden_size, attn_size, dropout=0.1):
        super().__init__()
        self.attn_size = attn_size
        
        self.W_q = nn.Linear(txt_hidden_size, attn_size, bias=False)
        self.W_k = nn.Linear(img_hidden_size, attn_size, bias=False)
        self.W_v = nn.Linear(img_hidden_size, attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features, img_features):
        """
        Args:
            text_features: [batch_size, seq_len, txt_hidden_size]
            img_features: [batch_size, num_regions, img_hidden_size]
        Returns:
            cross_attended: [batch_size, seq_len, attn_size]
            attn_scores: [batch_size, seq_len, num_regions]
        """
        Q = self.W_q(text_features)   # [B, S1, D]
        K = self.W_k(img_features)    # [B, S2, D]
        V = self.W_v(img_features)    # [B, S2, D]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_size)  # [B, S1, S2]
        attn_weights = F.softmax(scores, dim=-1)  # [B, S1, S2]
        attn_weights = self.dropout(attn_weights)
        
        cross_attended = torch.matmul(attn_weights, V)  # [B, S1, D]
        
        return cross_attended, scores


class GlobalGate(nn.Module):
    """Global gate to modulate cross-modal attention using global image features"""
    def __init__(self, txt_hidden_size, img_global_size):
        super().__init__()
        self.text_proj = nn.Linear(txt_hidden_size, 1, bias=False)
        self.img_proj = nn.Linear(img_global_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, text_features, img_global_features):
        """
        Args:
            text_features: [batch_size, seq_len, txt_hidden_size]
            img_global_features: [batch_size, img_global_size]
        Returns:
            gate: [batch_size, seq_len, 1]
        """
        d1 = self.text_proj(text_features)  # [B, S1, 1]
        d2 = self.img_proj(img_global_features).unsqueeze(1)  # [B, 1, 1]
        
        gate = torch.sigmoid(d1 + d2 + self.bias)  # [B, S1, 1]
        
        return gate


class RegionalGate(nn.Module):
    """Regional gate to select relevant image regions based on text"""
    def __init__(self, txt_hidden_size, img_hidden_size):
        super().__init__()
        self.text_proj = nn.Linear(txt_hidden_size, 1, bias=False)
        self.img_proj = nn.Linear(img_hidden_size, 1, bias=False)
        
    def forward(self, text_global, img_features):
        """
        Args:
            text_global: [batch_size, txt_hidden_size] - pooled text features
            img_features: [batch_size, num_regions, img_hidden_size]
        Returns:
            gate: [batch_size, 1, num_regions]
        """
        d1 = self.text_proj(text_global)  # [B, 1]
        d2 = self.img_proj(img_features).squeeze(-1)  # [B, S2]
        
        # Expand d1 to [B, 1] and d2 is [B, S2]
        # We want [B, 1, S2] output
        gate = torch.sigmoid(d1.unsqueeze(-1) + d2.unsqueeze(1))  # [B, 1, S2]
        
        return gate


class MJAVEPriceModel(nn.Module):
    """
    MJAVE-inspired Multimodal Model for Price Prediction
    
    Architecture Flow:
    1. Text Self-Attention: Enhance text representations
    2. Cross-Modal Attention: Text attends to image regions
    3. Global Gate: Modulate cross-attention with global image
    4. Regional Gate: Filter relevant image regions
    5. Fusion: Combine text + gated image features
    6. Regression: Predict price from fused features
    """
    def __init__(self, 
                 txt_hidden_size=768,      # BERT hidden size
                 img_hidden_size=2048,     # ResNet hidden size
                 img_global_size=2048,     # Global image feature size
                 img_num_regions=49,       # Number of image regions (7×7)
                 attn_size=512,            # Attention hidden size
                 use_global_gate=True,     # Use global image gate
                 use_regional_gate=True,   # Use regional image gate
                 dropout=0.3):
        super().__init__()
        
        self.txt_hidden_size = txt_hidden_size
        self.img_hidden_size = img_hidden_size
        self.img_global_size = img_global_size
        self.img_num_regions = img_num_regions
        self.attn_size = attn_size
        self.use_global_gate = use_global_gate
        self.use_regional_gate = use_regional_gate
        
        # Text self-attention
        self.text_self_attn = TextSelfAttention(txt_hidden_size, attn_size, dropout)
        
        # Cross-modal attention (text -> image)
        self.cross_modal_attn = CrossModalAttention(txt_hidden_size, img_hidden_size, attn_size, dropout)
        
        # Gates
        if use_global_gate:
            self.global_gate = GlobalGate(txt_hidden_size, img_global_size)
        
        if use_regional_gate:
            self.regional_gate = RegionalGate(txt_hidden_size, img_hidden_size)
        
        # Projection layers for fusion
        self.text_proj = nn.Linear(txt_hidden_size, attn_size, bias=False)
        self.text_attn_proj = nn.Linear(attn_size, attn_size, bias=False)
        self.cross_attn_proj = nn.Linear(attn_size, attn_size, bias=False)
        self.regional_attn_proj = nn.Linear(attn_size, attn_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(attn_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                text_features,         # [B, S1, D1] - BERT text features
                text_global_features,  # [B, D1] - BERT [CLS] token
                img_features,          # [B, S2, D2] - Regional image features
                img_global_features,   # [B, D2] - Global image features
                attention_mask=None):  # [B, S1] - Text attention mask
        """
        Forward pass
        
        Args:
            text_features: [batch_size, seq_len, txt_hidden_size]
            text_global_features: [batch_size, txt_hidden_size]
            img_features: [batch_size, num_regions, img_hidden_size]
            img_global_features: [batch_size, img_global_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            price_pred: [batch_size, 1] - Predicted log price
        """
        batch_size = text_features.size(0)
        seq_len = text_features.size(1)
        
        # Step 1: Text self-attention
        # Text tokens attend to each other for better representation
        text_self_attended = self.text_self_attn(text_features, attention_mask)  # [B, S1, D]
        
        # Step 2: Cross-modal attention (text queries image)
        # Each text token attends to image regions
        cross_attended, cross_attn_scores = self.cross_modal_attn(
            text_features, img_features
        )  # [B, S1, D], [B, S1, S2]
        
        # Step 3: Global gate modulation
        if self.use_global_gate:
            # Use global image features to modulate cross-attention
            global_gate = self.global_gate(text_features, img_global_features)  # [B, S1, 1]
            cross_attended_gated = global_gate * cross_attended  # [B, S1, D]
        else:
            cross_attended_gated = cross_attended
        
        # Step 4: Regional gate modulation
        if self.use_regional_gate:
            # Use text to filter relevant image regions
            text_pooled = text_features.mean(dim=1)  # [B, D1] - Simple pooling
            regional_gate = self.regional_gate(text_pooled, img_features)  # [B, 1, S2]
            
            # Expand regional gate to match cross_attn_scores dimensions
            # regional_gate: [B, 1, S2] -> [B, S1, S2]
            regional_gate_expanded = regional_gate.expand(-1, cross_attn_scores.size(1), -1)
            
            # Apply regional gate to attention scores
            gated_attn_scores = cross_attn_scores * regional_gate_expanded  # [B, S1, S2]
            gated_attn_weights = F.softmax(gated_attn_scores, dim=-1)  # [B, S1, S2]
            
            # Recompute attended features with gated attention
            V = self.cross_modal_attn.W_v(img_features)  # [B, S2, D]
            regional_attended = torch.matmul(gated_attn_weights, V)  # [B, S1, D]
        else:
            regional_attended = torch.zeros_like(cross_attended_gated)
        
        # Step 5: Feature fusion
        # Combine text, text self-attention, cross-attention, and regional attention
        text_proj = self.dropout(self.text_proj(text_features))  # [B, S1, D]
        text_attn_proj = self.dropout(self.text_attn_proj(text_self_attended))  # [B, S1, D]
        cross_attn_proj = self.dropout(self.cross_attn_proj(cross_attended_gated))  # [B, S1, D]
        regional_attn_proj = self.dropout(self.regional_attn_proj(regional_attended))  # [B, S1, D]
        
        # Fused multimodal features
        fused_features = text_proj + text_attn_proj + cross_attn_proj + regional_attn_proj  # [B, S1, D]
        
        # Step 6: Pooling for regression
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)  # [B, S1, 1]
            fused_features = fused_features * mask_expanded
            pooled_features = fused_features.sum(dim=1) / mask_expanded.sum(dim=1)  # [B, D]
        else:
            pooled_features = fused_features.mean(dim=1)  # [B, D]
        
        # Step 7: Price prediction
        price_pred = self.regression_head(pooled_features).squeeze(-1)  # [B]
        
        return price_pred


# ============================================================================
# Example Usage
# ============================================================================

def compute_smape(predictions, labels):
    """Compute Symmetric Mean Absolute Percentage Error"""
    import numpy as np
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Convert log predictions back to actual prices
    pred_prices = np.expm1(predictions)
    actual_prices = np.expm1(labels)
    
    numerator = np.abs(pred_prices - actual_prices)
    denominator = (np.abs(actual_prices) + np.abs(pred_prices)) / 2
    
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100


def example_usage():
    """Example of how to use the model"""
    
    # Model configuration
    model = MJAVEPriceModel(
        txt_hidden_size=768,      # BERT-base hidden size
        img_hidden_size=2048,     # ResNet50 feature size
        img_global_size=2048,     # Global image feature size
        img_num_regions=49,       # 7×7 image regions
        attn_size=512,            # Attention hidden dimension
        use_global_gate=True,     # Enable global gate
        use_regional_gate=True,   # Enable regional gate
        dropout=0.3
    )
    
    # Example inputs
    batch_size = 16
    seq_len = 50
    num_regions = 49
    
    # Text features from BERT
    text_features = torch.randn(batch_size, seq_len, 768)
    text_global_features = torch.randn(batch_size, 768)  # [CLS] token
    
    # Image features from ResNet
    img_features = torch.randn(batch_size, num_regions, 2048)
    img_global_features = torch.randn(batch_size, 2048)
    
    # Attention mask (1 for valid tokens, 0 for padding)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 40:] = 0  # Simulate padding
    
    # Forward pass
    price_pred = model(
        text_features=text_features,
        text_global_features=text_global_features,
        img_features=img_features,
        img_global_features=img_global_features,
        attention_mask=attention_mask
    )
    
    print(f"Predicted prices (log): {price_pred.shape}")  # [16]
    
    # Training loss
    target_prices = torch.randn(batch_size)  # Log-transformed target prices
    criterion = nn.MSELoss()
    loss = criterion(price_pred, target_prices)
    
    print(f"MSE Loss: {loss.item():.4f}")
    
    # SMAPE evaluation (after converting back from log space)
    import numpy as np
    smape = compute_smape(
        price_pred.detach().cpu().numpy(),
        target_prices.detach().cpu().numpy()
    )
    print(f"SMAPE: {smape:.2f}%")


if __name__ == "__main__":
    example_usage()
