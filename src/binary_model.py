"""
Binary Classification Models for Individual Action Mastery

This creates optimized binary classifiers for single actions:
- Much simpler than 101-class classification
- Expected: 95%+ accuracy in 1-2 hours
- Foundation for category-building system
"""

import torch
import torch.nn as nn
from torchvision import models

class BinaryActionClassifier(nn.Module):
    """
    Binary Action Classifier: [Target Action] vs [All Others]
    
    Key advantages for Individual Action Mastery:
    1. Only 2 classes → much easier to learn
    2. More focused feature extraction
    3. Higher accuracy per action
    4. Faster training time
    """
    
    def __init__(self, pretrained=True, dropout=0.5):
        super(BinaryActionClassifier, self).__init__()
        
        # Use R3D-18 as backbone (same as your original model)
        self.backbone = models.video.r3d_18(pretrained=pretrained)
        
        # Replace final layer with binary classifier
        # Original: 512 → 101 classes
        # New: 512 → 2 classes (target vs others)
        in_features = self.backbone.fc.in_features  # Should be 512
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)  # Binary classification
        )
        
        print(f"🔧 Binary classifier created: {in_features} → 2 classes")
        
    def forward(self, x):
        return self.backbone(x)

class EnhancedBinaryClassifier(nn.Module):
    """
    Enhanced binary classifier with additional regularization
    For even higher accuracy on individual actions
    """
    
    def __init__(self, pretrained=True, dropout=0.5, hidden_dim=256):
        super(EnhancedBinaryClassifier, self).__init__()
        
        # Use R3D-18 backbone
        self.backbone = models.video.r3d_18(pretrained=pretrained)
        
        # Remove the original FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final layer
        
        # Add enhanced classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 2)
        )
        
        print(f"🔧 Enhanced binary classifier: {in_features} → {hidden_dim} → 2 classes")
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_binary_model(model_type="simple", **kwargs):
    """
    Factory function to create binary classification models
    
    Args:
        model_type: "simple" or "enhanced"
        **kwargs: Additional arguments for the model
    
    Returns:
        Binary classification model (2 classes)
    """
    
    if model_type == "simple":
        model = BinaryActionClassifier(**kwargs)
        print("🎯 Created Simple Binary Classifier")
        
    elif model_type == "enhanced":
        model = EnhancedBinaryClassifier(**kwargs)
        print("🎯 Created Enhanced Binary Classifier")
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model

class BinaryModelTrainer:
    """
    Specialized trainer for binary action classification
    Optimized for high accuracy on individual actions
    """
    
    def __init__(self, model, device, class_weights=None):
        self.model = model
        self.device = device
        
        # Use weighted cross-entropy for class imbalance
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"⚖️ Using weighted loss with weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("📍 Using standard cross-entropy loss")
        
        # Optimizer with different learning rates for backbone vs classifier
        self.optimizer = self._create_optimizer()
        
    def _create_optimizer(self):
        """Create optimizer with differential learning rates"""
        
        # Lower learning rate for pretrained backbone
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-4},      # Lower LR for pretrained
            {'params': classifier_params, 'lr': 1e-3}     # Higher LR for new layers
        ])
        
        print("🎓 Created differential learning rate optimizer")
        return optimizer
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for clips, labels in train_loader:
            clips, labels = clips.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(clips)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, test_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # For detailed binary classification metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        with torch.no_grad():
            for clips, labels in test_loader:
                clips, labels = clips.to(self.device), labels.to(self.device)
                
                outputs = self.model(clips)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Binary classification metrics
                for pred, true in zip(predicted, labels):
                    if pred == 1 and true == 1:
                        true_positives += 1
                    elif pred == 1 and true == 0:
                        false_positives += 1
                    elif pred == 0 and true == 1:
                        false_negatives += 1
                    else:
                        true_negatives += 1
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return avg_loss, accuracy, precision, recall, f1_score

def save_binary_model(model, target_action, save_dir="C:\\ASH_PROJECT\\outputs\\binary_models"):
    """Save trained binary model with descriptive name"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f"{target_action.lower()}_binary_model.pth")
    torch.save(model.state_dict(), model_path)
    
    print(f"💾 Binary model saved: {model_path}")
    return model_path

def load_binary_model(target_action, model_type="simple", save_dir="C:\\ASH_PROJECT\\outputs\\binary_models"):
    """Load trained binary model"""
    import os
    
    model_path = os.path.join(save_dir, f"{target_action.lower()}_binary_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = get_binary_model(model_type)
    model.load_state_dict(torch.load(model_path))
    
    print(f"📁 Binary model loaded: {model_path}")
    return model

if __name__ == "__main__":
    # Test binary model creation
    print("🧪 Testing Binary Model Creation...")
    
    # Test simple model
    simple_model = get_binary_model("simple")
    
    # Test enhanced model  
    enhanced_model = get_binary_model("enhanced", hidden_dim=512, dropout=0.3)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 16, 112, 112)  # [batch, channels, frames, height, width]
    
    print("\n🔍 Testing forward pass...")
    with torch.no_grad():
        simple_output = simple_model(dummy_input)
        enhanced_output = enhanced_model(dummy_input)
        
        print(f"Simple model output: {simple_output.shape}")
        print(f"Enhanced model output: {enhanced_output.shape}")
        
        # Should be [batch_size, 2] for binary classification
        assert simple_output.shape == (2, 2), f"Expected (2, 2), got {simple_output.shape}"
        assert enhanced_output.shape == (2, 2), f"Expected (2, 2), got {enhanced_output.shape}"
    
    print("✅ Binary model test completed!")
    print("🎯 Ready for basketball binary classification training!")