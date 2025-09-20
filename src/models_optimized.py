import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.models.video as video_models

class MobileNet3D(nn.Module):
    """Lightweight 3D CNN based on MobileNet principles"""
    def __init__(self, num_classes=101, input_channels=3, dropout=0.2):
        super(MobileNet3D, self).__init__()
        
        # 3D Depthwise Separable Convolutions
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(32)
        
        # Depthwise separable blocks
        self.dw_blocks = nn.ModuleList([
            self._make_dw_block(32, 64, (1, 2, 2)),
            self._make_dw_block(64, 128, (2, 2, 2)),
            self._make_dw_block(128, 256, (2, 2, 2)),
            self._make_dw_block(256, 512, (2, 2, 2)),
        ])
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_dw_block(self, in_channels, out_channels, stride):
        """Create a depthwise separable 3D block"""
        return nn.Sequential(
            # Depthwise 3D conv
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise 3D conv
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: [B, C, T, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        
        for block in self.dw_blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class Efficient3DCNN(nn.Module):
    """Very lightweight 3D CNN for fast training"""
    def __init__(self, num_classes=101, input_channels=3):
        super(Efficient3DCNN, self).__init__()
        
        # Efficient 3D feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(input_channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Block 4 - Temporal reduction
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet2Plus1D_Light(nn.Module):
    """Lightweight (2+1)D ResNet - separates spatial and temporal convolutions"""
    def __init__(self, num_classes=101):
        super(ResNet2Plus1D_Light, self).__init__()
        
        # Use pretrained R(2+1)D but make it lighter
        try:
            # Use smaller version if available
            self.backbone = video_models.r2plus1d_18(pretrained=True)
        except:
            # Fallback to building our own
            self.backbone = self._build_light_r2plus1d()
        
        # Replace the final layer
        if hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            # Custom classifier
            self.classifier = nn.Linear(512, num_classes)
    
    def _build_light_r2plus1d(self):
        """Build a lightweight (2+1)D model"""
        return nn.Sequential(
            # Spatial conv + temporal conv blocks
            nn.Conv3d(3, 32, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # More (2+1)D blocks
            self._make_2plus1d_block(32, 64, stride=(2, 2, 2)),
            self._make_2plus1d_block(64, 128, stride=(2, 2, 2)),
            self._make_2plus1d_block(128, 256, stride=(2, 2, 2)),
            
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
    
    def _make_2plus1d_block(self, in_channels, out_channels, stride=(1, 1, 1)):
        """Create a (2+1)D block"""
        return nn.Sequential(
            # Spatial convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), 
                     stride=(1, stride[1], stride[2]), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # Temporal convolution
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), 
                     stride=(stride[0], 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if hasattr(self.backbone, 'fc'):
            return self.backbone(x)
        else:
            features = self.backbone(x)
            return self.classifier(features)

def get_optimized_model(model_type="efficient", num_classes=101, pretrained=True, freeze_backbone=False):
    """Get an optimized model for faster training"""
    
    if model_type == "mobilenet3d":
        print("🔧 Using MobileNet3D (lightest, fastest training)")
        return MobileNet3D(num_classes=num_classes)
    
    elif model_type == "efficient":
        print("🔧 Using Efficient3DCNN (very light, good balance)")
        return Efficient3DCNN(num_classes=num_classes)
    
    elif model_type == "r2plus1d_light":
        print("🔧 Using ResNet(2+1)D Light (moderate complexity)")
        return ResNet2Plus1D_Light(num_classes=num_classes)
    
    elif model_type == "r3d18_transfer":
        print("🔧 Using R3D-18 with TRANSFER LEARNING (frozen backbone)")
        # Load pre-trained R3D-18
        model = video_models.r3d_18(pretrained=pretrained)
        
        # Freeze all backbone layers if requested
        if freeze_backbone:
            print("🧊 Freezing backbone layers - only training classifier")
            for name, param in model.named_parameters():
                if 'fc' not in name:  # Don't freeze final classifier
                    param.requires_grad = False
        
        # Replace final classifier
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Total params: {total_params:,} | Trainable: {trainable_params:,}")
        
        return model
    
    elif model_type == "r3d18_light":
        print("🔧 Using R3D-18 with optimizations")
        # Original R3D-18 but with some optimizations
        model = video_models.r3d_18(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    return total_params, trainable_params

def get_model_comparison():
    """Compare different model architectures"""
    print("\n📊 Model Comparison:")
    print("=" * 60)
    
    models_info = {
        "MobileNet3D": {"params": "~2-5M", "speed": "Fastest", "accuracy": "Good"},
        "Efficient3DCNN": {"params": "~1-3M", "speed": "Very Fast", "accuracy": "Good"},
        "R(2+1)D Light": {"params": "~5-10M", "speed": "Fast", "accuracy": "Better"},
        "R3D-18 Original": {"params": "~33M", "speed": "Slow", "accuracy": "Best"}
    }
    
    for name, info in models_info.items():
        print(f"{name:20} | Params: {info['params']:8} | Speed: {info['speed']:9} | Accuracy: {info['accuracy']}")
    
    print("=" * 60)
    print("💡 Recommendation: Start with 'efficient' or 'mobilenet3d' for faster training!")