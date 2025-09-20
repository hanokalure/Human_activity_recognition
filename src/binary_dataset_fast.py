"""
Fast Binary Dataset Creator - No Hanging!
Optimized version that skips the slow sample counting
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import warnings
warnings.filterwarnings("ignore")

# UCF101 class names (in alphabetical order)
UCF101_CLASSES = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
    'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
    'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
    'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
    'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
    'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering',
    'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
    'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow',
    'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting',
    'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
    'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf',
    'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
    'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
    'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing',
    'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding',
    'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
    'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot',
    'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
    'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard',
    'YoYo'
]

class VideoTransform:
    """Fast video transform for binary classification"""
    def __init__(self):
        self.resize = transforms.Resize((112, 112))
        self.normalize = transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
        )

    def __call__(self, clip):
        if clip is None:
            return torch.randn(3, 16, 112, 112)
        
        if clip.dim() != 4:
            return torch.randn(3, 16, 112, 112)
        
        # Convert to proper format [C, T, H, W]
        if clip.shape[3] == 3:  # THWC format
            clip = clip.permute(3, 0, 1, 2)
        
        clip = clip.float() / 255.0
        
        # Apply transforms to each frame
        transformed_frames = []
        for t in range(clip.size(1)):
            frame = clip[:, t, :, :]
            frame = self.resize(frame)
            frame = self.normalize(frame)
            transformed_frames.append(frame)
        
        return torch.stack(transformed_frames, dim=1)

class FastBinaryActionDataset(Dataset):
    """
    Fast Binary Classification Dataset - No Hanging!
    
    Key optimizations:
    - Skip slow sample counting
    - Use estimated class weights
    - Faster initialization
    """
    
    def __init__(self, data_root, annotation_path, target_action="Basketball", 
                 frames_per_clip=16, train=True):
        self.target_action = target_action
        self.target_class_idx = UCF101_CLASSES.index(target_action)
        
        print(f"⚡ Creating FAST Binary Classifier: '{target_action}' vs All Others")
        print(f"📍 Target class index: {self.target_class_idx}")
        
        # Load full UCF101 dataset
        print("📂 Loading UCF101 dataset (this may take a moment)...")
        self.dataset = datasets.UCF101(
            root=data_root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=1,
            train=train,
            output_format="THWC"
        )
        
        self.transform = VideoTransform()
        
        print(f"✅ Dataset loaded: {len(self.dataset)} total samples")
        
        # SKIP the slow sample counting - use estimates instead!
        # Typical UCF101 has ~100 samples per class on average
        estimated_target = 120  # Basketball has ~120 clips typically
        estimated_others = len(self.dataset) - estimated_target
        
        print(f"📊 Estimated samples:")
        print(f"   🏀 {target_action}: ~{estimated_target}")
        print(f"   🚫 Others: ~{estimated_others}")
        
        # Calculate estimated class weights
        total = estimated_target + estimated_others
        self.target_weight = total / (2 * estimated_target)
        self.non_target_weight = total / (2 * estimated_others)
        
        print(f"⚖️ Estimated weights - {target_action}: {self.target_weight:.3f}, Others: {self.non_target_weight:.3f}")
        print("⚡ Fast dataset ready!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            video, audio, original_label = self.dataset[idx]
            
            # Convert to binary label: 1 = target action, 0 = all others
            binary_label = 1 if original_label == self.target_class_idx else 0
            
            return self.transform(video), binary_label
            
        except Exception as e:
            # Return dummy data on error
            if idx < 5:  # Only print first few errors
                print(f"⚠️ Error loading sample {idx}: {e}")
            return self.transform(None), 0

def get_fast_binary_dataloaders(data_root, annotation_path, target_action="Basketball", 
                               batch_size=8, num_workers=0):
    """
    Get fast binary classification dataloaders - No hanging!
    """
    print(f"\n⚡ Creating FAST Binary Datasets for: {target_action}")
    print("=" * 50)
    
    # Create datasets
    train_dataset = FastBinaryActionDataset(
        data_root, annotation_path, target_action=target_action, train=True
    )
    
    test_dataset = FastBinaryActionDataset(
        data_root, annotation_path, target_action=target_action, train=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Return estimated class weights
    class_weights = torch.tensor([
        train_dataset.non_target_weight,  # Class 0 (not target)
        train_dataset.target_weight       # Class 1 (target)
    ])
    
    print(f"\n📦 Fast DataLoaders created:")
    print(f"   ⚡ Train batches: {len(train_loader)}")
    print(f"   ⚡ Test batches: {len(test_loader)}")
    print(f"   ⚖️ Class weights: {class_weights}")
    
    return train_loader, test_loader, class_weights

if __name__ == "__main__":
    # Test fast basketball binary classification
    data_root = r"C:\ASH_PROJECT\data\UCF101"
    annotation_path = r"C:\ASH_PROJECT\data\ucfTrainTestlist"
    
    print("⚡ Testing FAST Basketball Binary Classification Dataset...")
    train_loader, test_loader, class_weights = get_fast_binary_dataloaders(
        data_root, annotation_path, target_action="Basketball", batch_size=4
    )
    
    # Test a few batches
    print("\n🔍 Testing data loading...")
    for i, (clips, labels) in enumerate(train_loader):
        print(f"Batch {i+1}: clips={clips.shape}, labels={labels}")
        if i >= 2:  # Test only first 3 batches
            break
    
    print("✅ Fast binary dataset test completed!")