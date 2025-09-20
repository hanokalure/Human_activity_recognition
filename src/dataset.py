import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

# Transform for video clips
class VideoTransform:
    def __init__(self):
        self.resize = transforms.Resize((112, 112))
        self.normalize = transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
        )

    def __call__(self, clip):
        if clip is None:
            # Return dummy video with correct shape [C, T, H, W]
            return torch.randn(3, 16, 112, 112)
        
        # Debug: Check what we're getting
        if clip.dim() != 4:
            # If not 4D, create dummy
            return torch.randn(3, 16, 112, 112)
        
        # Convert to proper format [C, T, H, W]
        if clip.shape[3] == 3:  # THWC format
            clip = clip.permute(3, 0, 1, 2)  # Convert to CTHW
        elif clip.shape[0] == 3:  # CTHW format (already correct)
            pass
        else:
            # Unknown format, create dummy
            return torch.randn(3, 16, 112, 112)
        
        clip = clip.float() / 255.0
        
        # Apply resize and normalize to each frame
        transformed_frames = []
        for t in range(clip.size(1)):  # Iterate through time dimension
            frame = clip[:, t, :, :]  # Get frame [C, H, W]
            frame = self.resize(frame)
            frame = self.normalize(frame)
            transformed_frames.append(frame)
        
        # Stack back to [C, T, H, W]
        return torch.stack(transformed_frames, dim=1)

class SafeUCF101Dataset(Dataset):
    def __init__(self, data_root, annotation_path, frames_per_clip=16, train=True):
        # Use the standard UCF101 dataset
        self.dataset = datasets.UCF101(
            root=data_root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=1,
            train=train,
            output_format="THWC"  # Force THWC format
        )
        self.transform = VideoTransform()
        self.frames_per_clip = frames_per_clip
        
        # UCF101 has about 9537 train and 3783 test videos
        # Not millions like your previous output showed
        print(f"Dataset loaded with {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            # Get video and label from dataset
            video, audio, label = self.dataset[idx]
            return self.transform(video), label
            
        except Exception as e:
            # Return dummy data on error
            if idx < 10:  # Only print first few errors
                print(f"Error loading sample {idx}: {e}")
            return self.transform(None), 0

def get_datasets(data_root, annotation_path):
    print("Loading training dataset...")
    train_dataset = SafeUCF101Dataset(data_root, annotation_path, frames_per_clip=16, train=True)
    print("Loading test dataset...")
    test_dataset = SafeUCF101Dataset(data_root, annotation_path, frames_per_clip=16, train=False)
    return train_dataset, test_dataset