from dataset import get_datasets
from model import get_model
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def main():
    # Paths
    data_root = r"C:\ASH_PROJECT\data\UCF101"
    annotation_path = r"C:\ASH_PROJECT\data\ucfTrainTestlist"

    # Device (force GPU)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ Using device: {gpu_name}")

    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset, test_dataset = get_datasets(data_root, annotation_path)

    print(f"📊 Train dataset size: {len(train_dataset)} clips")
    print(f"📊 Test dataset size: {len(test_dataset)} clips")

    # Use num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    print(f"🌀 Train loader batches: {len(train_loader)}")
    print(f"🌀 Test loader batches: {len(test_loader)}")

    # Model
    model = get_model(num_classes=101).to(device)
    print(f"✅ Model loaded on: {next(model.parameters()).device}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 1  # Start with 1 epoch to test
    for epoch in range(1, num_epochs + 1):
        print(f"\n🚀 Starting Epoch {epoch}/{num_epochs}...")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm for progress tracking
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (clips, labels) in enumerate(pbar):
            # Move to GPU
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'Acc': f"{100.*correct/total:.2f}%"
                })

        avg_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"✅ Epoch {epoch}/{num_epochs} completed | Average Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), r"C:\ASH_PROJECT\outputs\checkpoints\r3d18_ucf101.pth")
    print("\n💾 Model training completed and saved!")

if __name__ == "__main__":
    main()