import torch
from torch.utils.data import DataLoader
from dataset import get_datasets
from model import get_model

# Paths
data_root = r"C:\ASH_PROJECT\data\UCF101"
annotation_path = r"C:\ASH_PROJECT\data\ucfTrainTestlist"

# Dataset
_, test_dataset = get_datasets(data_root, annotation_path)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=101).to(device)
model.load_state_dict(torch.load(r"C:\ASH_PROJECT\outputs\checkpoints\r3d18_ucf101.pth"))
model.eval()

# Evaluation
correct, total = 0, 0
with torch.no_grad():
    for clips, labels in test_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
