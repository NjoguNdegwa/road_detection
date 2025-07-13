# scripts/preprocess.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.classes

if __name__ == "__main__":
    train_loader, classes = load_data('../data/train')
    print(f"Classes found: {classes}")
