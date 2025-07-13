# scripts/detect.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from preprocess import load_data

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def predict(image_path, model_path='../models/saved_model.pth', data_dir='../data/train'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, classes = load_data(data_dir, batch_size=1)

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    print(f"Predicted class: {classes[predicted.item()]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python detect.py path_to_image")
    else:
        predict(sys.argv[1])
