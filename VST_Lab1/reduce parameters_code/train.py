from net import my_network
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_dir = r'./dataset/train'


labels_df = pd.read_csv('D:\\python\\23VST_HW1\\dataset\\train.csv')
labels_dict = dict(zip(labels_df['name'], labels_df['label']))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_dict, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(data_dir)
        self.labels_dict = labels_dict
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels_dict.get((self.image_list[idx]), None)  

        if label is None:
            print((self.image_list[idx]))
            print(f"Label not found for image: {img_name}")

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.RandomRotation(10),  
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataset = CustomDataset(data_dir=data_dir, labels_dict=labels_dict, transform=transform)



model = my_network()
model = model.to('cuda')

summary(model,(3,224,224))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []


for epoch in range(75):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss/len(train_loader))
    train_accuracies.append(100 * correct / total)

    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(val_loss/len(val_loader))
    val_accuracies.append(100 * correct / total)

    print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {100 * correct / total}")

    val_loss = val_loss/len(val_loader)
    scheduler.step(val_loss)



torch.save(model.state_dict(), 'w_312605015.pth')


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

