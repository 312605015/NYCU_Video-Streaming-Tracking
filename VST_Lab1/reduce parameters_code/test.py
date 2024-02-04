from net import my_network
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import pandas as pd


def test():
    # Load model
    model = my_network()
    model.load_state_dict(torch.load("w_312605015.pth"))
    model.eval()
    model.to('cuda')

   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    
    test_data_dir = r'./dataset/test'

    test_image_list = os.listdir(test_data_dir)

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, image_list, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.image_list = image_list

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.image_list[idx])
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image

    test_dataset = TestDataset(data_dir=test_data_dir, image_list=test_image_list, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

 
    predictions = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())


    df = pd.DataFrame({'name': test_image_list, 'label': predictions})
    df['name'] = df['name'].apply(lambda x: int(x.split('.')[0]))  
    df = df.sort_values(by='name')  
    df['name'] = df['name'].apply(lambda x: f"{x}.jpg")  
    df.to_csv("pred_312605015.csv", index=False)


if __name__=="__main__":
    test()