import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import json
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create torch dataset
class KeyPointsDataset(Dataset):
    def __init__(self, img_path, data_file):
        self.img_path = img_path
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(f"{self.img_path}/{item['id']}.png")
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        kps = np.array(item['kps']).flatten()
        kps = kps.astype(np.float32)
        kps[::2] *= 224 / w
        kps[1::2] *= 224 / h
        return img, kps
        

if __name__ == '__main__':
    train_dataset = KeyPointsDataset(img_path=r'traning\data\images', data_file=r'traning\data\data_train.json')
    val_dataset = KeyPointsDataset(img_path=r'traning\data\images', data_file=r'traning\data\data_val.json')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # create model
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 14*2) #replace the last layer
    model.to(device)

    # train model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    for epoch in range(num_epochs):
        for i, (img, kps) in enumerate(train_loader):
            img = img.to(device)
            kps = kps.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, kps)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    # save model
    torch.save(model.state_dict(), f"keypoints_model_.pth")
                