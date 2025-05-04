import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd

class WildfireDataset(Dataset):
    def __init__(self, labels_file, base_dir, rgb_transform=None, thermal_transform=None):
        self.labels_df = self._parse_labels_file(labels_file)
        self.base_dir = base_dir
        self.rgb_transform = rgb_transform
        self.thermal_transform = thermal_transform
        self.rgb_dir = os.path.join(base_dir, "254p_Frame_Pairs", "254p RGB Images")
        self.thermal_dir = os.path.join(base_dir, "254p_Frame_Pairs", "254p Thermal Images")
        self.frame_numbers = []
        for _, row in self.labels_df.iterrows():
            self.frame_numbers.extend(range(row['first_frame'], row['last_frame'] + 1))

    def _parse_labels_file(self, labels_file):
        df = pd.read_csv(labels_file, sep='\t', skiprows=3, header=None)
        df.columns = ['first_frame', 'last_frame', 'labels']
        df['fire'] = df['labels'].str[0].map({'Y': 1, 'N': 0})
        df['smoke'] = df['labels'].str[1].map({'Y': 1, 'N': 0})
        return df

    def __len__(self):
        return len(self.frame_numbers)

    def __getitem__(self, idx):
        frame_number = self.frame_numbers[idx]
        label_row = self.labels_df[
            (self.labels_df['first_frame'] <= frame_number) &
            (self.labels_df['last_frame'] >= frame_number)
        ].iloc[0]
        rgb_img_path = os.path.join(self.rgb_dir, f"254p RGB Frame ({frame_number}).jpg")
        thermal_img_path = os.path.join(self.thermal_dir, f"254p Thermal Frame ({frame_number}).jpg")
        rgb_image = Image.open(rgb_img_path).convert('RGB')
        thermal_image = Image.open(thermal_img_path).convert('L')
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.thermal_transform:
            thermal_image = self.thermal_transform(thermal_image)
        labels = torch.tensor([label_row['fire'], label_row['smoke']], dtype=torch.float32)
        return {'rgb': rgb_image, 'thermal': thermal_image, 'labels': labels, 'frame_number': frame_number}

def get_fire_dataloaders(labels_file, base_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    thermal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = WildfireDataset(labels_file, base_dir, rgb_transform, thermal_transform)
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader