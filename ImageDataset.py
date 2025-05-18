from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_files: list, path: str, transform=None):
        self.path = path
        self.image_files = image_files
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.path, image_name)
        image = Image.open(image_path).convert("RGB")
        label = 0 if image_name.startswith("cat") else 1
        if self.transform:
            image = self.transform(image)
        return image, label