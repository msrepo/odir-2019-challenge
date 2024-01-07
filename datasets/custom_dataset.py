from torch.utils.data import Dataset
from os.path import join

class CSVDataset(Dataset):
    def __init__(self, data_root_dir:str,csv_path:str,img_transform=None, label_transform=None) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.csv_path = csv_path
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_paths, self.labels = self.read_csv()
        
    
    def read_csv(self):
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        image_paths = df['fundus'].to_list()
        # allow empty labels for test data sets
        if 'label' in df.columns:
            label_paths = df['label'].to_list()
        else:
            label_paths = None
        return image_paths, label_paths

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img =  join(self.data_root_dir,self.image_paths[index])
        if self.labels:
            label = self.labels[index]
        else:
            label = None

        if self.img_transform:
            img = self.img_transform(img)
            
        if self.label_transform and label:
            label = self.label_transform(label)
        
        return img, label

