from unittest import TestCase
import unittest
from datasets.custom_dataset import CSVDataset
from transforms.transforms import get_img_transform

class TestDataset(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.csv_path = 'csv/processed_val_ODIR-5K.csv'
        self.dataset = CSVDataset(data_root_dir='odir2019/ODIR-5K_Training_Dataset',
                                  csv_path=self.csv_path,img_transform=get_img_transform(img_size=300))
    
    def test_dataset(self):
        img, label = self.dataset[0]
        print(img.shape,label)

if __name__ == "__main__":
    unittest.main()