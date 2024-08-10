from torch.utils.data import Dataset

from hest.wsi import WSIPatcher


class WSIPatcherDataset(Dataset):
    
    def __init__(self, patcher: WSIPatcher, transform):
        self.patcher = patcher
        
        self.transform = transform
                              

    def __len__(self):
        return len(self.patcher)
    
    def __getitem__(self, index):
        tile, x, y = self.patcher[index]
        
        if self.transform:
            tile = self.transform(tile)

        return tile, (x, y)