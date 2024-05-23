import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class EEGDataLoader(Dataset):
    def __init__(self):
        self.populate_with_random_data()
        # Shape (n_epochs, n_channels, n_times)
        # print(eeg_data.shape)
        
        self.data_len  = 512
        self.n_channels, self.n_times = 128, 128
        # No need to wrap self.dataset again in TensorDataset
        # self.dataset = TensorDataset(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        eeg = self.dataset[index][0]
        return {'eeg': eeg}
    
    def populate_with_random_data(self, n_epochs=16, n_channels=128):
        random_data = torch.rand((n_epochs, n_channels, 128, 3), dtype=torch.float32)  # Correct the shape to 3, not 4
        self.dataset = TensorDataset(random_data)
        print(f"Random data shape: {random_data.shape}")

def create_dataset(batch_size=16):
    dataset = EEGDataLoader()
    return dataset

if __name__ == "__main__":
    batch_size = 16
    dataset = create_dataset()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sample = next(iter(data_loader))
    print(sample['eeg'].shape)  # Should print torch.Size([batch_size, 128, 128, 3])
