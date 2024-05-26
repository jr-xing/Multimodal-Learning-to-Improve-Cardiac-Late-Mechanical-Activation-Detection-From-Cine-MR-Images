import torch
from random import shuffle
class SliceDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.batch_sampler = self._create_batch_sampler()

    def __iter__(self):
        return self._iterator()
    
    def __len__(self):
        # Calculate the total number of batches
        num_samples = len(self.dataset)
        num_batches = num_samples // self.batch_size
        if num_samples % self.batch_size != 0:
            num_batches += 1
        return num_batches

    def _create_batch_sampler(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            torch.manual_seed(0)  # For reproducibility, you can remove this line
            # torch.random.shuffle(indices)
            shuffle(indices)
        batch_sampler = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_sampler.append(batch_indices)
        return batch_sampler

    def _iterator(self):
        for batch_indices in self.batch_sampler:
            batch_data = [self.dataset[i] for i in batch_indices]
            yield self._collate_batch(batch_data)

    # def _collate_batch(self, batch_data):
    #     # You can implement your custom collate logic here
    #     # For simplicity, we assume the dataset returns a list of samples
    #     # You may need to modify this part based on your dataset's structure        
    #     return batch_data
    def _collate_batch(self, batch_data):
        # Assuming that each sample in the dataset is a tuple of (data, label)
        data_batch, label_batch = zip(*batch_data)
        
        # Convert NumPy arrays to PyTorch tensors and stack them
        # stacked_data = torch.tensor(np.stack(data_batch), dtype=torch.float32)
        stacked_data = torch.stack([torch.tensor(data, dtype=torch.float32) for data in data_batch], dim=0)

        
        # Convert labels to a tensor
        stacked_labels = torch.tensor(label_batch, dtype=torch.int64)
        
        return stacked_data, stacked_labels
# DataLoader
# train_dataloader = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# test_dataloader = CustomDataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)