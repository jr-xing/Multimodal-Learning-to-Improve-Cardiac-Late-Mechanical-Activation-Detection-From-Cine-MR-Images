import torch
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def get_n_slices(self):
#         # Assuming your data is organized in such a way
#         # that this method returns the number of slices/data points
#         return len(self.data)

#     def get_slice(self, idx):
#         # Assuming this method returns the data slice at index idx
#         return self.data[idx]

def custom_collate_fn(batch):
    # Flatten the list of lists of dictionaries into a list of dictionaries
    flattened_batch = [item for sublist in batch for item in sublist]
    
    # Now we'll organize the data by keys, similar to before
    merged_batch = {key: [] for key in flattened_batch[0].keys()}
    for dictionary in flattened_batch:
        for key, value in dictionary.items():
            merged_batch[key].append(value)
    
    # Convert each list of values into a torch tensor
    # for key in merged_batch:
    #     merged_batch[key] = torch.stack(merged_batch[key])
    
    for key in merged_batch:
        if all(isinstance(item, torch.Tensor) for item in merged_batch[key]):
            merged_batch[key] = torch.stack(merged_batch[key])
        elif all(isinstance(item, (int, float)) for item in merged_batch[key]):
            # Optionally convert integers to tensor or leave as list
            # merged_batch[key] = torch.tensor(merged_batch[key], dtype=torch.long)
            merged_batch[key] = torch.tensor(merged_batch[key])
            # pass  # Leave as a list of integers
        elif all(isinstance(item, str) for item in merged_batch[key]):
            pass  # Leave as a list of strings
        else:
            raise ValueError(f"Unsupported data type for key: {key} with type {type(merged_batch[key][0])}")

    return merged_batch

class SliceDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, *args, **kwargs):
        super().__init__(dataset, batch_size, shuffle, collate_fn=custom_collate_fn, *args, **kwargs)
        self.current_idx = 0  # Initialize the current index to 0

    def __len__(self):
        # return self.dataset.get_n_slices()
        total_slices = self.dataset.get_n_slices()
        # Calculate the number of batches
        num_batches = (total_slices + self.batch_size - 1) // self.batch_size
        return num_batches
    # def __init__(self, dataset, batch_size=1, shuffle=False, *args, **kwargs):
    #     super().__init__(dataset, batch_size, shuffle, *args, **kwargs)

    # def __len__(self):
    #     # Override to use get_n_slices method of dataset
    #     return self.dataset.get_n_slices()

    # def __getitem__(self, idx):
    #     # Override to use get_slice method of dataset
    #     return self.dataset.get_slice(idx)
    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield self.collate_fn([self.dataset.get_slice(i)])

    def __iter__(self):
        # Determine the number of batches
        num_batches = len(self)
        for i in range(num_batches):
            # Get the starting and ending indices for the next batch
            start_idx = i * self.batch_size
            # end_idx = min((i + 1) * self.batch_size, len(self.dataset))
            end_idx = min((i + 1) * self.batch_size, self.dataset.get_n_slices())

            # print(f"start_idx: {start_idx}, end_idx: {end_idx}")

            # Collect all data slices for the next batch
            batch_data = []
            for j in range(start_idx, end_idx):
                # batch_data.extend(self.dataset.get_slice(j))  # Use extend since get_slice returns a list
                batch_data.append(self.dataset.get_slice(j))  # Use extend since get_slice returns a list

            # Collate the batch data
            yield self.collate_fn(batch_data)
    
    # def __iter__(self):
    #     self.current_idx = 0  # Reset the index at the start of each epoch
    #     while self.current_idx < len(self):
    #         batch_idx = self.current_idx  # Store the current index
    #         self.current_idx += 1  # Increment the current index for the next call
    #         yield self.collate_fn([self.dataset.get_slice(batch_idx)])

# # Assume data is your dataset
# data = CustomDataset(your_data)
# loader = CustomDataLoader(data, batch_size=32, shuffle=True)

# for batch in loader:
#     # your processing code
