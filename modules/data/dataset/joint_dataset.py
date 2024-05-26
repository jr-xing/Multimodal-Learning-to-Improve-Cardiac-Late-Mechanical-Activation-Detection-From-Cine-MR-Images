from typing import Any
from torch.utils.data import Dataset as TorchDataset
import pathlib
# from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from modules.data.datareader.DENSE_IO_utils import align_n_frames_to
class JointDataset(TorchDataset):
    def __init__(self, data, augmentation=None, dataset_config = {}, full_config={}, dataset_name=None):
        super().__init__()
        self.data = data
        self.dataset_config = dataset_config
        self.full_config = full_config
        print(self.full_config)
        self.dataset_config = dataset_config
        # self.transform = ToTensorV2()
        self.n_subjects = len(set([datum['subject_id'] for datum in self.data]))
        self.n_slices = len(set([datum['slice_full_id'] for datum in self.data]))
        self.slice_full_ids = self.get_slice_full_ids()
        self.n_myo_frames_to_use_for_regression = self.dataset_config.get('n_myo_frames_to_use_for_regression', 20)
        self.n_strainmat_frames_to_use_for_regression = self.dataset_config.get('n_strainmat_frames_to_use_for_regression', 40)
        
        self.cine_myo_mask_key = self.dataset_config.get('cine_myo_mask_key', 'cine_lv_myo_masks')
        self.strain_mat_key = self.dataset_config.get('strain_mat_key', 'strain_matrix')
        self.TOS_key = self.dataset_config.get('TOS_key', 'TOS')
        # orginially 25
        # 48 for merged data and 32 for original DENSE data

        # align the displacement field frames
        self.align_cine_myo_mask_frames()
        self.align_strain_mat_frames()

    def align_cine_myo_mask_frames(self):
        n_target_frames = self.n_myo_frames_to_use_for_regression
        frame_idx = -1
        padding_method = 'edge'
        # print(self.data[0].keys())
        print(f'Aligning displacement field frames to {n_target_frames} frames')
        
        print(f'# of frames before alignment: {list(set([d[self.cine_myo_mask_key].shape[-1] for d in self.data]))}')
        for datum_idx, datum in enumerate(self.data):
            self.data[datum_idx][self.cine_myo_mask_key] = align_n_frames_to(datum[self.cine_myo_mask_key], n_target_frames, frame_idx, padding_method)
        print(f'# of frames after alignment: {list(set([d[self.cine_myo_mask_key].shape[-1] for d in self.data]))}')
    
    def align_strain_mat_frames(self):
        n_target_frames = self.n_strainmat_frames_to_use_for_regression
        frame_idx = -1
        padding_method = 'edge'
        # print(self.data[0].keys())
        print(f'Aligning strain matrix frames to {n_target_frames} frames')

        print(f'# of frames before alignment: {list(set([d[self.strain_mat_key].shape[-1] for d in self.data]))}')
        for datum_idx, datum in enumerate(self.data):
            self.data[datum_idx]['strain_matrix'] = align_n_frames_to(datum['strain_matrix'], n_target_frames, frame_idx, padding_method)
        print(f'# of frames after alignment: {list(set([d[self.strain_mat_key].shape[-1] for d in self.data]))}')
    
    def __len__(self):
        # return self.N
        return len(self.data)
    
    def __getitem__(self, index):
        # datum = self.data[index].feed_to_network()
        raw_datum = self.data[index]
        datum = {}
        # datum = {
        #     'source_img': raw_datum['source_image'],
        #     'target_img': raw_datum['target_image'],            
        # }
        # datum['source_img'] = torch.from_numpy(datum['source_img'][None, :,:]).to(torch.float32)
        # datum['target_img'] = torch.from_numpy(datum['target_img'][None, :,:]).to(torch.float32)
        cine_myo_mask = torch.from_numpy(raw_datum[self.cine_myo_mask_key][None, ...]).moveaxis(-1, 1)
        strain_matrix = torch.from_numpy(raw_datum[self.strain_mat_key][None, ...])
        TOS = torch.from_numpy(raw_datum[self.TOS_key])
        datum['cine_myo_mask'] = cine_myo_mask.to(torch.float32)
        datum['strain_matrix'] = strain_matrix.to(torch.float32)
        datum['TOS'] = TOS.to(torch.float32)        

        # copy all non-numpy and non-torch objects to datum_augmented
        for k, v in raw_datum.items():
            if isinstance(v, pathlib.PosixPath):
                datum[k] = str(v)
            elif not isinstance(v, (torch.Tensor, np.ndarray, int)):
                datum[k] = v
            elif isinstance(v, int):
                datum[k] = torch.Tensor([v]).to(torch.long)
            
        # datum_augmented = self.transform(**datum)
        # datum_augmented['idx'] = index
        return datum
    

    def get_subject_ids(self):
        return list(set([datum['subject_id'] for datum in self.data]))
    
    def get_slice_full_ids(self):
        return list(set([datum['slice_full_id'] for datum in self.data]))
    
    def get_n_slices(self):
        return len(self.slice_full_ids)
    
    def get_slice(self, slice_idx):
        # data_of_slices = [datum for datum in self.data if datum['slice_full_id'] == self.slice_full_ids[slice_idx]]
        data_indices_of_slices = [idx for idx, datum in enumerate(self.data) if datum['slice_full_id'] == self.slice_full_ids[slice_idx]]
        data_of_slices = [self.__getitem__(idx) for idx in data_indices_of_slices]
        return data_of_slices
    