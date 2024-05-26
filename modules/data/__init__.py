from modules.data.augmentation import augment_all_data

def get_data_from_slice(data, loading_configs):
    """
    loading_configs should be a list of dictionaries e.g. [{'key': 'LMA_label', 'LMA_threshold'}]
    """
    loaded_data = {}
    for loading_config in loading_configs:
        key = loading_config['key']
        output_key = loading_config.get('output_key', key)
        if key == 'TOS':
            loaded_data[output_key] = data['TOSAnalysis']['TOSfullRes_Jerry']
        elif key == 'LMA_sector_labels':
            LMA_threshold = loading_config.get('LMA_threshold', 25)
            loaded_data[output_key] = (data['TOSAnalysis']['TOSfullRes_Jerry'] > LMA_threshold).astype(int)
        elif key == 'strain_matrix':
            loaded_data[output_key] = data['StrainInfo']['CCmid']
        else:
            loaded_data[output_key] = data[key]

        # select part of the data if needed
        if loading_config.get('use_only_original', False) and 'interp_frame_indicatior' in loading_config.keys():
            interp_frame_indicatior = data[loading_config['interp_frame_indicatior']]
            loaded_data[output_key] = loaded_data[output_key][..., np.where(interp_frame_indicatior==0)[0]]
    return loaded_data

def load_data(data_config, full_config=None):
    all_data = []
    
    LMA_threshold = data_config.get('LMA_threshold', 25)
    
    # Load npy file
    print('loading cine data from npy file...', end='')
    npy_filename = data_config['npy_filename']
    slices_data_list = np.load(npy_filename, allow_pickle=True).tolist()
    print('DONE!')
    print(f'{len(slices_data_list)} slices loaded')

    # Mark as original (unaugmented) data
    for datum in slices_data_list:
        datum['augmented'] = False
    
    # Keep only the first n_read slices if n_read is specified
    n_read = data_config.get('n_read', -1)
    n_read = len(slices_data_list) if n_read == -1 else n_read
    slices_data_list = slices_data_list[:n_read]

    # augmentation
    print('augmenting data...', end='')
    augmented_data = augment_all_data(slices_data_list, data_config)
    print('DONE!')
    print(type(augmented_data))
    print('len(augmentaed_data): ', len(augmented_data))
    slices_data_list += augmented_data
    print(f'# of data after augmentation: {len(slices_data_list)}')

    
    data_to_feed = data_config.get('data_to_feed', [{'key': 'LMA_label', 'LMA_threshold': 25}])
    loaded_data_list = []
    for slice_idx, datum in enumerate(slices_data_list):
        loaded_datum = {}
        for loading_config in data_to_feed:
            key = loading_config['key']
            output_key = loading_config.get('output_key', key)            
            loaded_datum[output_key] = datum[key]            
        
        loaded_datum['subject_id'] = datum['subject_id']
        loaded_datum['slice_idx'] = slice_idx
        loaded_datum['slice_full_id'] = f'{datum["subject_id"]}-{slice_idx}'
        
        loaded_data_list.append(loaded_datum)
    return all_data

import numpy as np
import torch
def check_dict(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            if value.size == 1:
                print('{:<60} {:<20}'.format(key, str(value)))
            else:
                print('{:<60} {:<20}'.format(key, str(value.shape)))
        elif isinstance(value, torch.Tensor):
            print('{:<60} {:<20}'.format(key, str(value.shape)))
        elif isinstance(value, dict):
            print('{:<60} {:<20}'.format(key, str(value.keys())))
        elif isinstance(value, list):
            print('{:<60} {:<20}'.format(key, 'list: (' + str(len(value))+')'))
        else:
            print('{:<60} {:<20}'.format(key, str(value)))


def split_vol_to_registration_pairs(vol, split_method:str='Lagrangian', output_dim=3):
    """
    Split the input volumes into pair of images for registraion

    vol: [batch_size, 2, n_frames, height, width]
    return: [batch_size*n_frames, 2, height, width]

    split_method: 'Lagrangian' or 'Eulerian'
    if split_method == 'Lagrangian', then the the pairs should be the first frame and each of the following frames.
    if split_method == 'Eulerian', then the pairs should be the every pair of adjacent frames.
    """

    batch_size, n_channels, n_frames, height, width = vol.shape
    assert n_frames > 1, f'n_frames should be larger than 1, but got {n_frames}'

    if split_method == 'Lagrangian':
        src = vol[:, :, :1, :, :].repeat(1, 1, n_frames-1, 1, 1)#.reshape(batch_size*(n_frames-1), n_channels, height, width)
        tar = vol[:, :, 1:, :, :]#.reshape(batch_size*(n_frames-1), n_channels, height, width)
    elif split_method == 'Eulerian':
        src = vol[:, :, :-1, :, :]#.reshape(batch_size*(n_frames-1), n_channels, height, width)
        tar = vol[:, :, 1:, :, :]#.reshape(batch_size*(n_frames-1), n_channels, height, width)
    else:
        raise ValueError(f'Unrecognized split_method: {split_method}')

    if output_dim == 2:
        src = src.reshape(batch_size*(n_frames-1), n_channels, height, width)
        tar = tar.reshape(batch_size*(n_frames-1), n_channels, height, width)

    return src, tar