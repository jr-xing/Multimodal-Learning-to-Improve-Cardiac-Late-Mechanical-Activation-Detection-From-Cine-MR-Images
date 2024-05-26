import numpy as np
from pathlib import Path
from modules.data.datareader.BaseDatum import BaseDatum
from modules.data.datareader.BaseDataReader import BaseDataReader


class DENSECINEDatum(BaseDatum):
    def load_datum(self):
        pass
    
    @staticmethod
    def load_data(data_list):
        pass

class DENSECINEDataReader(BaseDataReader):
    # def load_record_from_dir(self, config):
    #     return super().load_record_from_dir(config)

    def load_record_from_npy(self, data_config):
        npy_filename = data_config['loading']['npy_filename']
        # raw_pairs = load_pair_list_from_npy_file(npy_filename, data_config)
        raw_pairs = load_cine_pairs_from_npy_file(npy_filename, data_config)
        
        all_data = []
        for raw_datum_idx, raw_datum in enumerate(raw_pairs):
            datum_dict = raw_datum
            if 'patient_id' in raw_datum.keys():
                datum_dict['subject_id'] = raw_datum['patient_id']
            # datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["source_DENSE_time_idx"]}_{raw_datum["target_DENSE_time_idx"]}'
            datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["source_time_idx"]}_{raw_datum["target_time_idx"]}'
            # {
            #     'source_img': raw_datum['source_cine_mask'],
            #     'target_img': raw_datum['target_cine_mask'],
            #     'subject_id': raw_datum['patient_id'],
            # }
            datum = DENSECINEDatum(data_dict=datum_dict)
            all_data.append(datum)

        resize = data_config['loading'].get('resize', False)
        if resize:
            print('resizing DENSE-cine images...', end='')
            from skimage.transform import resize
            for datum in all_data:
                datum['image'] = resize(datum['image'], [128, 128])
            print('DONE!')

        return all_data
    

# def load_pairs_list_from_npy_file(npy_filename, data_config=None):
#     slices_data_list = np.load(npy_filename, allow_pickle=True)
#     n_read = data_config.get('n_read', -1)

#     # convert to list of pairs
#     # When you have the cine_lv_myo_masks_interpolated data, you can use this code to convert the list of slices into list of registration pairs
#     # cine_LV_masks_key = 'cine_lv_myo_masks_interpolated'
#     # cine_LV_masks_key = 'cine_lv_myo_masks'
#     cine_LV_masks_key = 'interpolation'
#     registraion_pair_list = []
#     for slice_idx, slice_data in enumerate(slices_data_list[:n_read]):
#         if cine_LV_masks_key not in slice_data.keys():
#             print('Warning: cine_lv_myo_masks_interpolated not found in slice_data')
#             continue
#         slice_cine_LV_masks = slice_data[cine_LV_masks_key] # should have shape (n_DENSE_frames, H, W)
#         n_DENSE_frames = slice_cine_LV_masks.shape[-1]

#         for DENSE_frame_idx in range(n_DENSE_frames):
#             if DENSE_frame_idx == n_DENSE_frames - 1:
#                 source_DENSE_time_idx = DENSE_frame_idx
#                 target_DENSE_time_idx = 0
#             else:                
#                 source_DENSE_time_idx = DENSE_frame_idx
#                 target_DENSE_time_idx = DENSE_frame_idx + 1
            
#             source_image = slice_cine_LV_masks[source_DENSE_time_idx,  ...].astype(np.float32)
#             target_image = slice_cine_LV_masks[target_DENSE_time_idx,  ...].astype(np.float32)

#             if np.sum(source_image) < 10 or np.sum(target_image) < 10:
#                 # print('Warning: source_image or target_image is (almost) empty')
#                 continue
            
#             # initialize pair_data_dict
#             pair_data_dict = {}

#             # copy the data from slice_data that are not numpy arrays
#             for key, value in slice_data.items():
#                 if not isinstance(value, np.ndarray):
#                     pair_data_dict[key] = value
            
#             # Set the other data
#             pair_data_dict['source_cine_mask'] = source_image
#             pair_data_dict['target_cine_mask'] = target_image
#             pair_data_dict['source_DENSE_time_idx'] = source_DENSE_time_idx
#             pair_data_dict['target_DENSE_time_idx'] = target_DENSE_time_idx
#             pair_data_dict['GT_DENSE_displacement_field_X'] = slice_data['DENSE_displacement_field_X'][:, :, DENSE_frame_idx]
#             pair_data_dict['GT_DENSE_displacement_field_Y'] = slice_data['DENSE_displacement_field_Y'][:, :, DENSE_frame_idx]

#             registraion_pair_list.append(pair_data_dict)
#     return registraion_pair_list

def load_cine_pairs_from_npy_file(npy_filename, data_config=None):
    slices_data_list = np.load(npy_filename, allow_pickle=True)
    n_read = data_config.get('n_read', -1)    
    use_interpolated_data = data_config['loading'].get('use_interpolated_data', False)
    interpolated_cine_key = data_config['loading'].get('interpolated_cine_key', 'cine_lv_myo_masks_interpolated')
    interpolated_DENSE_key = data_config['loading'].get('interpolated_DENSE_key', 'DENSE_displacement_field_interpolated')
    strain_matrix_n_frames = 50
    pair_data_list = []
    for slice_idx, slice_data in enumerate(slices_data_list):
        subject_id = slice_data['patient_id']
        subject_LV_masks = slice_data[interpolated_cine_key]
        H, W, n_frames = subject_LV_masks.shape
        # append DENSE data if use interpolated data
        if use_interpolated_data:
            subject_DENSE_displacement_field_X = slice_data[interpolated_DENSE_key+'_X']
            subject_DENSE_displacement_field_Y = slice_data[interpolated_DENSE_key+'_Y']
            # check if the shape of DENSE data matches the shape of cine data
            if subject_DENSE_displacement_field_X.shape != subject_LV_masks.shape:
                print(f'Warning: shape of DENSE data {subject_DENSE_displacement_field_X.shape} does not match the shape of cine data {subject_LV_masks.shape}')
                continue

        # load TOS and strain matrix
        if 'TOSAnalysis' not in slice_data.keys():
            print('Warning: TOSAnalysis not found in slice_data')
            continue
        slice_TOS = slice_data['TOSAnalysis']['TOSfullRes_Jerry']
        slice_strain_matrix = slice_data['StrainInfo']['CCmid']
        # slice_TOS = slice_data['TOSAnalysis']['TOS18_Jerry']
        # slice_strain_matrix = slice_data[0]['TransmuralStrainInfo']['Ecc']['mid'].T
        
        for frame_idx in range(n_frames):
            if frame_idx == n_frames - 1:
                source_time_idx = frame_idx
                target_time_idx = 0
            else:                
                source_time_idx = frame_idx
                target_time_idx = frame_idx + 1
            
            source_image = subject_LV_masks[:, :, source_time_idx].astype(np.float32)
            target_image = subject_LV_masks[:, :, target_time_idx].astype(np.float32)

            if np.sum(source_image) == 0 or np.sum(target_image) == 0:
                continue

            pair_data_dict = {
                'subject_id': subject_id,
                'slice_idx': slice_idx,
                'slice_full_id': f'{subject_id}-{slice_idx}',
                'source_time_idx': source_time_idx,
                'target_time_idx': target_time_idx,
                'source_image': source_image,
                'target_image': target_image,
            }

            if use_interpolated_data:
                pair_data_dict['DENSE_displacement_field_X'] = subject_DENSE_displacement_field_X[:, :, frame_idx]
                pair_data_dict['DENSE_displacement_field_Y'] = subject_DENSE_displacement_field_Y[:, :, frame_idx]

            # replace the NaN in displacement field with 0
            pair_data_dict['DENSE_displacement_field_X'][np.isnan(pair_data_dict['DENSE_displacement_field_X'])] = 0
            pair_data_dict['DENSE_displacement_field_Y'][np.isnan(pair_data_dict['DENSE_displacement_field_Y'])] = 0

            # append TOS and strain matrix
            LMA_threshold = 25
            pair_data_dict['TOS'] = slice_TOS
            pair_data_dict['sector_LMA_labels'] = (pair_data_dict['TOS'] > LMA_threshold).astype(int)
            pair_data_dict['slice_LMA_label'] = int(pair_data_dict['TOS'].max() > LMA_threshold)
            # pair_data_dict['strain_matrix'] = slice_strain_matrix
            # align the dim 2 of strain matrices by zero padding or cropping
            if slice_strain_matrix.shape[1] > strain_matrix_n_frames:
                slice_strain_matrix_aligned = slice_strain_matrix[:, :strain_matrix_n_frames]
            elif slice_strain_matrix.shape[1] < strain_matrix_n_frames:
                slice_strain_matrix_aligned = np.zeros((slice_strain_matrix.shape[0], strain_matrix_n_frames))
                slice_strain_matrix_aligned[:, :slice_strain_matrix.shape[1]] = slice_strain_matrix
            else:
                slice_strain_matrix_aligned = slice_strain_matrix
            pair_data_dict['strain_matrix'] = slice_strain_matrix_aligned

            pair_data_list.append(pair_data_dict)
    return pair_data_list