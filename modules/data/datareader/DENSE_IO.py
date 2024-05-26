import numpy as np
from pathlib import Path
import copy
from modules.data.datareader.BaseDatum import BaseDatum
from modules.data.datareader.BaseDataReader import BaseDataReader
from modules.data.augmentation.affine import rotate, translate
from skimage.morphology import dilation
class DENSEDatum(BaseDatum):
    def load_datum(self):
        pass
    
    @staticmethod
    def load_data(data_list):
        pass

class DENSEDataReader(BaseDataReader):
    # def load_record_from_dir(self, config):
    #     return super().load_record_from_dir(config)

    def load_record_from_npy(self, data_config):
        npy_filename = data_config['loading']['npy_filename']
        loading_method = data_config['loading'].get('method', 'cine_registration_pairs')
        # raw_pairs = load_pair_list_from_npy_file(npy_filename, data_config)
        if loading_method == 'cine_registration_pairs':
            raw_data = load_cine_pairs_from_npy_file(npy_filename, data_config)
        elif loading_method == 'DENSE_slices':
            raw_data = load_DENSE_slices_from_npy_file(npy_filename, data_config)
        elif loading_method == 'general_slice':
            raw_data = load_slices_from_npy_file(npy_filename, data_config)
        else:
            raise NotImplementedError(f'loading_method {loading_method} not implemented')
        
        all_data = []
        for raw_datum_idx, raw_datum in enumerate(raw_data):
            datum_dict = raw_datum
            if 'patient_id' in raw_datum.keys():
                datum_dict['subject_id'] = raw_datum['patient_id']
            # datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["source_DENSE_time_idx"]}_{raw_datum["target_DENSE_time_idx"]}'
            if loading_method == 'cine_registration_pairs':
                # datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["slice_idx"]}_{raw_datum["source_time_idx"]}_{raw_datum["target_time_idx"]}'
                datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["source_time_idx"]}_{raw_datum["target_time_idx"]}'
            elif loading_method in ['DENSE_slices', 'general_slice']:
                datum_dict['full_name'] = raw_datum['subject_id'] + f'_{raw_datum["slice_idx"]}'
            # {
            #     'source_img': raw_datum['source_cine_mask'],
            #     'target_img': raw_datum['target_cine_mask'],
            #     'subject_id': raw_datum['patient_id'],
            # }
            datum = DENSEDatum(data_dict=datum_dict)
            all_data.append(datum)

        resize = data_config['loading'].get('resize', False)
        if resize:
            print('resizing DENSE-cine images...', end='')
            from skimage.transform import resize
            for datum in all_data:
                datum['image'] = resize(datum['image'], [128, 128])
            print('DONE!')

        return all_data

def augment_datum(datum: dict, config=None):
    if config is None:
        config = {
            'translate': {
                'y': 0,
                'x': 0,
            },
            'rotate': {
                'n_rotate_sectors': 0,
            },
        }
    translate_y = config['translate']['y']
    translate_x = config['translate']['x']
    n_rotate_sectors = config['rotate']['n_rotate_sectors']    
    
    datum_aug = rotate(datum, n_rotate_sectors)
    datum_aug = translate(datum_aug, translate_y, translate_x)
    datum_aug['augmented'] = True
    
    return datum_aug

def augment_all_data(data_list, data_config):
    augment_translate_times_y = data_config['loading'].get('augment_translate_times_y', 0)
    augment_translate_times_x = data_config['loading'].get('augment_translate_times_x', 0)
    augment_rotate_times = data_config['loading'].get('augment_rotate_times', 0)
    if augment_translate_times_y == 0:
        augment_translate_ys = [0]
    elif augment_translate_times_y == 1:
        augment_translate_ys = [5]
    else:
        if augment_translate_times_y % 2 == 0:
            augment_translate_ys_positive = np.linspace(0, 10, augment_translate_times_y//2+2).astype(int)[1:-1]
            augment_translate_ys_negative = -augment_translate_ys_positive
        else:
            augment_translate_ys_positive = np.linspace(0, 10, np.ceil(augment_translate_times_y/2)+2).astype(int)[1:-1]
            augment_translate_ys_negative = -augment_translate_ys_positive[:-1]
        augment_translate_ys = np.concatenate([augment_translate_ys_positive, augment_translate_ys_negative])
    
    if augment_translate_times_x == 0:
        augment_translate_xs = [0]
    elif augment_translate_times_x == 1:
        augment_translate_xs = [5]
    else:
        if augment_translate_times_x % 2 == 0:
            augment_translate_xs_positive = np.linspace(0, 10, augment_translate_times_x//2+2).astype(int)[1:-1]
            augment_translate_xs_negative = -augment_translate_xs_positive
        else:
            augment_translate_xs_positive = np.linspace(0, 10, np.ceil(augment_translate_times_x/2).astype(int)+2).astype(int)[1:-1]
            augment_translate_xs_negative = -augment_translate_xs_positive[:-1]
        augment_translate_xs = np.concatenate([augment_translate_xs_positive, augment_translate_xs_negative])
    augment_rotate_interval = data_config['loading'].get('augment_rotate_interval', 10)
    if augment_rotate_interval == -1:
        augment_rotate_n_sectors = np.linspace(1, 126, augment_rotate_times+2).astype(int)[1:-1]
    else:
        augment_rotate_n_sectors = (np.arange(1,20)*augment_rotate_interval)[:augment_rotate_times]

    augmented_data_list = []
    default_augment_config = {
        'translate': {
            'y': 0,
            'x': 0,
        },
        'rotate': {
            'n_rotate_sectors': 0,
        },
    }
    print(f'Augmenting data: translate_ys={augment_translate_ys}, translate_xs={augment_translate_xs}, rotate_n_sectors={augment_rotate_n_sectors}')
    data_keys_to_check = ['StrainInfo', 'TOSAnalysis']
    for datum in data_list:
        # check whether the data is valid
        key_missing = False
        for key in data_keys_to_check:
            if key not in datum.keys():
                print(f'Warning: key {key} not found in datum of patient {datum["patient_id"]}')
                key_missing = True
                continue
        if key_missing:
            continue


        for augment_translate_y in augment_translate_ys:
            for augment_translate_x in augment_translate_xs:
                for augment_rotate_n_sector in augment_rotate_n_sectors:
                    augment_config = default_augment_config
                    augment_config['translate']['y'] = augment_translate_y
                    augment_config['translate']['x'] = augment_translate_x
                    augment_config['rotate']['n_rotate_sectors'] = augment_rotate_n_sector
                    augment_config['rotate']['augment_rotate_angle'] = - augment_rotate_n_sector * 360 / 126
                    augmented_datum = copy.deepcopy(datum)
                    augmented_datum.update(augment_datum(augmented_datum, augment_config))
                    augmented_datum['augmented'] = True
                    # augmented_datum['augment_translate_y'] = augment_translate_y
                    # augmented_datum['augment_translate_x'] = augment_translate_x
                    # augmented_datum['augment_rotate_n_sector'] = augment_rotate_n_sector
                    # augmented_datum['augment_rotate_angle'] = - augment_rotate_n_sector * 360 / 126
                    augmented_data_list.append(augmented_datum)
    
    print(f'Augmented data from {len(data_list)} to {len(augmented_data_list)}')
    return augmented_data_list

def load_DENSE_slices_from_npy_file(npy_filename, data_config=None):
    LMA_threshold = data_config.get('LMA_threshold', 25)
    print('loading DENSE slices from npy file...', end='')
    raw_slices_data_list = np.load(npy_filename, allow_pickle=True).tolist()
    print('DONE!')
    print(f'{len(raw_slices_data_list)} slices loaded')

    # filter slices if needed
    filter_npy_file = data_config['loading'].get('filter_npy_file', False)
    if filter_npy_file:
        print('filtering slices...')
        print('len(raw_slices_data_list) before filtering:', len(raw_slices_data_list))
        filter_npy_file_based_filename = data_config['loading']['filter_npy_file_based_filename']
        filter_npy_file_based_data = np.load(filter_npy_file_based_filename, allow_pickle=True)

        # append data for filtering
        for filter_npy_file_based_datum in filter_npy_file_based_data:
            filter_npy_file_based_datum['patient_id_cine_slice_idx'] = filter_npy_file_based_datum['patient_id'] + '_' + str(filter_npy_file_based_datum['cine_slice_idx'])

        for datum in raw_slices_data_list:
            datum['patient_id_cine_slice_idx'] = datum['patient_id'] + '_' + str(datum['cine_slice_idx'])

        # filter
        filtered_data = []
        for i in range(len(raw_slices_data_list)):
            raw_slice_datum = raw_slices_data_list[i]
            # find the corresponding datum in filter_npy_file_based_data
            filter_npy_file_based_datum = [d for d in filter_npy_file_based_data if d['patient_id_cine_slice_idx'] == raw_slice_datum['patient_id_cine_slice_idx']]
            if len(filter_npy_file_based_datum) == 0:
                print(f'Warning: cannot find corresponding datum in filter_npy_file_based_data for patient {raw_slice_datum["patient_id"]} cine_slice_idx {raw_slice_datum["cine_slice_idx"]}')
                continue
            elif len(filter_npy_file_based_datum) > 1:
                print(f'Warning: more than 1 corresponding datum in filter_npy_file_based_data for patient {raw_slice_datum["patient_id"]} cine_slice_idx {raw_slice_datum["cine_slice_idx"]}, using the first')
                filter_npy_file_based_datum = filter_npy_file_based_datum[0]
            #     continue
            else:
                filter_npy_file_based_datum = filter_npy_file_based_datum[0]
                # copy keys from filter_npy_file_based_datum to raw_slice_datum if key not exist
                for key in filter_npy_file_based_datum.keys():
                    if key not in raw_slice_datum.keys():
                        raw_slice_datum[key] = filter_npy_file_based_datum[key]
                # raw_slice_datum['TOSAnalysis'] = filter_npy_file_based_datum['TOSAnalysis']
                # raw_slice_datum['StrainInfo'] = filter_npy_file_based_datum['StrainInfo']
                filtered_data.append(raw_slice_datum)
        
        raw_slices_data_list = filtered_data
        print('len(raw_slices_data_list) after filtering:', len(raw_slices_data_list))
        print(raw_slices_data_list[0].keys())

    # append additional data if needed
    append_additional_data = data_config['loading'].get('append_additional_data', False)
    if append_additional_data:
        from modules.data.datareader.DENSE_IO_utils import append_additional_data_from_npy
        raw_slices_data_list = append_additional_data_from_npy(
            raw_slices_data_list, 
            npy_filename=data_config['loading']['additional_data_npy_filename'],
            config=data_config,
            file_source='from_Nellie')

    n_read = data_config.get('n_read', -1)
    n_read = len(raw_slices_data_list) if n_read == -1 else n_read
    raw_slices_data_list = raw_slices_data_list[:n_read]

    for datum in raw_slices_data_list:
        datum['augmented'] = False
    
    use_interpolated_data = data_config['loading'].get('use_interpolated_data', False)
    interpolated_cine_key = data_config['loading'].get('interpolated_cine_key', 'cine_lv_myo_masks_merged')
    interpolated_DENSE_key = data_config['loading'].get('interpolated_DENSE_key', 'DENSE_displacement_field_merged')

    # split displacement field into X and Y if needed
    if interpolated_DENSE_key in raw_slices_data_list[0].keys() and interpolated_DENSE_key+'_X' not in raw_slices_data_list[0].keys():
        print('splitting displacement field into X and Y...', end='')
        for datum in raw_slices_data_list:
            datum[interpolated_DENSE_key+'_X'] = datum[interpolated_DENSE_key][0]
            datum[interpolated_DENSE_key+'_Y'] = datum[interpolated_DENSE_key][1]
        print('DONE!')

    # take only the original (instead of interpolated) frames if needed
    is_Lagrangian_displacement = data_config['loading'].get('Lagrangian_displacement', False)
    if not use_interpolated_data:
        for datum in raw_slices_data_list:
            interpolation_indicator = datum['cine_lv_myo_masks_merged_is_interpolated_labels']
            if is_Lagrangian_displacement:
                # if displacement field is Lagrangian, then the first frame is not interpolated
                interpolation_indicator = interpolation_indicator[1:]
            datum[interpolated_DENSE_key+'_X'] = datum[interpolated_DENSE_key+'_X'][...,np.where(interpolation_indicator==0)[0]]
            datum[interpolated_DENSE_key+'_Y'] = datum[interpolated_DENSE_key+'_Y'][...,np.where(interpolation_indicator==0)[0]]
                
            
                

    
    # augmentation
    augmented_data = augment_all_data(raw_slices_data_list, data_config)
    print(type(augmented_data))
    print('len(augmentaed_data): ', len(augmented_data))
    raw_slices_data_list += augmented_data
    print(f'# of data after augmentation: {len(raw_slices_data_list)}')

    cine_DENSE_must_same_n_frame = data_config['loading'].get('cine_DENSE_must_same_n_frame', True)

    strain_matrix_n_frames = 50
    slice_data_list = []
    for slice_idx, slice_data in enumerate(raw_slices_data_list):
        subject_id = slice_data['patient_id']
        subject_LV_masks = slice_data[interpolated_cine_key]
        H, W, n_frames = subject_LV_masks.shape
        # append DENSE data if use interpolated data
        # if use_interpolated_data:
        subject_DENSE_displacement_field_X = slice_data[interpolated_DENSE_key+'_X']
        subject_DENSE_displacement_field_Y = slice_data[interpolated_DENSE_key+'_Y']
        # check if the shape of DENSE data matches the shape of cine data
        if subject_DENSE_displacement_field_X.shape != subject_LV_masks.shape and cine_DENSE_must_same_n_frame:
            print(f'Warning: shape of DENSE data {subject_DENSE_displacement_field_X.shape} does not match the shape of cine data {subject_LV_masks.shape}')
            continue
        
        subject_DENSE_displacement_field_X[np.isnan(subject_DENSE_displacement_field_X)] = 0
        subject_DENSE_displacement_field_Y[np.isnan(subject_DENSE_displacement_field_Y)] = 0
        
        # load TOS and strain matrix
        if 'TOSAnalysis' not in slice_data.keys():
            print('Warning: TOSAnalysis not found in slice_data')
            continue
        slice_TOS = slice_data['TOSAnalysis']['TOSfullRes_Jerry']
        slice_strain_matrix = slice_data['StrainInfo']['CCmidSVD'] if 'CCmidSVD' in slice_data['StrainInfo'].keys() else slice_data['StrainInfo']['CCmid']
        # slice_strain_matrix_SVD = slice_data['StrainInfo']['CCmidSVD']
        if slice_strain_matrix.shape[1] > strain_matrix_n_frames:
            slice_strain_matrix_aligned = slice_strain_matrix[:, :strain_matrix_n_frames]
        elif slice_strain_matrix.shape[1] < strain_matrix_n_frames:
            slice_strain_matrix_aligned = np.zeros((slice_strain_matrix.shape[0], strain_matrix_n_frames))
            slice_strain_matrix_aligned[:, :slice_strain_matrix.shape[1]] = slice_strain_matrix
        else:
            slice_strain_matrix_aligned = slice_strain_matrix
        # slice_TOS = slice_data['TOSAnalysis']['TOS18_Jerry']
        # slice_strain_matrix = slice_data[0]['TransmuralStrainInfo']['Ecc']['mid'].T

        sector_LMA_labels = (slice_TOS > LMA_threshold).astype(int)        

        
        datum = {
            'subject_id': subject_id,
            'slice_idx': slice_idx,
            'slice_full_id': f'{subject_id}-{slice_idx}',
            'slice_LMA_label': int(slice_TOS.max() > LMA_threshold),
            'TOS': slice_TOS,
            'sector_LMA_labels': sector_LMA_labels,
            'strain_matrix': slice_strain_matrix_aligned,
            'LV_masks': subject_LV_masks,
            'DENSE_displacement_field_X': subject_DENSE_displacement_field_X,
            'DENSE_displacement_field_Y': subject_DENSE_displacement_field_Y,
            'DENSE_slice_mat_filename': str(slice_data['DENSE_slice_mat_filename']),
            'augmented': slice_data['augmented'],
            'augment_translate_y': slice_data.get('augment_translate_y', -1),
            'augment_translate_x': slice_data.get('augment_translate_x', -1),
            'augment_rotate_n_sector': slice_data.get('augment_rotate_n_sector', -1),
            'augment_rotate_angle': slice_data.get('augment_rotate_angle', -1),
            'cine_slice_idx': int(slice_data.get('cine_slice_idx', -1)),
            'cine_slice_location': float(slice_data.get('cine_slice_location', -1)),
            'DENSE_slice_mat_filename': str(slice_data['DENSE_slice_mat_filename']),
            'DENSE_slice_location': float(slice_data['DENSE_slice_location']),
        }
        slice_data_list.append(datum)
    return slice_data_list

def load_cine_pairs_from_npy_file(npy_filename, data_config=None):
    LMA_threshold = data_config.get('LMA_threshold', 25)
    print('loading cine pairs from npy file...', end='')
    slices_data_list = np.load(npy_filename, allow_pickle=True).tolist()
    print('DONE!')
    print(f'{len(slices_data_list)} slices loaded')
    for datum in slices_data_list:
        datum['augmented'] = False
    
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

    # normalization
    normalize_interpolated_cine_key = data_config['loading'].get('normalize_interpolated_cine_key', False)
    def normalize_img(img):
        # normalize the range of image to [0, 1]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return img

    use_interpolated_data = data_config['loading'].get('use_interpolated_data', False)
    interpolated_cine_key = data_config['loading'].get('interpolated_cine_key', 'cine_lv_myo_masks_merged')
    interpolated_DENSE_key = data_config['loading'].get('interpolated_DENSE_key', 'DENSE_displacement_field_merged')
    feed_masks = data_config['loading'].get('feed_masks', False)
    interpolated_cine_mask_key = data_config['loading'].get('interpolated_cine_mask_key', 'cine_lv_myo_masks_merged')
    interpolated_cine_mask_dilation = data_config['loading'].get('interpolated_cine_mask_dilation', 0)
    

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
        slice_augmented = slice_data['augmented']
        # slice_TOS = slice_data['TOSAnalysis']['TOS18_Jerry']
        # slice_strain_matrix = slice_data[0]['TransmuralStrainInfo']['Ecc']['mid'].T

        if feed_masks:
            slice_cine_mask = slice_data[interpolated_cine_mask_key]
            if interpolated_cine_mask_dilation > 0:                
                for frame_idx in range(slice_cine_mask.shape[-1]):
                    slice_cine_mask[:, :, frame_idx] = dilation(slice_cine_mask[:, :, frame_idx], np.ones((interpolated_cine_mask_dilation, interpolated_cine_mask_dilation)))
                # slice_cine_mask = dilation(slice_cine_mask, np.ones((interpolated_cine_mask_dilation, interpolated_cine_mask_dilation)))
        
        for frame_idx in range(n_frames):
            if frame_idx == n_frames - 1:
                source_time_idx = frame_idx
                target_time_idx = 0
            else:                
                source_time_idx = frame_idx
                target_time_idx = frame_idx + 1
            
            source_image = subject_LV_masks[:, :, source_time_idx].astype(np.float32)
            target_image = subject_LV_masks[:, :, target_time_idx].astype(np.float32)

            if normalize_interpolated_cine_key:
                source_image = normalize_img(source_image)
                target_image = normalize_img(target_image)

            if feed_masks:
                source_mask = slice_cine_mask[:, :, source_time_idx].astype(np.float32)
                target_mask = slice_cine_mask[:, :, target_time_idx].astype(np.float32)
            else:
                source_mask = np.zeros_like(source_image)
                target_mask = np.zeros_like(target_image)

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
                'source_mask': source_mask,
                'target_mask': target_mask,
                'augmented': slice_augmented,
                'cine_slice_idx': int(slice_data['cine_slice_idx']),
                'cine_slice_location': float(slice_data['cine_slice_location']),
                'DENSE_slice_mat_filename': str(slice_data['DENSE_slice_mat_filename']),
                'DENSE_slice_location': float(slice_data['DENSE_slice_location']),
            }

            if use_interpolated_data:
                pair_data_dict['DENSE_displacement_field_X'] = subject_DENSE_displacement_field_X[:, :, frame_idx]
                pair_data_dict['DENSE_displacement_field_Y'] = subject_DENSE_displacement_field_Y[:, :, frame_idx]

            # replace the NaN in displacement field with 0
            pair_data_dict['DENSE_displacement_field_X'][np.isnan(pair_data_dict['DENSE_displacement_field_X'])] = 0
            pair_data_dict['DENSE_displacement_field_Y'][np.isnan(pair_data_dict['DENSE_displacement_field_Y'])] = 0

            # append TOS and strain matrix
            
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

import copy
def try_merge_displacements(datum: dict):
    """
    Try to merge the displacement X and Y to a single displacement field
    
    Specially, for any key of the dictionary that (1) includes "disp" and (2) have data ends with both "X" and "Y"
    e.g. "DENSE_displacement_field_X" and "DENSE_displacement_field_Y"
    The new key should be named as the original key without the "X" or "Y" at the end
    Also, is the end of the new key is "-" or "_", it should be also removed
    """
    ori_datum_keys = copy.copy(list(datum.keys()))
    for key in ori_datum_keys:
        if 'disp' in key and key.endswith('X'):
            key_Y = key[:-1] + 'Y'
            if key_Y in datum.keys():
                new_key = key[:-1]
                if new_key.endswith('_') or new_key.endswith('-'):
                    new_key = new_key[:-1]
                datum[new_key] = np.stack([datum[key], datum[key_Y]], axis=0)
                datum.pop(key)
                datum.pop(key_Y)
    return datum

def load_slices_from_npy_file(npy_filename, data_config=None):
    LMA_threshold = data_config.get('LMA_threshold', 25)
    print('loading cine data from npy file...', end='')
    slices_data_list = np.load(npy_filename, allow_pickle=True).tolist()
    print('DONE!')
    print(f'{len(slices_data_list)} slices loaded')
    for datum in slices_data_list:
        datum['augmented'] = False
    
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

    # normalization
    normalize_interpolated_cine_key = data_config['loading'].get('normalize_interpolated_cine_key', False)
    def normalize_img(img):
        # normalize the range of image to [0, 1]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return img

    
    data_to_feed = data_config['loading'].get('data_to_feed', [{'key': 'LMA_label', 'LMA_threshold': 25}])
    try_merge_displacements_flag = data_config['loading'].get('try_merge_displacements', True)
    # for additional_data_keys in ['augmented', 'cine_slice_idx', 'cine_slice_location', 'DENSE_slice_mat_filename', 'DENSE_slice_location']:
    #     data_to_feed.append({'key': additional_data_keys})
    loaded_data_list = []
    for slice_idx, datum in enumerate(slices_data_list):
        if 'TOSAnalysis' not in datum.keys():
            print('Warning: TOSAnalysis not found in slice_data of patient', datum['patient_id'])
            continue
        loaded_datum = get_data_from_slice(datum, data_to_feed)
        loaded_datum['augmented'] = datum['augmented']
        loaded_datum['cine_slice_idx'] = int(datum['cine_slice_idx'])
        loaded_datum['cine_slice_location'] = float(datum['cine_slice_location'])
        loaded_datum['DENSE_slice_mat_filename'] = str(datum['DENSE_slice_mat_filename'])
        loaded_datum['DENSE_slice_location'] = float(datum['DENSE_slice_location'])
        
        loaded_datum['subject_id'] = datum['patient_id']
        loaded_datum['slice_idx'] = slice_idx
        loaded_datum['slice_full_id'] = f'{datum["patient_id"]}-{slice_idx}'
        

        if try_merge_displacements_flag:
            loaded_datum = try_merge_displacements(loaded_datum)

        loaded_data_list.append(loaded_datum)
    return loaded_data_list
        

