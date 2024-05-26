from modules.data.augmentation.affine import rotate, translate

import numpy as np
import copy
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