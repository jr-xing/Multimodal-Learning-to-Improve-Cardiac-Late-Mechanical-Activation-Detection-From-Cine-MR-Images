import random
import re
def split_data(data: list, config = {'method': 'fixed_ratio', 'paras': {'training_ratio': 0.8}}):
    method = config['method']
    splits = config['splits']
    # keep_augmented = config.get('keep_augmented', False)
    if method == 'by_pattern':
        datalists = data_split_by_pattern(data, splits)
    elif method == 'by_ratio':
        datalists = data_split_by_ratio(data, config)
    elif method == 'by_count':
        datalists = data_split_by_count(data, config)
    else:
        raise ValueError(f'Unsupported data split method: {method}')
    for split_name, split_config in splits.items():
        split_keep_augmented = split_config.get('keep_augmented', False)
        if not split_keep_augmented:
            print(f'Filtering out augmented data in {split_name}')
            split_original_len = len(datalists[split_name]['data'])
            datalists[split_name]['data'] = [datum for datum in datalists[split_name]['data'] if datum.get('augmented', False) == False]
            split_filtered_len = len(datalists[split_name]['data'])
            print(f'Filtered out {split_original_len - split_filtered_len} augmented data in {split_name}')
            print(f'Now {split_name} has {split_filtered_len} data')
    return datalists

def match_name_with_patterns(name, include_patterns, exclude_patterns, verbose=False):
    include_patterns = [include_patterns] if type(include_patterns) is str else include_patterns
    exclude_patterns = [exclude_patterns] if type(exclude_patterns) is str else exclude_patterns
    # Check if current name is explicitly excluded
    exclude_matching_results = [re.findall(pattern, name) for pattern in exclude_patterns]
    exclude_matching_results_len = [len(matching_result) for matching_result in exclude_matching_results]        
    if len(exclude_patterns) > 0 and max(exclude_matching_results_len) > 0:
        # If has exclusion terms and explicitly excluded, return False
        # print('Excluded: ', name)
        if verbose:
            print(f'{name} not matched')
        return False
    else:
        # If not explicitly excluded, check whether it's explicitly included
        include_matching_results = [re.findall(pattern, name) for pattern in include_patterns]        
        include_matching_results_len = [len(matching_result) for matching_result in include_matching_results]
        # Return True is explicitly included, else return False
        match_result = (max(include_matching_results_len) > 0)
        if verbose:            
            print(f'{name} not matched') if match_result else print(f'{name} matched')
        return match_result

def data_split_by_pattern(data: list,   paras=None):
    if paras is None:
        paras = [
            {"role": "train_labeled", "patterns": "^LGE1K-+"},
            {"role": "train_unlabeled", "patterns": "^Ken-+",
                "exclude_patterns": [".*Cihurub.*",".*Bimoqua.*",".*Bujisri.*",".*ClinTrial42.*",".*114_42_BC_MR.*"]
            },
            {"role": "test",
                "patterns": [".*Cihurub.*",".*Bimoqua.*",".*Bujisri.*",".*ClinTrial42.*",".*114_42_BC_MR.*"]
            }]
    datasets = {}
    for dataaet_name, dataset_info in paras.items():
        dataset_role = dataset_info['role']
        dataset_patterns = dataset_info.get('patterns', ['^.*$'])
        dataset_exclude_patterns = dataset_info.get('exclude_patterns', [])
        dataset_data = [datum for datum in data if match_name_with_patterns(datum['full_name'], dataset_patterns, dataset_exclude_patterns, verbose=False)]
        for datum_idx, datum in enumerate(dataset_data):
            datum['idx_in_dataset'] = datum_idx
        datasets[dataset_role] = {'data': dataset_data, 'info': dataset_info}

    return datasets

def ratio_to_count(n_data, paras):
    """
    convert ratio (including possible "rest") into actual counts
    """
    accumulated_count = 0
    for dataset_info in paras:
        if type(dataset_info['ratio']) is float:
            dataset_info['count'] = int(n_data * dataset_info['ratio'])
            accumulated_count += dataset_info['count']
    # Convert "rest" to count
    for dataset_info in paras:
        if dataset_info['ratio'] == 'rest':
            dataset_info['count'] = n_data - accumulated_count
    return paras
    

def data_split_by_ratio(data, config):
    """
    "data-split": {
        "method": "by_ratio",
        "balance_classes": true,
        "shuffle": false,
        "label_role": "label",
        "paras": [{
                "role": "train",
                "ratio": 0.7
            },
            {
                "role": "test",
                "ratio": "rest"
            }]
        },
    """
    # datasets = {'data': dataset_data, 'info': dataset_info}
    datasets = {}
    for dataset_info in config['paras']:
        datasets[dataset_info['role']] = {'data': [], 'info': dataset_info}
    

    balance_classes = config.get('balance_classes', True)
    shuffle = config.get('shuffle', False)
    if shuffle:
        random.shuffle(data)
    if balance_classes:        
        label_role = config.get('label_role', 'label')
        unique_labels = list(set([datum[label_role] for datum in data]))

        for cls_label in unique_labels:
            data_of_class = [datum for datum in data if datum[label_role] == cls_label]

            # Convert ratio into count, except the one with ratio "rest"
            config['paras'] = ratio_to_count(len(data_of_class), config['paras'])
            
            # Split data into each datasets in order
            accumulated_count = 0
            for dataset_info in config['paras']:
                datasets[dataset_info['role']]['data'] += data_of_class[accumulated_count:accumulated_count+dataset_info['count']]
                accumulated_count += dataset_info['count']
    else:
        config['paras'] = ratio_to_count(len(data), config['paras'])
            
        # Split data into each datasets in order
        accumulated_count = 0
        for dataset_info in config['paras']:
            datasets[dataset_info['role']]['data'] = data[accumulated_count:accumulated_count+dataset_info['count']]
            accumulated_count += dataset_info['count']
    
    return datasets

def data_split_by_count(data, config):
    """
    "data-split": {
        "method": "by_count",
        "balance_classes": true,
        "shuffle": false,
        "label_role": "label",
        "paras": [{
                "role": "train",
                "count": 100
            },
            {
                "role": "test",
                "ratio": 900
            }]
        },
    """
    datasets = {}
    # print("config['paras']: ", config['paras'])
    for dataset_name, dataset_info in config['paras'].items():
        # print("dataset_info: ", dataset_info)
        datasets[dataset_info['role']] = {'data': [], 'info': dataset_info}
    
    balance_classes = config.get('balance_classes', True)
    shuffle = config.get('shuffle', False)
    if shuffle:
        random.shuffle(data)
    if balance_classes:        
        label_role = config.get('label_role', 'label')
        unique_labels = list(set([datum[label_role] for datum in data]))

        for cls_label in unique_labels:
            data_of_class = [datum for datum in data if datum[label_role] == cls_label]

            # Convert "rest" into count
            # config['paras'] = ratio_to_count(len(data_of_class), config['paras'])
            
            # Split data into each datasets in order
            accumulated_count = 0
            for dataset_name, dataset_info in config['paras'].items():
                datasets[dataset_info['role']]['data'] += data_of_class[accumulated_count:accumulated_count+dataset_info['count']]
                accumulated_count += dataset_info['count']
    else:
        # config['paras'] = ratio_to_count(len(data), config['paras'])
            
        # Split data into each datasets in order
        accumulated_count = 0
        for dataset_info in config['paras']:
            datasets[dataset_info['role']]['data'] = data[accumulated_count:accumulated_count+dataset_info['count']]
            accumulated_count += dataset_info['count']
    
    return datasets


class SplitManager:
    def __init__(self, config: dict):
        """
        config should be like:
        {
            "data_split": {
            "method": "by_pattern",        
            "shuffle": false,
            "label_role": "label",
            "cross_validation": true,
            "folds":[
                    [".*SET01-CT01.*",".*SET01-CT02.*",".*SET01-CT11.*",".*SET01-CT18.*",".*SET02-CT35.*"],
                    [".*SET01-CT20.*",".*SETOLD02-134.*",".*SET02-CT36.*",".*SET02-CT39.*",".*SET03-EC10.*"],
                    [".*SET03-EC21.*",".*SETOLD01-CRT_104.*", ".*SETOLD02-146.*", ".*SET03-UP36.*"],
                    [".*SET01-CT16.*",".*SET01-CT19.*",".*SET02-CT33.*",".*SETOLD01-CRT_120.*", ".*SET03-UP34.*"],
                    [".*SET01-CT14.*",".*SET02-CT26.*",".*SETOLD02-136.*",".*SETOLD02-148.*",".*SETOLD02-150.*"]
                ]
            },
        }

        OR
        {
            "method": "by_pattern",        
            "shuffle": false,
            "label_role": "label",
            "splits": {
                "train":{
                    "role": "train",
                    "patterns": [".*"],
                    "exclude_patterns": ["SETOLD01-CRT_111", ".*SET02-CT35.*", ".*SET01-CT14.*", ".*SET01-CT16.*",".*SET01-CT19.*",".*SET02-CT33.*",".*SETOLD01-CRT_120.*", ".*SETOLD02-134.*",".*SETOLD02-136.*",".*SETOLD02-148.*",".*SETOLD02-150.*", ".*SETOLD01-CRT_104.*", ".*SET03-UP34.*"],
                    "repeat_times": 0,
                    "keep_augmented": true
                },
                "val":{
                    "role": "val",
                    "patterns": [".*SETOLD02-134.*", ".*SET01-CT14.*",".*SET01-CT16.*",".*SET01-CT19.*",".*SET02-CT33.*",".*SETOLD01-CRT_120.*"],
                    "keep_augmented": true
                },
                "test":{
                    "role": "test",
                    "patterns": [".*SET01-CT14.*",".*SETOLD02-134.*",".*SETOLD02-136.*",".*SETOLD02-148.*",".*SETOLD02-150.*", ".*SET03-UP34.*"],
                    "keep_augmented": true
                    }
                }
        },
        
        """
        self.config = config
        self.split_setting = {}
        for key, value in config.items():
            if key not in ['folds', 'splits']:
                self.split_setting[key] = value
        self.cross_validation = config.get('cross_validation', False)
        self.n_used_folds = config.get('n_used_folds', None)
        self.build_splits()

    def build_splits(self):
        """
        Build list of split configs from self.config
        If cross_validation is True, the list should contain k-fold splits by using one fold as test, on fold as val, and the rest as train
        If cross_validation is False, the list should contain only one split. If self.config contains "split", then just copy the split config to the list; otherwise, use the last fold as test, the second last fold as val, and the rest as train
        """
        self.splits_configs = []
        if self.cross_validation:
            if self.n_used_folds is None:
                self.n_used_folds = len(self.config['folds'])
            for fold_idx, fold in enumerate(self.config['folds']):
                train_val_folds_subject_ids = [f for f_idx, f in enumerate(self.config['folds']) if f_idx != fold_idx]
                train_folds_subject_ids = []
                for f in train_val_folds_subject_ids[:-1]:
                    train_folds_subject_ids += f
                val_fold_subject_ids = train_val_folds_subject_ids[-1]
                test_fold_subject_ids = fold
                curr_fold_split_config = {
                    "train":{
                        "role": "train",
                        "patterns": train_folds_subject_ids,
                        "repeat_times": 0,
                        "keep_augmented": True
                    },
                    "val":{
                        "role": "val",
                        "patterns": val_fold_subject_ids,
                        "keep_augmented": True
                    },
                    "test":{
                        "role": "test",
                        "patterns": test_fold_subject_ids,
                        "keep_augmented": True
                        }
                }
                self.splits_configs.append(curr_fold_split_config)
        else:
            self.n_used_folds = 1
            if 'splits' in self.config:
                self.splits_configs = [self.config['splits']]
            else:
                train_val_folds_subject_ids = self.config['folds'][:-1]
                train_folds_subject_ids = []
                for f in train_val_folds_subject_ids[:-1]:
                    train_folds_subject_ids += f
                val_fold_subject_ids = train_val_folds_subject_ids[-1]
                test_fold_subject_ids = self.config['folds'][-1]
                curr_fold_split_config = {
                    "train":{
                        "role": "train",
                        "patterns": train_folds_subject_ids,
                        "repeat_times": 0,
                        "keep_augmented": True
                    },
                    "val":{
                        "role": "val",
                        "patterns": val_fold_subject_ids,
                        "keep_augmented": True
                    },
                    "test":{
                        "role": "test",
                        "patterns": test_fold_subject_ids,
                        "keep_augmented": True
                        }
                }
                self.splits_configs.append(curr_fold_split_config)
        return self.splits_configs
    

    def __getitem__(self, idx):
        split_config = {}
        split_config.update(self.split_setting)
        split_config['splits'] = self.splits_configs[idx]
        return split_config
    
    def __len__(self):
        return self.n_used_folds