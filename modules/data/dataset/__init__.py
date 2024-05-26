from modules.data.dataset.registration_dataset import BasicRegistrationDataset
from modules.data.dataset.LMA_dataset import LMADataset
from modules.data.dataset.strainmat_dataset import StrainMatDataset
from modules.data.dataset.joint_dataset import JointDataset
def build_datasets(datasets_configs, data_splits, all_config=None):
    datasets = {}
    for dataset_name, dataset_config in datasets_configs.items():
        data_split = data_splits[dataset_name]
        if dataset_config['type'] == 'BasicRegistrationDataset':
            datasets[dataset_name] = BasicRegistrationDataset(
                data_split['data'], 
                dataset_name=dataset_name,
                config=dataset_config,
                full_config=all_config)
        elif dataset_config['type'] == 'LMADataset':
            datasets[dataset_name] = LMADataset(
                data_split['data'], 
                dataset_config=dataset_config,
                full_config=all_config, 
                dataset_name=dataset_name)
        elif dataset_config['type'] == 'StrainMatDataset':
            datasets[dataset_name] = StrainMatDataset(
                data_split['data'], 
                dataset_config=dataset_config,
                full_config=all_config, 
                dataset_name=dataset_name)
        elif dataset_config['type'] == 'JointDataset':
            datasets[dataset_name] = JointDataset(
                data_split['data'], 
                dataset_config=dataset_config,
                full_config=all_config, 
                dataset_name=dataset_name)
        else:
            raise ValueError(f'Unknown dataset type: {dataset_config["type"]}')
    return datasets