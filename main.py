# %%
# setup device
import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%  1. setup arguments
import numpy as np
from pathlib import Path
from modules.config.config import load_config_from_json
from modules.config.config import get_args, update_config_by_args, update_config_by_undefined_args
args, undefined_args = get_args()
print('args: \n', args, '\n')

config = load_config_from_json(args.config_file)
# override config file wandb setting if given command line parameter
config = update_config_by_args(config, args)
config = update_config_by_undefined_args(config, undefined_args)


#%% 2. load all data
from modules.data import load_data
all_data = load_data(config['data'])

# 3. data splitting
from modules.data.data_split import split_data
data_splits = split_data(all_data, config['data_split'])
# print the length of each split
for split_name, split in data_splits.items():
    split_unique_patient_names = list(set([d['subject_id'] for d in split['data']]))
    print(f'split {split_name}: {len(split["data"])} from {len(split_unique_patient_names)} patients')

# 4. Building dataset
from modules.data.dataset import build_datasets
datasets = build_datasets(config['datasets'], data_splits)
# print the length of each dataset
for dataset_name, dataset in datasets.items():
    print(f'dateset {dataset_name}: {len(dataset)}')


# 6. Building model
from models import build_model
networks = {}
for model_name, model_config in config['networks'].items():
    networks[model_name] = build_model(model_config)
    networks[model_name] = networks[model_name].to(device)

#%% 7. Training
from modules.trainer import build_trainer
# config['training']['epochs'] = 10
# config['training']['batch_size'] = 5
training_seed = config['training'].get('seed', 2434)
torch.manual_seed(training_seed)
trainer = build_trainer(config['training'], device, config)

inference_only = config['training'].get('inference_only', False)
if not inference_only:
    print('training...', end='')
    trained_models, wandb_experiment = trainer.train(
        models=networks, 
        datasets=datasets, 
        trainer_config=config['training'], 
        full_config=config, 
        device=device,
        use_tensorboard=False,
        tensorboard_log_dir='tensorboard',
        use_wandb=config['others'].get('use_wandb', True),
        enable_wandb_upload=config['others'].get('enable_wandb_upload', True))
    print('done')
else:
    print('Skip training, only do inference')

#%% 8.inference
print('Inferencing on val data...', end='')
val_pred, val_performance_dict, _ = trainer.test(
    models=trained_models, 
    datasets=datasets, 
    trainer_config=config['training'], 
    full_config=config, 
    device=device,
    wandb_experiment=wandb_experiment,
    target_dataset='val')
print('done')

print('Inferencing on test data...', end='')
test_pred, test_performance_dict, _ = trainer.test(
    models=trained_models, 
    datasets=datasets, 
    trainer_config=config['training'], 
    full_config=config, 
    device=device,
    wandb_experiment=wandb_experiment,
    target_dataset='test')
print('done')


#%% 9. save inference results
from pathlib import Path
saving_dir = Path(config['saving'].get('saving_dir', './test_results'))
saving_dir.mkdir(parents=True, exist_ok=True)
print('experiment results saving dir: ', saving_dir)

# save val and test predictions as npy file
val_save_filename = config['saving'].get('val_save_filename', 'val_pred.npy')
test_save_filename = config['saving'].get('test_save_filename', 'test_pred.npy')
np.save(Path(saving_dir, val_save_filename), val_pred)
np.save(Path(saving_dir, test_save_filename), test_pred)