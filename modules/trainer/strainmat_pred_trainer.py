import torch
from torch.utils.data.dataloader import DataLoader
from modules.data.dataloader import SliceDataLoader
from tqdm import tqdm
import lagomorph as lm

from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import copy
from torch.utils.tensorboard import SummaryWriter
import wandb
# from torch.optim.lr_scheduler
from modules.loss import LossCalculator
from sklearn.metrics import roc_auc_score
import json

class DummyLrScheduler:
    """
    Dummy learning rate scheduler that does nothing
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass




def get_lr_scheduler(optimizer, lr_scheduler_config):
    lr_scheduler_type = lr_scheduler_config['type']
    lr_scheduler_enabled = lr_scheduler_config['enable']
    if not lr_scheduler_enabled:
        lr_scheduler = DummyLrScheduler(optimizer)
    elif lr_scheduler_type in ['CosineAnnealingLR']:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=lr_scheduler_config['T_max'])
    else:
        raise NotImplementedError(f'Learning rate scheduler {lr_scheduler_type} not implemented')
    return lr_scheduler


class StrainmatPredTrainer:
    def __init__(self, trainer_config, device=None, full_config=None) -> None:
        self.trainer_config = trainer_config
        self.full_config = full_config
        self.device = device if device is not None else torch.device('cpu')
        self.LMA_task = trainer_config.get('LMA_task', 'TOS_regression')

    # def setup_

    def train(self, models: dict, datasets: dict, trainer_config=None, full_config=None, device=None, use_tensorboard=False, tensorboard_log_dir=None, early_stop=True, use_wandb=False, exp_save_dir = './test_results', enable_wandb_upload=True):
        import numpy as np
        # allow for overloading config
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device


        # task-related parameters
        self.LMA_modality = used_train_config.get('LMA_modality', 'myocardium_mask')
        self.LMA_task = used_train_config.get('LMA_task', 'TOS_regression')

        # unpack models
        masks_to_strain_mat_model = models['masks_to_strain_mat']

        # unpack datasets
        test_as_val = used_train_config.get('test_as_val', False)
        train_dataset = datasets['train']
        if test_as_val:
            val_dataset = datasets['test']
        else:
            val_dataset = datasets['val']
        test_dataset = datasets['test']
        # Build dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=used_train_config['batch_size'], shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0)


        # loss calculator
        self.loss_calculator = LossCalculator(used_full_config['losses'])


        # optimizers
        optimizer_type = used_train_config['optimizers']['LMA'].get('type', 'Adam') 
        if optimizer_type == 'Adam':
            masks_to_strain_mat_optimizer = torch.optim.Adam(
                masks_to_strain_mat_model.parameters(), 
                lr=used_train_config['optimizers']['LMA']['learning_rate'],
                weight_decay=used_train_config['optimizers']['LMA']['weight_decay'])
        elif optimizer_type == 'SGD':
            masks_to_strain_mat_optimizer = torch.optim.SGD(
                masks_to_strain_mat_model.parameters(), 
                lr=used_train_config['optimizers']['LMA']['learning_rate'],
                momentum=used_train_config['optimizers']['LMA'].get('momentum', 0),
                weight_decay=used_train_config['optimizers']['LMA']['weight_decay'])
        masks_to_strain_mat_lr_scheduler = get_lr_scheduler(masks_to_strain_mat_optimizer, used_train_config['optimizers']['LMA']['lr_scheduler'])


        # progress bar
        progress_bar = tqdm(range(used_train_config['epochs']))
        # early stop parameters
        if early_stop:
            best_LMA_model = None
            best_val_loss = float('inf')
            best_epoch = 0
            best_epoch_loss_dict = {}
            epochs_without_improvement = 0
            epochs_without_improvement_tolerance = used_train_config.get('epochs_without_improvement_tolerance', 10)

        # Experiment tracking
        if use_tensorboard:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            writer = SummaryWriter(log_dir=Path(tensorboard_log_dir, f'{current_time}'))
        
        
        if use_wandb:
            import random
            import string

            # wandb_base_dir = './exp_results'
            wandb_base_dir = '/p/miauva/data/Jerry/exp-results'
            # exp_random_ID = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
            # exp_name = full_config['info'].get('experiment_name', 'unnamed') + f'-{exp_random_ID}'
            exp_name = full_config['info'].get('experiment_name', 'unnamed')
            exp_name_with_date = datetime.datetime.now().strftime('%Y-%m-%d')+ '-' + exp_name
            # use_wandb = full_config['others'].get('use_wandb', False)
            wandb_path = Path(wandb_base_dir, f"{datetime.datetime.now().strftime('%Y-%m')}/{exp_name_with_date}-wandb")
            wandb_path.mkdir(parents=True, exist_ok=True)
            
            wandb_visualize_interval = full_config['others'].get('wandb_visualize_interval', -1) # -1 means no visualization
            # if wandb_visualize_interval is a float number, it means the interval is a fraction of the total number of epochs
            # convert it to an integer
            if wandb_visualize_interval > 0 and isinstance(wandb_visualize_interval, float):
                wandb_visualize_interval = int(wandb_visualize_interval * used_train_config['epochs'])
            
            wandb_experiment = wandb.init( 
                project = full_config['info'].get('project_name', 'trials'),
                entity = "jrxing", 
                save_code = True,
                name = full_config['info']['experiment_name'] if full_config['info'].get('use_experiment_name', False) else None,
                dir = wandb_path,
                resume = 'allow', 
                anonymous = 'must',
                mode='online' if use_wandb else 'disabled')
                # mode = 'disabled')
            exp_save_dir = wandb.run.dir
        else:
            # exp_save_dir = Path(used_full_config['others']['save_dir'])
            exp_save_dir = './test_results'
            wandb_experiment = None

        # evaluation metrics
        # self.batch_train_loss_dict = {}
        self.training_epoch_loss_dict_list = []
        # self.batch_val_loss_dict = {}
        # self.epoch_val_loss_dict = {}
        
        # Training Loop
        for epoch in progress_bar:
            # initialize epoch_loss_dict
            epoch_loss_dict = {}
            # Train
            masks_to_strain_mat_model.train()
            for train_batch_idx, train_batch in enumerate(train_dataloader):
                # forward pass
                train_loss, training_batch_loss_dict, _, _ = self.batch_forward(train_batch, masks_to_strain_mat_model, epoch_loss_dict, loss_name_prefix='train')

                # backward pass
                masks_to_strain_mat_optimizer.zero_grad()
                train_loss.backward()
                masks_to_strain_mat_optimizer.step()

                batch_eval_dict = self.batch_eval(train_batch, train_batch)
            
            # update learning rate
            masks_to_strain_mat_lr_scheduler.step()
            
            # update progress bar
            progress_bar.set_description(f'Epoch {epoch} | Train Loss {train_loss.item():.3e}')  

            # Validate
            epoch_total_val_loss = 0
            masks_to_strain_mat_model.eval()
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_dataloader):
                    val_loss, val_batch_loss_dict, _, _ = self.batch_forward(val_batch, masks_to_strain_mat_model, epoch_loss_dict, loss_name_prefix='val')
                    epoch_total_val_loss += val_loss.item()
                    batch_eval_dict = self.batch_eval(val_batch, val_batch)
        
            # print epoch_loss_dict with indentation
            # convert the data in epoch_loss_dict to json serializable format
            # import numpy as np
            for key, value in epoch_loss_dict.items():
                if isinstance(value, np.ndarray):
                    epoch_loss_dict[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    epoch_loss_dict[key] = value.item()
            print(json.dumps(epoch_loss_dict, indent=4))

            # append epoch_loss_dict to epoch_loss_dict_list
            self.training_epoch_loss_dict_list.append(epoch_loss_dict)

            if use_wandb and enable_wandb_upload:
                wandb_experiment.log(epoch_loss_dict, step=epoch)                
            
            if use_tensorboard:
                for loss_name, loss_value in epoch_loss_dict.items():
                    writer.add_scalar(loss_name, loss_value, epoch)
                # writer.add_scalar('validation?total_loss', epoch_total_val_loss, epoch)
                # writer.close()
                writer.flush()

            if early_stop:
                # save best model
                if epoch_total_val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = epoch_total_val_loss
                    best_model = copy.deepcopy(masks_to_strain_mat_model)
                    best_epoch_loss_dict = copy.deepcopy(epoch_loss_dict)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f'Epochs without improvement: {epochs_without_improvement} / {epochs_without_improvement_tolerance}')

                if use_wandb and enable_wandb_upload:
                    best_epoch_loss_dict_wandb = {}
                    for key, value in best_epoch_loss_dict.items():
                        # append 'best-' to the second level of the key
                        best_key = '/'.join(key.split('/')[:1] + ['best-' + key.split('/')[1]])
                        best_epoch_loss_dict_wandb[best_key] = value
                    best_epoch_loss_dict_wandb['best_epoch'] = best_epoch
                    wandb_experiment.log(best_epoch_loss_dict_wandb, step=epoch)

                # early stopping
                if epochs_without_improvement >= epochs_without_improvement_tolerance:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        if use_tensorboard:
            writer.close()

        if early_stop:
            masks_to_strain_mat_model = best_model
            epoch_loss_dict = best_epoch_loss_dict
        
        self.epoch_loss_dict = epoch_loss_dict
        # convert the data in epoch_loss_dict to json serializable format
        import numpy as np
        for key, value in self.epoch_loss_dict.items():
            if isinstance(value, np.ndarray):
                self.epoch_loss_dict[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                self.epoch_loss_dict[key] = value.item()

        exp_dict = {
            'epoch': epoch,
            'epoch_loss_dict': self.epoch_loss_dict,
            'best_epoch': best_epoch,
            'epoch_loss_dict_list': self.training_epoch_loss_dict_list,
            'masks_to_strain_mat': masks_to_strain_mat_model,
        }

        return exp_dict, wandb_experiment



    
    def batch_forward(self, batch, model, epoch_loss_dict, loss_name_prefix):
        # myo_mask_volume = batch['myo_mask_volume'].to(self.device)
        myo_disp_field_volume = batch['displacement_field'].to(self.device)
        strain_mat = batch['strain_mat'].to(self.device)
        strain_mat_pred = model(myo_disp_field_volume)

        pred_dict = {
            'strainmat': strain_mat_pred['strainmat'],
        }
        target_dict = {
            'strainmat': strain_mat,
        }
        total_loss, losses_values_dict = self.loss_calculator(pred_dict, target_dict)

        # update the epoch loss dict
        for loss_name, loss_value in losses_values_dict.items():
            record_name = f'{loss_name_prefix}/{loss_name}'
            if record_name not in epoch_loss_dict.keys():
                epoch_loss_dict[record_name] = loss_value
            else:
                epoch_loss_dict[record_name] += loss_value

        # update the pred_dict key names by adding the loss_name_prefix
        # pred_dict = {f'{loss_name_prefix}/{k}': v for k, v in pred_dict.items()}
        return total_loss, losses_values_dict, pred_dict, target_dict

    def batch_eval(self, batch_pred, batch):
        return {}
    

    def test(self, models, datasets, trainer_config=None, full_config=None, device=None, wandb_experiment=None, target_dataset='test'):
        import numpy as np
        # allow for overloading config
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # task-related parameters
        self.LMA_modality = used_train_config.get('LMA_modality', 'myocardium_mask')
        self.LMA_task = used_train_config.get('LMA_task', 'TOS_regression')

        # unpack models
        masks_to_strain_mat_model = models['masks_to_strain_mat']

        # unpack datasets
        test_dataset = datasets[target_dataset]
        # Build dataloaders
        batch_size = used_train_config['batch_size']
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # loss calculator
        self.loss_calculator = LossCalculator(used_full_config['losses'])

        test_preds = []
        test_performance_dict = {}
        test_loss_dict = {}
        with torch.no_grad():
            masks_to_strain_mat_model.eval()
            for test_batch_idx, test_batch in enumerate(test_dataloader):
                # forward pass
                test_loss, test_batch_loss_dict, test_batch_pred_dict, target_dict = self.batch_forward(test_batch, masks_to_strain_mat_model, test_loss_dict, loss_name_prefix='test')
                batch_eval_dict = self.batch_eval(test_batch, test_batch)

                # break the batch into individual images and append to test_preds
                curr_batch_size = test_batch['displacement_field'].shape[0]
                for i in range(curr_batch_size):
                    test_pred_dict = {}
                    # copy all key-value from batch to test_batch_pred_dict and test_batch if
                    # (1) the value is not a torch.Tensor or np.ndarray, or
                    # (2) the value is a torch.Tensor or np.ndarray and the shape of the value is the same as the shape of the batch
                    for k, v in test_batch_pred_dict.items():
                        if not isinstance(v, (torch.Tensor, np.ndarray)):
                            test_pred_dict[k+'_pred'] = v[i]
                        elif isinstance(v, torch.Tensor) and v.shape == test_batch_pred_dict[k].shape:
                            test_pred_dict[k+'_pred'] = v[i].cpu().numpy()
                    
                    for k, v in test_batch.items():
                        if not isinstance(v, (torch.Tensor, np.ndarray)):
                            test_pred_dict[k] = v[i]
                        elif isinstance(v, torch.Tensor) and v.shape == test_batch[k].shape:
                            test_pred_dict[k] = v[i].cpu().numpy()
                    
                    # Add the test_pred_dict to test_preds
                    test_preds.append(test_pred_dict)

        if wandb_experiment is not None:
            wandb_experiment.log(test_loss_dict)
        print('inference_performance_dict: ', test_loss_dict)
        return test_preds, test_loss_dict, wandb_experiment