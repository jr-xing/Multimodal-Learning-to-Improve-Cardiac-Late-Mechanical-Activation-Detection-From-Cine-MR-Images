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
import numpy as np
from modules.data import split_vol_to_registration_pairs

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


class JointRegisterStrainmatLMATrainer:
    def __init__(self, trainer_config, device=None, full_config=None) -> None:
        self.trainer_config = trainer_config
        self.full_config = full_config
        self.device = device if device is not None else torch.device('cpu')
        self.LMA_task = trainer_config.get('LMA_task', 'TOS_regression')

    # def setup_
    def build_optimizer(self, model, optimizer_config):
        optimizer_type = optimizer_config['type']
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay'])
        elif optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0),
                weight_decay=optimizer_config['weight_decay'])
        else:
            raise NotImplementedError(f'Optimizer {optimizer_type} not implemented')
        return optimizer

    def train(self, models: dict, datasets: dict, trainer_config=None, full_config=None, device=None, use_tensorboard=False, tensorboard_log_dir=None, early_stop=True, use_wandb=False, wandb_exp=None, exp_save_dir = './test_results', enable_wandb_upload=True, prefix=''):
        import numpy as np
        # allow for overloading config
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device


        # task-related parameters
        self.LMA_modality = used_train_config.get('LMA_modality', 'myocardium_mask')
        self.LMA_task = used_train_config.get('LMA_task', 'TOS_regression')
        self.LMA_threshold = used_train_config.get('LMA_threshold', 20)

        # unpack models
        
        joint_register_strainmat_model = models['joint_register_strainmat']
        LMA_model = models['LMA']

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
        joint_register_strainmat_model_optimizer = self.build_optimizer(joint_register_strainmat_model, used_train_config['optimizers']['joint_register_strainmat'])
        LMA_model_optimizer = self.build_optimizer(LMA_model, used_train_config['optimizers']['LMA'])
        joint_register_strainmat_model_lr_scheduler = get_lr_scheduler(joint_register_strainmat_model_optimizer, used_train_config['optimizers']['joint_register_strainmat']['lr_scheduler'])
        LMA_model_lr_scheduler = get_lr_scheduler(LMA_model_optimizer, used_train_config['optimizers']['LMA']['lr_scheduler'])


        # progress bar
        progress_bar = tqdm(range(used_train_config['epochs']))
        # early stop parameters
        if early_stop:
            best_LMA_model = None
            best_val_loss = float('inf')
            best_epoch = 0
            # best_models = {}
            best_epoch_loss_dict = {}
            epochs_without_improvement = 0
            epochs_without_improvement_tolerance = used_train_config.get('epochs_without_improvement_tolerance', 10)

        # Experiment tracking
        if use_tensorboard:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            writer = SummaryWriter(log_dir=Path(tensorboard_log_dir, f'{current_time}'))
        
        
        if use_wandb and wandb_exp is None:
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
        elif use_wandb and wandb_exp is not None:
            wandb_experiment = wandb_exp
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
            joint_register_strainmat_model.train()
            LMA_model.train()
            for train_batch_idx, train_batch in enumerate(train_dataloader):
                # forward pass
                train_loss, training_batch_loss_dict, _, _ = self.batch_forward(train_batch, [joint_register_strainmat_model, LMA_model], epoch_loss_dict, loss_name_prefix=f'{prefix}train', training_config=used_train_config)

                # backward pass
                joint_register_strainmat_model_optimizer.zero_grad()
                LMA_model_optimizer.zero_grad()
                train_loss.backward()
                joint_register_strainmat_model_optimizer.step()
                LMA_model_optimizer.step()

                batch_eval_dict = self.batch_eval(train_batch, train_batch)
            
            # update learning rate
            joint_register_strainmat_model_lr_scheduler.step()
            LMA_model_lr_scheduler.step()
            
            # update progress bar
            progress_bar.set_description(f'Epoch {epoch} | Train Loss {train_loss.item():.3e}')  

            # Validate
            epoch_total_val_loss = 0
            joint_register_strainmat_model.eval()
            LMA_model.eval()
            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_dataloader):
                    val_loss, val_batch_loss_dict, _, _ = self.batch_forward(val_batch, [joint_register_strainmat_model, LMA_model], epoch_loss_dict, loss_name_prefix=f'{prefix}train')
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
                    best_joint_register_strainmat_model = copy.deepcopy(joint_register_strainmat_model)
                    best_LMA_model = copy.deepcopy(LMA_model)
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
            joint_register_strainmat_model = best_joint_register_strainmat_model
            LMA_model = best_LMA_model
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
            'joint_register_strainmat_model': joint_register_strainmat_model,
            'LMA_model': LMA_model,
        }
        # add prefix to each key in exp_dict
        exp_dict = {f'{prefix}{k}': v for k, v in exp_dict.items()}

        return exp_dict, wandb_experiment



    
    def batch_forward(self, batch, models, epoch_loss_dict, loss_name_prefix, training_config={}):
        joint_register_strainmat_model, LMA_model = models

        # myo_mask_volume = batch['myo_mask_volume'].to(self.device)
        myo_mask_volume = batch['cine_myo_mask'].to(self.device)
        displacement_type = training_config.get('displacement_type', 'Lagrangian')
        src_vol, tar_vol = split_vol_to_registration_pairs(myo_mask_volume, split_method=displacement_type, output_dim=3)

        strain_mat = batch['strain_matrix'].to(self.device)
        joint_register_strainmat_pred = joint_register_strainmat_model.forward_volume(src_vol, tar_vol)
        LMA_pred = LMA_model(joint_register_strainmat_pred['strain_matrix'])

        LMA_sector_labels_GT = (batch['TOS'] > self.LMA_threshold).to(torch.long)
        LMA_sector_labels_pred = torch.stack([LMA_pred['TOS'] <= self.LMA_threshold, LMA_pred['TOS'] > self.LMA_threshold], dim=1).to(float)

        pred_dict = {
            'strainmat': joint_register_strainmat_pred['strain_matrix'],
            'deformed_source': joint_register_strainmat_pred['deformed_source'],
            'TOS': LMA_pred['TOS'],
            'velocity': joint_register_strainmat_pred['velocity'],
            'momentum': joint_register_strainmat_pred['momentum'],
            'sector_LMA_labels': LMA_sector_labels_pred,
        }
        target_dict = {
            'strainmat': strain_mat,
            'registration_target': tar_vol,
            'TOS': batch['TOS'].to(self.device),
            'sector_LMA_labels': LMA_sector_labels_GT.to(self.device),
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
    

    def test(self, models, datasets, trainer_config=None, full_config=None, device=None, wandb_experiment=None, target_dataset='test', prefix=''):
        import numpy as np
        # allow for overloading config
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # task-related parameters
        self.LMA_modality = used_train_config.get('LMA_modality', 'myocardium_mask')
        self.LMA_task = used_train_config.get('LMA_task', 'TOS_regression')

        # unpack models
        print('existing models: ', models.keys())
        LMA_task = used_train_config.get('LMA_task', 'TOS_regression')
        joint_register_strainmat_model = models[f'{prefix}joint_register_strainmat_model']
        LMA_model = models[f'{prefix}LMA_model']


        # unpack datasets
        test_dataset = datasets[target_dataset]
        # Build dataloaders
        batch_size = used_train_config['batch_size']
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # loss calculator
        self.loss_calculator = LossCalculator(used_full_config['losses'])

        test_preds = []
        test_performance_dict = {}
        test_sector_num = 0
        test_sector_error = 0
        with torch.no_grad():
            joint_register_strainmat_model.eval()
            LMA_model.eval()
            for test_batch_idx, test_batch in enumerate(test_dataloader):
                # forward pass
                test_loss, test_batch_loss_dict, test_batch_pred_dict, target_dict = self.batch_forward(test_batch, [joint_register_strainmat_model, LMA_model], test_performance_dict, loss_name_prefix=f'{prefix}test')
                batch_eval_dict = self.batch_eval(test_batch, test_batch)

                if LMA_task == 'TOS_regression':
                    test_TOS_pred = test_batch_pred_dict['TOS'].cpu()
                    test_batch_size, test_TOS_n_sectors = test_TOS_pred.shape # (N, 126)
                    test_TOS_GT = test_batch['TOS']
                    # test_TOS_MSE += nn.MSELoss()(test_TOS_pred, test_TOS_GT).item() * test_batch_size
                    test_sector_num += test_batch_size * test_TOS_n_sectors
                    test_sector_error += torch.sum(torch.abs(test_TOS_pred - test_TOS_GT)).item() 

                # break the batch into individual images and append to test_preds
                curr_batch_size = test_batch['TOS'].shape[0]
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

        if LMA_task == 'LMA_slice_classification':
            pass
            # test_classification_accuracy = test_classification_correct_pred_num / test_data_num
            # test_performance_dict[f'final-{target_dataset}/LMA_accuracy'] = test_classification_accuracy
        elif LMA_task == 'LMA_sector_classification':
            pass
            # test_classification_accuracy = test_classification_correct_pred_num / test_sector_num
            # # if test_classification_accuracy is pytorch tensor, convert it to numpy
            # if isinstance(test_classification_accuracy, torch.Tensor):
            #     test_classification_accuracy = test_classification_accuracy.item()
            # test_performance_dict[f'final-{target_dataset}/LMA_accuracy'] = test_classification_accuracy
        elif LMA_task == 'TOS_regression':
            # test_classification_accuracy = None
            test_overall_sector_error = test_sector_error / test_sector_num
            test_performance_dict[f'{prefix}final-{target_dataset}/sector_error'] = test_overall_sector_error
        
        if wandb_experiment is not None:
            wandb_experiment.log(test_performance_dict)
        print('inference_performance_dict: ', test_performance_dict)
        return test_preds, test_performance_dict, wandb_experiment

    def visualize_pred_regression(self, preds, n_vis=5, vis_indices=None, save_plots=False, save_dir=None, save_name=''):
        # plot the strain matrices and TOS curves
        if vis_indices is not None:
            n_vis = len(vis_indices)
        else:
            vis_indices = np.random.randint(0, len(preds), n_vis)
        # vis_indices = np.random.randint(0, len(preds), n_vis)
        fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        for plot_idx, vis_test_idx in enumerate(vis_indices):
            axs[plot_idx].pcolor(preds[vis_test_idx]['strain_matrix'][0], cmap='jet', vmin=-0.3, vmax=0.3)
            axs[plot_idx].plot(preds[vis_test_idx]['TOS']/17+1, np.arange(126), color='black')
            axs[plot_idx].plot(preds[vis_test_idx]['TOS_pred']/17+1, np.arange(126), color='red', linestyle='--')
            # axs[plot_idx].set_xlim([0, 50])
        if save_plots:
            if save_dir is None:
                save_dir = Path(self.full_config['others']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / save_name)
        return fig, axs