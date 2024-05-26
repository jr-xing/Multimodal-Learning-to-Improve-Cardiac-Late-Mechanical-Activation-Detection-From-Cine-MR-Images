import torch
from torch.utils.data.dataloader import DataLoader
from modules.data.dataloader import SliceDataLoader
from tqdm import tqdm
import lagomorph as lm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import copy
from torch.utils.tensorboard import SummaryWriter
import wandb
# from torch.optim.lr_scheduler
from modules.loss import LossCalculator

class DummyLrScheduler:
    """
    Dummy learning rate scheduler that does nothing
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

def check_tensor_rows_identity(input_tensor):
    # check if all rows have exactly the same values as the first row
    first_row = input_tensor[0]
    all_rows_equal = True
    for row in input_tensor:
        if not torch.equal(row, first_row):
            all_rows_equal = False
            break
    return all_rows_equal

    # print(all_rows_equal)  # False

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


def merge_data_of_same_slice_from_batch(batch, reg_pred_dict, n_frames_to_use_for_regression, used_device):
    pred_displacement_fields = []
    TOS_curves = []
    sector_LMA_labels = []
    slice_slice_LMA_labels = []
    batch_slice_full_ids = list(set(batch['slice_full_id']))
    for slice_full_id in batch_slice_full_ids:                    
        # indices of the pairs with of the current slice
        slice_pair_indices = [idx for idx, id in enumerate(batch['slice_full_id']) if id == slice_full_id]
        # collect the displacement fields for the same slice
        slice_displacement_fields = [reg_pred_dict['displacement'][idx] for idx in slice_pair_indices] # list of displacement fields with shape (2, H, W)
        slice_displacement_fields = torch.stack(slice_displacement_fields, dim=0) # shape (n_pairs, 2, H, W)
        slice_displacement_fields = slice_displacement_fields.permute(1, 0, 2, 3) # shape (2, n_pairs, H, W)
        
        # align the # of frames of the displacement fields to n_frames_to_use_for_regression
        # crop if the # of frames is greater than n_frames_to_use_for_regression
        # pad if the # of frames is less than n_frames_to_use_for_regression
        n_frames = slice_displacement_fields.shape[1]
        if n_frames >= n_frames_to_use_for_regression:
            slice_displacement_fields_aligned = slice_displacement_fields[:, :n_frames_to_use_for_regression, ...]
        else:
            padding_amount = n_frames_to_use_for_regression - n_frames
            # paddings = [(0, 0)] * slice_displacement_fields.ndim
            # paddings[1] = (0, padding_amount)
            paddings = [0] * (slice_displacement_fields.ndim * 2)
            paddings[-3] = padding_amount
            slice_displacement_fields_aligned = torch.nn.functional.pad(slice_displacement_fields, paddings, mode='constant', value=0)
            # print('padding_amount: ', padding_amount)
            # print('paddings.shape: ', paddings)
            # print('slice_displacement_fields_aligned.shape: ', slice_displacement_fields_aligned.shape)
            # print('sum of padded region: ', torch.sum(slice_displacement_fields_aligned[:, n_frames:, ...]))
        
        pred_displacement_fields.append(slice_displacement_fields_aligned)

        # collect the TOS curves for the same slice
        # check whether all pairs of the current slice have the same TOS curve
        # if not, raise an error
        # if yes, add the TOS curve to TOS_curves
        slice_TOS_curves = batch['TOS'][torch.tensor(slice_pair_indices)]
        slice_sector_LMA_labels = batch['sector_LMA_labels'][torch.tensor(slice_pair_indices)]
        # print('slice_pair_indices:', type(slice_pair_indices))
        # print("batch['slice_LMA_label']: ", type(batch['slice_LMA_label']))
        slice_LMA_labels = batch['slice_LMA_label'][torch.tensor(slice_pair_indices)]
        # slice_TOS_curves_identical = check_tensor_rows_identity(slice_TOS_curves)
        # if not slice_TOS_curves_identical:
        #     raise ValueError(f'TOS curves for the same slice {slice_full_id} are not identical')
        # else:
        TOS_curves.append(slice_TOS_curves[0])
        sector_LMA_labels.append(slice_sector_LMA_labels[0])
        slice_slice_LMA_labels.append(slice_LMA_labels[0].item())
    
    # print('pred_displacement_fields shapes: ', [d.shape for d in pred_displacement_fields])
    pred_displacement_fields = torch.stack(pred_displacement_fields, dim=0) # shape (n_slices, 2, n_frames_to_use, H, W)
    TOS_curves = torch.stack(TOS_curves, dim=0).to(used_device) # shape (n_slices, 126)
    sector_LMA_labels = torch.stack(sector_LMA_labels, dim=0).to(used_device) # shape (n_slices, 126)
    
    # print('slice_slice_LMA_labels: ', slice_slice_LMA_labels)
    # slice_slice_LMA_labels = torch.concat(slice_slice_LMA_labels, dim=0).to(used_device) # shape (n_slices, )
    slice_slice_LMA_labels = torch.tensor(slice_slice_LMA_labels).to(used_device) # shape (n_slices, )
    # return pred_displacement_fields, TOS_curves, sector_LMA_labels, slice_slice_LMA_labels, batch_slice_full_ids
    return {
        'pred_displacement_fields': pred_displacement_fields,
        'TOS': TOS_curves,
        'sector_LMA_labels': sector_LMA_labels,
        'slice_LMA_label': slice_slice_LMA_labels,
        'batch_slice_full_ids': batch_slice_full_ids,        
    }

class JointRegistrationRegressionTrainer:
    def __init__(self, trainer_config, device=None, full_config=None):
        self.trainer_config = trainer_config
        self.full_config = full_config
        self.device = device if device is not None else torch.device('cpu')
        self.LMA_task = trainer_config.get('LMA_task', 'TOS_regression')

    def train(self, models: dict, datasets: dict, trainer_config=None, full_config=None, device=None, use_tensorboard=False, tensorboard_log_dir=None, early_stop=True, use_wandb=False, enable_wandb_upload=True):
        import numpy as np
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device


        

        # other parameters
        n_frames_to_use_for_regression = 48
        if 'registration_supervision' not in full_config['losses']:
            use_disp_supervision = False
        elif full_config['losses']['registration_supervision']['enable']:
            use_disp_supervision = True
        else:
            use_disp_supervision = False

        # dataloaders
        test_as_val = used_train_config.get('test_as_val', True)
        train_loader = SliceDataLoader(datasets['train'], batch_size=used_train_config['batch_size'], shuffle=True, num_workers=1)
        if test_as_val:
            val_loader = SliceDataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=1)
        else:
            val_loader = SliceDataLoader(datasets['val'], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=1)
        
        # LMA model
        LMA_task = trainer_config.get('LMA_task', 'TOS_regression')
        self.LMA_task = LMA_task
        registration_model = models['cine_registraion']
        LMA_model = models['LMA']

        # loss functions
        self.loss_calculator = LossCalculator(full_config['losses'], full_config, device=used_device)
        
        # optimizers
        registration_optimizer = torch.optim.Adam(
            registration_model.parameters(), 
            lr=used_train_config['optimizers']['registration']['learning_rate'],
            weight_decay=used_train_config['optimizers']['registration']['weight_decay'])
        LMA_optimizer = torch.optim.Adam(
            LMA_model.parameters(), 
            lr=used_train_config['optimizers']['LMA']['learning_rate'],
            weight_decay=used_train_config['optimizers']['LMA']['weight_decay'])
        registration_lr_scheduler = get_lr_scheduler(registration_optimizer, used_train_config['optimizers']['registration']['lr_scheduler'])
        LMA_lr_scheduler = get_lr_scheduler(LMA_optimizer, used_train_config['optimizers']['LMA']['lr_scheduler'])

        # build loss function
        
        # print the device of model
        print(f'Model is on {next(LMA_model.parameters()).device}')
        
        # progress bar
        progress_bar = tqdm(range(used_train_config['epochs']))

        # early stop parameters
        if early_stop:
            best_registration_model = None
            best_LMA_model = None
            best_val_loss = float('inf')
            best_val_performance = float('-inf')
            best_epoch = 0
            best_epoch_loss_dict = {}
            epochs_without_improvement = 0
            epochs_without_improvement_tolerance = 10

        
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
            exp_save_dir = './exp_results'
            wandb_experiment = None
            
            
        for epoch in progress_bar:
            # train
            # for model_name, model in models.items():
            #     model.train()
            registration_model.train()
            LMA_model.train()            
            epoch_training_classification_correct_pred_num = 0
            epoch_training_data_num = 0
            epoch_training_sector_num = 0
            
            # initialize epoch loss dict
            epoch_loss_dict = {}

            for train_batch_idx, train_batch in enumerate(train_loader):
                # get data
                src, tar = train_batch['source_img'], train_batch['target_img']
                src = src.to(used_device)
                tar = tar.to(used_device)

                # forward
                reg_pred_dict = registration_model(src, tar)

                # compute registration loss
                # registration_loss = self.compute_training_loss(reg_pred_dict, registration_model, src, tar)

                # Merge the displacement fields of the same slice as a video / image sequence
                # Remove the redundant TOS for the same slice                 
                # pred_displacement_fields, TOS_curves, sector_LMA_labels, LMA_labels, batch_slice_full_ids = merge_data_of_same_slice_from_batch(train_batch, reg_pred_dict, n_frames_to_use_for_regression, used_device)
                train_slice_merged_data = merge_data_of_same_slice_from_batch(train_batch, reg_pred_dict, n_frames_to_use_for_regression, used_device)

                # compute regression loss
                train_LMA_pred = LMA_model(train_slice_merged_data['pred_displacement_fields'])
                
                # update the prediction dict and the targe dict for the loss calculator
                joint_pred_dict = {
                    'deformed_source': reg_pred_dict['deformed_source'],
                    'velocity': reg_pred_dict['velocity'],
                    'momentum': reg_pred_dict['momentum'],
                    'displacement': reg_pred_dict['displacement'],
                    # 'LMA_pred': LMA_pred,
                }
                joint_pred_dict.update(train_LMA_pred)
                joint_target_dict = {
                    'registration_target': tar,
                    'TOS': train_slice_merged_data['TOS'],
                    'slice_LMA_label': train_slice_merged_data['slice_LMA_label'],
                    'sector_LMA_labels': train_slice_merged_data['sector_LMA_labels'],
                    'DENSE_displacement': torch.cat((train_batch['displacement_field_X'], train_batch['displacement_field_Y']), dim=1).to(used_device) if use_disp_supervision else None,
                }
                train_total_loss, losses_values_dict = self.loss_calculator(joint_pred_dict, joint_target_dict)

                # backward
                registration_optimizer.zero_grad()
                LMA_optimizer.zero_grad()
                train_total_loss.backward()
                registration_optimizer.step()
                LMA_optimizer.step()

                # update progress bar
                progress_bar.set_description(f'Epoch {epoch} | Train Loss {train_total_loss.item():.3e}')

                # update the epoch loss dict
                for loss_name, loss_value in losses_values_dict.items():
                    record_name = f'training/{loss_name}'
                    if record_name not in epoch_loss_dict.keys():
                        epoch_loss_dict[record_name] = loss_value
                    else:
                        epoch_loss_dict[record_name] += loss_value

                # if LMA task is classification, 
                # record the number of data and correct predictions 
                # to compute the accuracy of the current epoch
                if LMA_task == 'LMA_slice_classification':
                    train_batch_size = train_LMA_pred['slice_LMA_label'].shape[0]
                    _, LMA_pred_class = torch.max(train_LMA_pred['slice_LMA_label'], dim=1)
                    correct_pred_num = torch.sum(LMA_pred_class == train_slice_merged_data['slice_LMA_labels'])

                    epoch_training_data_num += train_batch_size
                    epoch_training_classification_correct_pred_num += correct_pred_num
                elif LMA_task == 'LMA_sector_classification':
                    # the shape of LMA_pred should be (batch_size, n_classes=2, n_sectors=128)
                    # and the ground truth LMA_labels should be (batch_size, n_sectors=128)
                    train_batch_size = train_LMA_pred['sector_LMA_labels'].shape[0]
                    _, LMA_pred_class = torch.max(train_LMA_pred['sector_LMA_labels'], dim=1)
                    correct_pred_num = torch.sum(LMA_pred_class == train_slice_merged_data['sector_LMA_labels'])

                    epoch_training_sector_num += train_batch_size * train_LMA_pred['sector_LMA_labels'].shape[2]
                    epoch_training_classification_correct_pred_num += correct_pred_num
                elif LMA_task == 'TOS_regression':
                    train_batch_size = train_LMA_pred['TOS'].shape[0]
                else:
                    raise NotImplementedError(f'LMA task {LMA_task} not implemented')

            
            # compute epoch training accuracy
            if LMA_task == 'LMA_slice_classification':
                epoch_training_classification_accuracy = epoch_training_classification_correct_pred_num / epoch_training_data_num
                epoch_loss_dict['training/LMA_accuracy'] = epoch_training_classification_accuracy
            elif LMA_task == 'LMA_sector_classification':
                epoch_training_classification_accuracy = epoch_training_classification_correct_pred_num / epoch_training_sector_num
                epoch_loss_dict['training/LMA_accuracy'] = epoch_training_classification_accuracy
            
            # update learning rate
            registration_lr_scheduler.step()
            LMA_lr_scheduler.step()

            # validate
            registration_model.eval()
            LMA_model.eval()
            epoch_total_val_loss = 0
            epoch_total_val_loss_dict = {}
            for loss_term_name in self.loss_calculator.losses.keys():
                epoch_total_val_loss_dict[loss_term_name] = 0
            epoch_total_val_loss_dict['total_loss'] = 0
            epoch_loss_dict['validation/total_loss'] = 0
            epoch_val_classification_correct_pred_num = 0
            epoch_val_data_num = 0
            epoch_val_sector_num = 0
            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_loader):
                    # get data
                    src, tar = val_batch['source_img'], val_batch['target_img']
                    src = src.to(used_device)
                    tar = tar.to(used_device)

                    # forward
                    pred_dict = registration_model(src, tar)

                    # compute loss
                    # val_registration_loss = self.compute_training_loss(pred_dict, registration_model, src, tar)

                    # Merge the displacement fields of the same slice as a video / image sequence
                    # Remove the redundant TOS for the same slice                 
                    # pred_displacement_fields, TOS_curves, sector_LMA_labels, LMA_labels, batch_slice_full_ids = merge_data_of_same_slice_from_batch(val_batch, pred_dict, n_frames_to_use_for_regression, used_device)
                    val_slice_merged_data = merge_data_of_same_slice_from_batch(val_batch, pred_dict, n_frames_to_use_for_regression, used_device)

                    # compute regression loss
                    val_LMA_pred = LMA_model(val_slice_merged_data['pred_displacement_fields'])


                    joint_val_pred_dict = {
                        'deformed_source': pred_dict['deformed_source'],
                        'velocity': pred_dict['velocity'],
                        'momentum': pred_dict['momentum'],
                        'displacement': pred_dict['displacement'],
                        # 'LMA_pred': LMA_pred,
                    }
                    joint_val_pred_dict.update(val_LMA_pred)
                    joint_val_target_dict = {
                        'registration_target': tar,
                        'TOS': val_slice_merged_data['TOS'],
                        'sector_LMA_labels': val_slice_merged_data['sector_LMA_labels'],
                        'slice_LMA_label': val_slice_merged_data['slice_LMA_label'],
                        'DENSE_displacement': torch.cat((val_batch['displacement_field_X'], val_batch['displacement_field_Y']), dim=1).to(used_device) if use_disp_supervision else None,
                    }
                    val_total_loss, val_losses_values_dict = self.loss_calculator(joint_val_pred_dict, joint_val_target_dict)

                    # update the val loss dict
                    for loss_name, loss_value in val_losses_values_dict.items():
                        record_name = f'validation/{loss_name}'
                        if record_name not in epoch_loss_dict.keys():
                            epoch_loss_dict[record_name] = loss_value
                        else:
                            epoch_loss_dict[record_name] += loss_value
                    # val_total_loss = val_registration_loss + 1e-1 * LMA_loss

                    # update progress bar
                    progress_bar.set_description(
                        f'Epoch {epoch} | Train Loss {val_total_loss.item():.3e} | Val Loss {val_total_loss.item():.3e}')
                    epoch_total_val_loss += val_total_loss.item()

                    # if LMA_task == 'LMA_classification':
                    #     batch_size = val_LMA_pred.shape[0]
                    #     _, LMA_pred_class = torch.max(val_LMA_pred, dim=1)
                    #     correct_pred_num = torch.sum(LMA_pred_class == val_slice_merged_data['LMA_labels'])

                    #     epoch_val_data_num += batch_size
                    #     epoch_val_classification_correct_pred_num += correct_pred_num

                    if LMA_task == 'LMA_slice_classification':
                        val_batch_size = val_LMA_pred['slice_LMA_label'].shape[0]
                        _, LMA_pred_class = torch.max(val_LMA_pred['slice_LMA_label'], dim=1)
                        correct_pred_num = torch.sum(LMA_pred_class == val_slice_merged_data['slice_LMA_labels'])

                        epoch_val_data_num += val_batch_size
                        epoch_val_classification_correct_pred_num += correct_pred_num
                    elif LMA_task == 'LMA_sector_classification':
                        # the shape of LMA_pred should be (batch_size, n_classes=2, n_sectors=128)
                        # and the ground truth LMA_labels should be (batch_size, n_sectors=128)
                        val_batch_size = val_LMA_pred['sector_LMA_labels'].shape[0]
                        _, LMA_pred_class = torch.max(val_LMA_pred['sector_LMA_labels'], dim=1)
                        correct_pred_num = torch.sum(LMA_pred_class == val_slice_merged_data['sector_LMA_labels'])

                        epoch_val_sector_num += val_batch_size * val_LMA_pred['sector_LMA_labels'].shape[2]
                        epoch_val_classification_correct_pred_num += correct_pred_num
                    elif LMA_task == 'TOS_regression':
                        val_batch_size = val_LMA_pred['TOS'].shape[0]
                    else:
                        raise NotImplementedError(f'LMA task {LMA_task} not implemented')
            
            
            # epoch_loss_dict['validation/total_loss'] = epoch_total_val_loss
            # add validation losses to epoch_loss_dict
            for loss_name, loss_value in val_losses_values_dict.items():
                record_name = f'validation/{loss_name}'
                if record_name not in epoch_loss_dict.keys():
                    epoch_loss_dict[record_name] = loss_value
                else:
                    epoch_loss_dict[record_name] += loss_value
            if LMA_task == 'LMA_slice_classification':
                epoch_val_classification_accuracy = epoch_val_classification_correct_pred_num / epoch_val_data_num
                epoch_loss_dict['validation/LMA_accuracy'] = epoch_val_classification_accuracy
            elif LMA_task == 'LMA_sector_classification':
                epoch_val_classification_accuracy = epoch_val_classification_correct_pred_num / epoch_val_sector_num
                epoch_loss_dict['validation/LMA_accuracy'] = epoch_val_classification_accuracy
            
            if use_wandb:
                wandb_experiment.log(epoch_loss_dict, step=epoch)
                # print('epoch_loss_dict at step', epoch)
                # visualize the LMA results if the interval is reached
                if wandb_visualize_interval > 0 and epoch % wandb_visualize_interval == 0:
                    train_fig, train_axs = self.visualize_LMA_batch(train_batch, train_LMA_pred, 5, np.arange(min(5, train_batch_size)))
                    val_fig, val_axs = self.visualize_LMA_batch(val_batch, val_LMA_pred, 5, np.arange(min(5, val_batch_size)))
                    # wandb_experiment.log({
                    #     'training/LMA_visualization': train_fig,
                    #     'validation/LMA_visualization': val_fig,
                    # }, step=epoch)
                    wandb_experiment.log({
                        'training/LMA_visualization': wandb.Image(train_fig),
                        'validation/LMA_visualization': wandb.Image(val_fig),
                    }, step=epoch)
                    # print('visualize at step', epoch)
            
            if use_tensorboard:
                for loss_name, loss_value in epoch_loss_dict.items():
                    writer.add_scalar(loss_name, loss_value, epoch)
                # writer.add_scalar('validation?total_loss', epoch_total_val_loss, epoch)
                # writer.close()
                writer.flush()
            
            
            if early_stop:
                # save best model
                # if LMA_task == 'LMA_slice_classification':
                #     epoch_val_performance = epoch_val_classification_accuracy
                # elif LMA_task == 'LMA_sector_classification':
                #     epoch_val_performance = epoch_val_classification_accuracy
                # elif LMA_task == 'TOS_regression':
                #     epoch_val_performance = -epoch_total_val_loss
                # else:
                #     raise NotImplementedError(f'LMA task {LMA_task} not implemented')
                if epoch_total_val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = epoch_total_val_loss
                    best_registration_model = copy.deepcopy(registration_model)
                    best_LMA_model = copy.deepcopy(LMA_model)
                    best_epoch_loss_dict = copy.deepcopy(epoch_loss_dict)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f'Epochs without improvement: {epochs_without_improvement} / {epochs_without_improvement_tolerance}')

                if use_wandb:
                    best_epoch_loss_dict_wandb = {}
                    for key, value in best_epoch_loss_dict.items():
                        # append 'best-' to the second level of the key
                        best_key = '/'.join(key.split('/')[:1] + ['best-' + key.split('/')[1]])
                        best_epoch_loss_dict_wandb[best_key] = value
                    best_epoch_loss_dict_wandb['best_epoch'] = best_epoch
                    wandb_experiment.log(best_epoch_loss_dict_wandb, step=epoch)
                    # wandb_experiment.log({
                    #     'best_val_loss': best_val_loss,
                    #     'best_epoch': best_epoch,
                    # }, step=epoch)
                    # print('best_val_loss at step', epoch)

                # early stopping
                if epochs_without_improvement >= epochs_without_improvement_tolerance:
                    print(f'Early stopping at epoch {epoch}')
                    return {
                        'epoch': best_epoch,
                        'epoch_loss_dict': best_epoch_loss_dict,
                        'cine_registraion': best_registration_model,
                        'LMA': best_LMA_model,
                    }, wandb_experiment

            
            
        if use_tensorboard:
            writer.close()

        if early_stop:
            registration_model = best_registration_model
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
        # save trained models
        # save_dir = Path(used_full_config['others']['save_dir'])
        self.save_trained_models(
            save_dir=exp_save_dir, 
            loss_dict = epoch_loss_dict, 
            full_config=used_full_config, 
            models = {
                'cine_registraion': registration_model,
                'LMA': LMA_model,
            })

        exp_dict = {
            'epoch': epoch,
            'epoch_loss_dict': best_epoch_loss_dict,
            'cine_registraion': registration_model,
            'LMA': LMA_model,
        }

        return exp_dict, wandb_experiment
    
    # function inference: alias to test
    def inference(self, model, datasets, trainer_config=None, full_config=None, device=None):
        return self.test(model, datasets, trainer_config, full_config, device)
    
    def test(self, models, datasets, trainer_config=None, full_config=None, device=None, wandb_experiment=None):
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # build dataloaders
        test_loader = SliceDataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=True, num_workers=1)

        # other parameters
        # n_frames_to_use_for_regression = full_config['networks']['TOS_regression']['input_frame_num']
        n_frames_to_use_for_regression = 48
        
        # get models
        LMA_task = trainer_config.get('LMA_task', 'TOS_regression')
        registration_model = models['cine_registraion']
        LMA_model = models['LMA']
        
        registration_model.eval()
        LMA_model.eval()

        # build loss function
        self.regression_loss_fn = torch.nn.MSELoss()
        
        # print the device of model
        # print(f'Model is on {next(regression_model.parameters()).device}')
        # build progress bar
        test_preds = []
        # progress_bar = tqdm(range(used_train_config['epochs']))
        # for epoch in progress_bar:
            # train
            # for model_name, model in models.items():
            #     model.train()
            # registration_model.train()
            # LMA_model.train()
        test_performance_dict = {}
        # if LMA_task == 'TOS_regression':
        test_classification_correct_pred_num = 0
        test_data_num = 0
        test_sector_num = 0
        # test_TOS_MSE = 0
        test_sector_error = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # get data
                src, tar = batch['source_img'], batch['target_img']
                src = src.to(used_device)
                tar = tar.to(used_device)

                # forward
                reg_pred_dict = registration_model(src, tar)

                # compute registration loss
                registration_loss = self.compute_training_loss(reg_pred_dict, registration_model, src, tar)
                
                # pred_displacement_fields, TOS_curves, sector_LMA_labels, LMA_labels, batch_slice_full_ids = merge_data_of_same_slice_from_batch(batch, reg_pred_dict, n_frames_to_use_for_regression, used_device)
                val_slice_merged_data = merge_data_of_same_slice_from_batch(batch, reg_pred_dict, n_frames_to_use_for_regression, used_device)

                # compute regression loss
                test_LMA_pred = LMA_model(val_slice_merged_data['pred_displacement_fields'])
                # regression_loss = self.regression_loss_fn(test_LMA_pred, TOS_curves)
                

                # train_total_loss = registration_loss + 1e-2 * regression_loss
                test_total_loss = registration_loss

                # update progress bar
                # progress_bar.set_description(f'Epoch {epoch} | Train Loss {test_total_loss.item():.3e}')
                # compute batch performance
                if LMA_task == 'LMA_slice_classification':
                    test_batch_size = test_LMA_pred['slice_LMA_label'].shape[0]
                    _, LMA_pred_class = torch.max(test_LMA_pred['slice_LMA_label'], dim=1)
                    correct_pred_num = torch.sum(LMA_pred_class == val_slice_merged_data['slice_LMA_labels'])

                    test_data_num += test_batch_size
                    test_classification_correct_pred_num += correct_pred_num
                elif LMA_task == 'LMA_sector_classification':
                    # the shape of LMA_pred should be (batch_size, n_classes=2, n_sectors=128)
                    # and the ground truth LMA_labels should be (batch_size, n_sectors=128)
                    test_batch_size = test_LMA_pred['sector_LMA_labels'].shape[0]
                    _, LMA_pred_class = torch.max(test_LMA_pred['sector_LMA_labels'], dim=1)
                    correct_pred_num = torch.sum(LMA_pred_class == val_slice_merged_data['sector_LMA_labels'])

                    test_sector_num += test_batch_size * test_LMA_pred['sector_LMA_labels'].shape[2]
                    test_classification_correct_pred_num += correct_pred_num
                elif LMA_task == 'TOS_regression':
                    test_TOS_pred = test_LMA_pred['TOS']
                    test_batch_size, test_TOS_n_sectors = test_TOS_pred.shape # (N, 126)
                    test_TOS_GT = val_slice_merged_data['TOS']
                    # test_TOS_MSE += nn.MSELoss()(test_TOS_pred, test_TOS_GT).item() * test_batch_size
                    test_sector_num += test_batch_size * test_TOS_n_sectors
                    test_sector_error += torch.sum(torch.abs(test_TOS_pred - test_TOS_GT)).item() 
                    # test_sector_error += 17 * 126 * 5 
                else:
                    raise NotImplementedError(f'LMA task {LMA_task} not implemented')

                # break the batch into individual images and append to test_preds
                for i in range(src.shape[0]):
                    test_pred_dict = {}
                    # copy all key-value from batch to test_pred_dict if
                    # (1) the value is not a torch.Tensor or np.ndarray, or
                    # (2) the value is a torch.Tensor or np.ndarray and the shape of the value is the same as the shape of the batch
                    for k, v in batch.items():
                        if not isinstance(v, (torch.Tensor, np.ndarray)):
                            test_pred_dict[k] = v[i]
                        elif isinstance(v, torch.Tensor) and v.shape == batch[k].shape:
                            test_pred_dict[k] = v[i].cpu().numpy()
                    
                    # Add the components in pred_dict to test_pred_dict
                    for k, v in reg_pred_dict.items():
                        test_pred_dict[k] = v[i].cpu().numpy()

                    # Add the TOS prediction and strain matrix from the batch
                    slice_idx_of_current_pair = val_slice_merged_data['batch_slice_full_ids'].index(batch['slice_full_id'][i])

                    # copy all key-value from test_LMA_pred to test_pred_dict
                    for k, v in test_LMA_pred.items():
                        test_pred_dict[k+'_pred'] = v[slice_idx_of_current_pair].cpu().numpy()
                    # test_pred_dict['LMA_pred'] = test_LMA_pred[slice_idx_of_current_pair].cpu().numpy()
                    
                    
                    # Add the test_pred_dict to test_preds
                    test_preds.append(test_pred_dict)
        
        # compute the overall performance
        if LMA_task == 'LMA_slice_classification':
            test_classification_accuracy = test_classification_correct_pred_num / test_data_num
            test_performance_dict['test/LMA_accuracy'] = test_classification_accuracy
        elif LMA_task == 'LMA_sector_classification':
            test_classification_accuracy = test_classification_correct_pred_num / test_sector_num
            # if test_classification_accuracy is pytorch tensor, convert it to numpy
            if isinstance(test_classification_accuracy, torch.Tensor):
                test_classification_accuracy = test_classification_accuracy.item()
            test_performance_dict['test/LMA_accuracy'] = test_classification_accuracy
        elif LMA_task == 'TOS_regression':
            # test_classification_accuracy = None
            test_overall_sector_error = test_sector_error / test_sector_num
            test_performance_dict['test/sector_error'] = test_overall_sector_error
        
        if wandb_experiment is not None:
            wandb_experiment.log(test_performance_dict)
        print('test_performance_dict: ', test_performance_dict)
        return test_preds, test_performance_dict, wandb_experiment

    def test_registration_only(self, models, datasets, trainer_config=None, full_config=None, device=None):
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # build dataloaders
        test_loader = DataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0)

        # build loss function
        self.loss_fn = torch.nn.MSELoss()

        # test
        test_preds = []
        # model.eval()
        registration_model = models['cine_registraion']
        regression_model = models['TOS_regression']
        registration_model.eval()
        regression_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # get data
                src, tar = batch['source_img'], batch['target_img']
                src = src.to(used_device)
                tar = tar.to(used_device)

                # forward
                pred_dict = registration_model(src, tar)

                # test visualization
                # _, _ = self.visualize_LMA_batch(batch, pred_dict)

                # compute loss
                test_loss = self.compute_training_loss(pred_dict, registration_model, src, tar)

                # update progress bar
                print(f'Test Loss {test_loss.item():.3e}')

                # break the batch into individual images and append to test_preds
                for i in range(src.shape[0]):
                    test_pred_dict = {}
                    # copy all key-value from batch to test_pred_dict if
                    # (1) the value is not a torch.Tensor or np.ndarray, or
                    # (2) the value is a torch.Tensor or np.ndarray and the shape of the value is the same as the shape of the batch
                    for k, v in batch.items():
                        if not isinstance(v, (torch.Tensor, np.ndarray)):
                            test_pred_dict[k] = v[i]
                        elif isinstance(v, torch.Tensor) and v.shape == batch[k].shape:
                            test_pred_dict[k] = v[i].cpu().numpy()
                    
                    # Add the components in pred_dict to test_pred_dict
                    for k, v in pred_dict.items():
                        test_pred_dict[k] = v[i].cpu().numpy()
                    
                    
                    # Add the test_pred_dict to test_preds
                    test_preds.append(test_pred_dict)

        return test_preds
    
    def compute_training_loss(self, pred_dict, model, src, tar):
        reg_loss_weight = 1
        u = pred_dict['displacement']
        v = pred_dict['velocity']
        m = pred_dict['momentum']
        Sdef = pred_dict['deformed_source']

        # compute loss
        loss_fn = torch.nn.MSELoss()
        loss1 = loss_fn(tar, Sdef)
        loss2 = (v*m).sum() / (src.numel())
        loss_regis = 0.5 * loss1/(model.sigma*model.sigma) + reg_loss_weight * loss2

        return loss_regis
    
    
    def visualize_pred_registraion(self, preds, n_vis=5, vis_indices=None, save_plots=False, save_dir=None):
        # visualize the data in preds, which is the output of self.test() or self.inference()

        # check n_vis random sample from preds
        # make (5, n_vis) subplot using matplotlib
        # where each column shows 
        # (1) source image, (2) deformed source image, (3) target image
        # (4) abs(source - target), (5) abs(deformed source - target)
        if vis_indices is None:
            vis_indices = np.random.randint(0, len(preds), n_vis)
        else:
            n_vis = len(vis_indices)
        fig, axs = plt.subplots(5, n_vis, figsize=(n_vis*3, 15))
        for plot_idx, vis_test_idx in enumerate(vis_indices):
            curr_vis_test_pred = preds[vis_test_idx]
            # source image
            axs[0, plot_idx].imshow(curr_vis_test_pred['source_img'][0, :, :], cmap='gray')
            axs[0, plot_idx].set_title(f"{curr_vis_test_pred['subject_id']}-{curr_vis_test_pred['slice_idx']}")

            # deformed source image
            axs[1, plot_idx].imshow(curr_vis_test_pred['deformed_source'][0, :, :], cmap='gray')

            # target image
            axs[2, plot_idx].imshow(curr_vis_test_pred['target_img'][0, :, :], cmap='gray')

            # abs(source - target)
            source_target_diff = np.abs(curr_vis_test_pred['source_img'][0, :, :] - curr_vis_test_pred['target_img'][0, :, :])
            axs[3, plot_idx].imshow(source_target_diff, cmap='gray')
            axs[3, plot_idx].set_title(f"diff = {int(source_target_diff.sum())}")

            # abs(deformed source - target)
            deformed_source_target_diff = np.abs(curr_vis_test_pred['deformed_source'][0, :, :] - curr_vis_test_pred['target_img'][0, :, :])
            axs[4, plot_idx].imshow(deformed_source_target_diff, cmap='gray')
            axs[4, plot_idx].set_title(f"diff = {int(deformed_source_target_diff.sum())}")

            # hide x ticks and y ticks
            for i in range(5):
                axs[i, plot_idx].set_xticks([])
                axs[i, plot_idx].set_yticks([])

            # add y label for the first column
            if plot_idx == 0:
                axs[0, plot_idx].set_ylabel('source')
                axs[1, plot_idx].set_ylabel('deformed source')
                axs[2, plot_idx].set_ylabel('target')
                axs[3, plot_idx].set_ylabel('abs(source - target)')
                axs[4, plot_idx].set_ylabel('abs(deformed source - target)')
        
        if save_plots:
            if save_dir is None:
                save_dir = Path(self.full_config['others']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / 'registration_visualization.png')

        # plot the strain matrices and TOS curves
        # fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        # for plot_idx, vis_test_idx in enumerate(vis_indices):

        # axs[0].imshow(np.random.randn(128, 128), cmap='gray')

    def visualize_LMA_batch(self, batch, preds, n_vis=5, vis_indices=None):
        if vis_indices is None:
            vis_indices = np.random.randint(0, len(preds), n_vis)
        else:
            n_vis = len(vis_indices)
        print(f'Visualizing {n_vis} samples')
        fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        if n_vis == 1:
            axs = [axs]
        if self.LMA_task in ['TOS_regression']:
            for plot_idx, vis_data_idx in enumerate(vis_indices):
                axs[plot_idx].pcolor(batch['strain_mat'][vis_data_idx][0], cmap='jet', vmin=-0.3, vmax=0.3)
                axs[plot_idx].plot(batch['TOS'][vis_data_idx].detach().cpu().numpy()/17+1, np.arange(126), color='black', label='GT')
                axs[plot_idx].plot(preds['TOS'][vis_data_idx].detach().cpu().numpy()/17+1, np.arange(126), color='red', linestyle='--', label='pred')
                axs[plot_idx].set_xlim([0, 50])                
                axs[plot_idx].legend()
        elif self.LMA_task in ['LMA_slice_classification']:
            for plot_idx, vis_data_idx in enumerate(vis_indices):
                axs[plot_idx].pcolor(batch['strain_mat'][vis_data_idx][0], cmap='jet', vmin=-0.3, vmax=0.3)
                # axs[plot_idx].plot(batch['slice_LMA_label'][vis_data_idx].detach().cpu().numpy()/17+1, np.arange(126), color='black')
                # axs[plot_idx].plot(preds['slice_LMA_label'][vis_data_idx].detach().cpu().numpy()/17+1, np.arange(126), color='red', linestyle='--')
                axs[plot_idx].set_xlim([0, 50])
        elif self.LMA_task in ['LMA_sector_classification']:
            for plot_idx, vis_data_idx in enumerate(vis_indices):
                axs[plot_idx].pcolor(batch['strain_mat'][vis_data_idx][0], cmap='jet', vmin=-0.3, vmax=0.3)
                axs[plot_idx].plot(batch['sector_LMA_labels'][vis_data_idx].detach().cpu().numpy()*10+1, np.arange(126), color='black', label='GT')
                axs[plot_idx].plot(preds['sector_LMA_labels'][vis_data_idx].detach().cpu().numpy().argmax(axis=0)*10+1, np.arange(126), color='red', linestyle='--', label='pred')
                axs[plot_idx].set_xlim([0, 50])
                axs[plot_idx].legend()
        else:
            raise ValueError(f'Unknown LMA task: {self.LMA_task}')
        return fig, axs
    
    def visualize_pred_regression(self, preds, n_vis=5, vis_indices=None, save_plots=False, save_dir=None, save_name=''):
        # plot the strain matrices and TOS curves
        if vis_indices is not None:
            n_vis = len(vis_indices)
        else:
            vis_indices = np.random.randint(0, len(preds), n_vis)
        vis_indices = np.random.randint(0, len(preds), n_vis)
        fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        for plot_idx, vis_test_idx in enumerate(vis_indices):
            axs[plot_idx].pcolor(preds[vis_test_idx]['strain_mat'][0], cmap='jet', vmin=-0.3, vmax=0.3)
            axs[plot_idx].plot(preds[vis_test_idx]['TOS']/17+1, np.arange(126), color='black')
            axs[plot_idx].plot(preds[vis_test_idx]['TOS_pred']/17+1, np.arange(126), color='red', linestyle='--')
            axs[plot_idx].set_xlim([0, 50])
        if save_plots:
            if save_dir is None:
                save_dir = Path(self.full_config['others']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / save_name)
        return fig, axs

    def visualize_pred_sector_classification(self, preds, n_vis=5, vis_indices=None, save_plots=False, save_dir=None, save_name=''):
        # plot the strain matrices and sector classification results
        if vis_indices is not None:
            n_vis = len(vis_indices)
        else:
            vis_indices = np.random.randint(0, len(preds), n_vis)
        fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        for plot_idx, vis_test_idx in enumerate(vis_indices):
            axs[plot_idx].pcolor(preds[vis_test_idx]['strain_mat'][0], cmap='jet', vmin=-0.3, vmax=0.3)
            axs[plot_idx].plot(preds[vis_test_idx]['sector_LMA_labels']*10+1, np.arange(126), color='black', linewidth=5)
            axs[plot_idx].plot(preds[vis_test_idx]['sector_LMA_labels_pred'].argmax(axis=0)*10+1, np.arange(126), color='red', linestyle='--', linewidth=5)
            axs[plot_idx].set_xlim([0, 50])
        if save_plots:
            if save_dir is None:
                save_dir = Path(self.full_config['others']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / save_name)
        return fig, axs

    def save_model(self, model, config=None):        
        if config is not None:
            used_config = config
        else:
            used_config = self.full_config['saving']
        
        if used_config['method'] == 'jit':
            # save the model and parameters separately
            # save the model using ScriptModule (jit)
            # save the parameters using torch.save
            # the model can be loaded using torch.jit.load
            # the parameters can be loaded using torch.load
            save_path = Path(used_config['path'])
            model_path = save_path / 'model.pt'
            param_path = save_path / 'param.pt'
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, model_path)
            torch.save(model.state_dict(), param_path)
        elif used_config['method'] == 'onnx':
            # ONNX export failed: Couldn't export operator aten::size
            # https://github.com/onnx/tutorials/issues/63
            # => operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            # ScalarType UNKNOWN_SCALAR is an unexpected tensor scalar type
            # https://discuss.pytorch.org/t/runtimeerror-unexpected-tensor-scalar-type/124031/5 => complex numbers
            dummy_input_src = torch.randn(1, 1, 128, 128).to(next(model.parameters()).device)
            dummy_input_tar = torch.randn(1, 1, 128, 128).to(next(model.parameters()).device)

            # put both dummy_input_src and dummy_input_tar on the same device as the model
            # dummy_input_src = dummy_input_src.to(next(model.parameters()).device)
            # dummy_input_tar = dummy_input_tar.to(next(model.parameters()).device)

            save_path = Path(used_config['path'])
            model_path = save_path / 'model.onnx'
            torch.onnx.export(model, (dummy_input_src, dummy_input_tar), model_path, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        elif used_config['method'] == 'state_dict':
            # save state_dict only
            save_path = Path(used_config['path'])
            model_path = save_path / 'model.pt'
            torch.save(model.state_dict(), model_path)
        elif used_config['method'] == 'model_zip_state_dict_pt':
        # elif True:
            # save the model source code by calling collect_and_zip_python_files
            # save the model parameters together using torch.save
            save_path = Path(used_config['path'])
            # save the model source code
            model_src_code_zip_path = save_path / 'model_src.zip'
            collect_and_zip_python_files(used_config['model_src_code_dirs'], model_src_code_zip_path)
            model_paras_path = save_path / 'model.pt'
            torch.save(model.state_dict(), model_paras_path)

        else:
            raise NotImplementedError(f"Saving method {used_config['method']} is not implemented")
        
    def load_model(self, model_parameter_filename, model_definition_filename=None, config=None):
        if config is not None:
            used_config = config
        else:
            used_config = self.full_config['saving']

        if used_config['method'] == 'jit':
            # load the model and parameters separately
            # load the model using torch.jit.load
            # load the parameters using torch.load
            # the model can be saved using torch.jit.save
            # the parameters can be saved using torch.save
            model = torch.jit.load(model_definition_filename)
            model.load_state_dict(torch.load(model_parameter_filename))
        else:
            raise NotImplementedError(f"Saving method {used_config['method']} is not implemented")

    def save_trained_models(self, save_dir, loss_dict, full_config, models):
        import json

        exp_results_save_path = Path(save_dir)
        
        from datetime import date
        if exp_results_save_path == '':
            today = date.today()
            exp_name = full_config['info']['experiment_name']
            exp_fullname = str(date.today()) + '-' + exp_name
            exp_results_save_path = Path('.', 'exp_results', exp_fullname)
        exp_results_save_path.mkdir(exist_ok=True, parents=True)
        training_record_save_filename = Path(exp_results_save_path, f"performance.json")

        # dump config
        config_save_filename = Path(exp_results_save_path, f"config.json")
        with open(str(config_save_filename), "w") as outfile:
            json.dump(full_config, outfile)
        
        # training_record = {
        #     'val_accs': val_accs, 
        #     'train_accs': train_accs, 
        #     'best_val_acc': best_val_acc, 
        #     'epochs': epoch, 
        #     'early_stop_triggered': early_stop_triggered, 
        #     'best_model_saved': best_model_saved,
        #     'network_structure': str(net),
        #     }
        training_record = loss_dict
        
        with open(str(training_record_save_filename), "w") as outfile:
            json.dump(training_record, outfile)
        
        # save model if needed
        save_final_model = full_config['saving'].get('save_final_model', False) or full_config['saving'].get('save_final_model', False)
        if save_final_model is True:
            registration_model_save_filename = Path(exp_results_save_path, "model-registration.pth")
            torch.save(models['cine_registraion'].state_dict(), registration_model_save_filename)
            LMA_model_save_filename = Path(exp_results_save_path, "model-LMA.pth")
            torch.save(models['LMA'].state_dict(), LMA_model_save_filename)
            print(f'Model saved to {exp_results_save_path}')
        #     # torch.save(best_model_weights, model_save_filename)
        #     torch.save(best_model.state_dict(), model_save_filename)
        #     if verbose:
        #         print(f'Model saved to {model_save_filename}')


import zipfile
from pathlib import Path

def collect_and_zip_python_files(src_dirs, zip_name):
    # Create a new Zip file (or overwrite the existing one)
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for src_dir in src_dirs:
            src_dir_path = Path(src_dir)
            # Walk through the source directory
            for file_path in src_dir_path.rglob('*.py'):
                # Add the file to the Zip file
                # The arcname parameter avoids storing the full path in the Zip file
                zipf.write(file_path, arcname=file_path.relative_to(src_dir_path))

# List of directories containing the .py files related to the model definition
src_dirs = ['path_to_model_definition_dir1', 'path_to_model_definition_dir2']

# Name of the Zip file to create
zip_name = 'model_definition.zip'

# Collect and zip the .py files
collect_and_zip_python_files(src_dirs, zip_name)
