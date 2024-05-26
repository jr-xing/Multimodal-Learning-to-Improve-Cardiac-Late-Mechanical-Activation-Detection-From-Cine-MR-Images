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
    LMA_labels = []
    batch_slice_full_ids = list(set(batch['slice_full_id']))
    for slice_full_id in batch_slice_full_ids:                    
        # indices of the pairs with of the current slice
        slice_pair_indices = [idx for idx, id in enumerate(batch['slice_full_id']) if id == slice_full_id]
        # collect the displacement fields for the same slice
        slice_displacement_fields = [reg_pred_dict['displacement'][idx] for idx in slice_pair_indices] # list of displacement fields with shape (2, H, W)
        slice_displacement_fields = torch.stack(slice_displacement_fields, dim=0) # shape (n_pairs, 2, H, W)
        slice_displacement_fields = slice_displacement_fields.permute(1, 0, 2, 3) # shape (2, n_pairs, H, W)
        pred_displacement_fields.append(slice_displacement_fields[:, :n_frames_to_use_for_regression, ...])

        # collect the TOS curves for the same slice
        # check whether all pairs of the current slice have the same TOS curve
        # if not, raise an error
        # if yes, add the TOS curve to TOS_curves
        slice_TOS_curves = batch['TOS'][torch.tensor(slice_pair_indices)]
        slice_LMA_labels = batch['LMA'][torch.tensor(slice_pair_indices)]
        # slice_TOS_curves_identical = check_tensor_rows_identity(slice_TOS_curves)
        # if not slice_TOS_curves_identical:
        #     raise ValueError(f'TOS curves for the same slice {slice_full_id} are not identical')
        # else:
        TOS_curves.append(slice_TOS_curves[0])
        LMA_labels.append(slice_LMA_labels[0])
    pred_displacement_fields = torch.stack(pred_displacement_fields, dim=0) # shape (n_slices, 2, n_frames_to_use, H, W)
    TOS_curves = torch.stack(TOS_curves, dim=0).to(used_device) # shape (n_slices, 126)
    LMA_labels = torch.concat(LMA_labels, dim=0).to(used_device) # shape (n_slices, )
    return pred_displacement_fields, TOS_curves, LMA_labels, batch_slice_full_ids

class JointRegistrationRegressionTrainer:
    def __init__(self, trainer_config, device=None, full_config=None):
        self.trainer_config = trainer_config
        self.full_config = full_config
        self.device = device if device is not None else torch.device('cpu')

    def train(self, models: dict, datasets: dict, trainer_config=None, full_config=None, device=None, use_tensorboard=False, tensorboard_log_dir=None, early_stop=True, use_wandb=False):
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # build dataloaders
        # train_loader = DataLoader(datasets['train'], batch_size=used_train_config['batch_size'], shuffle=True, num_workers=1)
        # val_loader = DataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=1)
        train_loader = SliceDataLoader(datasets['train'], batch_size=used_train_config['batch_size'], shuffle=True, num_workers=1)
        val_loader = SliceDataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=1)

        # other parameters
        # n_frames_to_use_for_regression = full_config['networks']['TOS_regression']['input_frame_num']
        n_frames_to_use_for_regression = 25
        
        # LMA model
        LMA_task = trainer_config.get('LMA_task', 'TOS_regression')
        registration_model = models['cine_registraion']
        if LMA_task == 'TOS_regression':
            LMA_model = models['TOS_regression']
            # self.LMA_loss_fn = torch.nn.MSELoss()
        elif LMA_task == 'LMA_classification':
            LMA_model = models['LMA_classification']
            # self.LMA_loss_fn = torch.nn.CrossEntropyLoss()

        # loss functions
        self.loss_calculator = LossCalculator(full_config['losses'], full_config)
        
        # build optimizer
        # optimizer = torch.optim.Adam(
        #     list(registration_model.parameters()) + list(LMA_model.parameters()), 
        #     lr=used_train_config['optimizers']['registration']['learning_rate'])
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
        # self.loss_fn = torch.nn.MSELoss()
        # self.regression_loss_fn = torch.nn.MSELoss()
        
        # print the device of model
        print(f'Model is on {next(LMA_model.parameters()).device}')
        # build progress bar
        progress_bar = tqdm(range(used_train_config['epochs']))

        # early stop parameters
        # early_stop = True
        if early_stop:
            best_registration_model = None
            best_regression_model = None
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            epochs_without_improvement_tolerance = 10            
        
        # Experiment tracking
        if use_tensorboard:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            writer = SummaryWriter(log_dir=Path(tensorboard_log_dir, f'{current_time}'))
        
        if use_wandb:
            import random
            import string
            exp_random_ID = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
            exp_name = full_config['info'].get('experiment_name', 'unnamed') + f'-{exp_random_ID}'
            exp_name_with_date = datetime.datetime.now().strftime('%Y-%m-%d')+ '-' + exp_name
            use_wandb = full_config['others'].get('use_wandb', False)
            # if Path(exp_name_with_date).exists():
            #     raise ValueError(f'Experiment {exp_name_with_date} already exists!')
            # else:
            #     wandb_path = Path(f'./exp_results/{exp_name_with_date}-wandb')
            #     if use_wandb:
            #         wandb_path.mkdir(parents=True, exist_ok=True)    
            wandb_path = Path(f"./exp_results/{datetime.datetime.now().strftime('%Y-%m')}/{exp_name_with_date}-wandb")
            wandb_path.mkdir(parents=True, exist_ok=True)
            
            wandb_experiment = wandb.init( 
                project = full_config['info'].get('project_name', 'trials'),
                entity = "jrxing", 
                save_code = True,
                name = full_config['info']['experiment_name'] if full_config['info'].get('use_experiment_name', False) else None,
                dir = wandb_path,
                resume = 'allow', 
                anonymous = 'must',
                mode='online' if use_wandb else 'disabled')

        for epoch in progress_bar:
            # train
            # for model_name, model in models.items():
            #     model.train()
            registration_model.train()
            LMA_model.train()
            # epoch_total_training_loss = 0
            # epoch_total_training_registration_reconstruction_loss = 0
            # epoch_total_training_registration_supervised_loss = 0
            # epoch_total_training_regression_loss = 0
            epoch_training_classification_correct_pred_num = 0
            epoch_training_data_num = 0
            # initialize epoch loss dict
            epoch_loss_dict = {}

            # epoch_loss_dict = {
            #     'training/total_loss': 0,
            #     'training/registration_reconstruction_loss': 0,
            #     'training/registration_supervised_loss': 0,
            #     'training/LMA_loss': 0,
            # }
            for batch_idx, batch in enumerate(train_loader):
                # get data
                src, tar = batch['source_img'], batch['target_img']
                src = src.to(used_device)
                tar = tar.to(used_device)

                # forward
                reg_pred_dict = registration_model(src, tar)

                # compute registration loss
                # registration_loss = self.compute_training_loss(reg_pred_dict, registration_model, src, tar)

                # Merge the displacement fields of the same slice as a video / image sequence
                # Remove the redundant TOS for the same slice                 
                pred_displacement_fields, TOS_curves, LMA_labels, batch_slice_full_ids = merge_data_of_same_slice_from_batch(batch, reg_pred_dict, n_frames_to_use_for_regression, used_device)

                # compute regression loss
                LMA_pred = LMA_model(pred_displacement_fields)
                # if LMA_task == 'TOS_regression':
                #     LMA_loss = self.LMA_loss_fn(LMA_pred, TOS_curves)
                # elif LMA_task == 'LMA_classification':
                #     LMA_loss = self.LMA_loss_fn(LMA_pred, LMA_labels)
                

                # train_total_loss = registration_loss + 1e-1 * LMA_loss
                # train_total_loss = registration_loss
                joint_pred_dict = {
                    'deformed_source': reg_pred_dict['deformed_source'],
                    'velocity': reg_pred_dict['velocity'],
                    'momentum': reg_pred_dict['momentum'],
                    'displacement': reg_pred_dict['displacement'],
                    'LMA_pred': LMA_pred,
                }
                joint_target_dict = {
                    'registration_target': tar,
                    'TOS_curves': TOS_curves,
                    'LMA_labels': LMA_labels,
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
                
                # epoch_total_training_registration_reconstruction_loss += registration_loss.item()
                # epoch_total_training_registration_supervised_loss += 0
                # epoch_total_training_regression_loss += regression_loss.item()
                # epoch_total_training_loss += train_total_loss.item()
                # epoch_loss_dict['training/total_loss'] += train_total_loss.item()
                # epoch_loss_dict['training/registration_reconstruction_loss'] += registration_loss.item()
                # epoch_loss_dict['training/registration_supervised_loss'] += 0
                # epoch_loss_dict['training/LMA_loss'] += LMA_loss.item()
                for loss_name, loss_value in losses_values_dict.items():
                    record_name = f'training/{loss_name}'
                    if record_name not in epoch_loss_dict.keys():
                        epoch_loss_dict[record_name] = loss_value
                    else:
                        epoch_loss_dict[record_name] += loss_value

                    # epoch_loss_dict[record_name] += loss_value if record_name in epoch_loss_dict.keys() else loss_value

                # if LMA task is classification, 
                # record the number of data and correct predictions 
                # to compute the accuracy of the current epoch
                if LMA_task == 'LMA_classification':
                    batch_size = LMA_pred.shape[0]
                    _, LMA_pred_class = torch.max(LMA_pred, dim=1)
                    correct_pred_num = torch.sum(LMA_pred_class == LMA_labels)

                    epoch_training_data_num += batch_size
                    epoch_training_classification_correct_pred_num += correct_pred_num
            # compute epoch training accuracy
            if LMA_task == 'LMA_classification':
                epoch_training_classification_accuracy = epoch_training_classification_correct_pred_num / epoch_training_data_num
                epoch_loss_dict['training/LMA_accuracy'] = epoch_training_classification_accuracy
            
            # update learning rate
            registration_lr_scheduler.step()
            LMA_lr_scheduler.step()
            
            
            # if use_tensorboard:
            #     for loss_name, loss_value in epoch_loss_dict.items():
            #         writer.add_scalar(loss_name, loss_value, epoch)
                # writer.add_scalar('training/total_loss', epoch_total_training_loss, epoch)
                # writer.add_scalar('training/registration_reconstruction_loss', epoch_total_training_registration_reconstruction_loss, epoch)
                # writer.add_scalar('training/registration_supervised_loss', epoch_total_training_registration_supervised_loss, epoch)
                # writer.add_scalar('training/regression_loss', epoch_total_training_regression_loss, epoch)
                # writer.close()

            # validate
            registration_model.eval()
            LMA_model.eval()
            epoch_total_val_loss = 0
            epoch_loss_dict['validation/total_loss'] = 0
            epoch_val_classification_correct_pred_num = 0
            epoch_val_data_num = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # get data
                    src, tar = batch['source_img'], batch['target_img']
                    src = src.to(used_device)
                    tar = tar.to(used_device)

                    # forward
                    pred_dict = registration_model(src, tar)

                    # compute loss
                    val_registration_loss = self.compute_training_loss(pred_dict, registration_model, src, tar)

                    # Merge the displacement fields of the same slice as a video / image sequence
                    # Remove the redundant TOS for the same slice                 
                    pred_displacement_fields, TOS_curves, LMA_labels, batch_slice_full_ids = merge_data_of_same_slice_from_batch(batch, pred_dict, n_frames_to_use_for_regression, used_device)

                    # compute regression loss
                    LMA_pred = LMA_model(pred_displacement_fields)
                    # if LMA_task == 'TOS_regression':
                    #     LMA_loss = self.LMA_loss_fn(LMA_pred, TOS_curves)
                    # elif LMA_task == 'LMA_classification':
                    #     LMA_loss = self.LMA_loss_fn(LMA_pred, LMA_labels)

                    joint_val_pred_dict = {
                        'deformed_source': pred_dict['deformed_source'],
                        'velocity': pred_dict['velocity'],
                        'momentum': pred_dict['momentum'],
                        'displacement': pred_dict['displacement'],
                        'LMA_pred': LMA_pred,
                    }
                    joint_val_target_dict = {
                        'registration_target': tar,
                        'TOS_curves': TOS_curves,
                        'LMA_labels': LMA_labels,
                    }
                    val_total_loss, val_losses_values_dict = self.loss_calculator(joint_val_pred_dict, joint_val_target_dict)

                    # val_total_loss = val_registration_loss + 1e-1 * LMA_loss

                    # update progress bar
                    progress_bar.set_description(
                        f'Epoch {epoch} | Train Loss {val_total_loss.item():.3e} | Val Loss {val_total_loss.item():.3e}')
                    epoch_total_val_loss += val_total_loss.item()

                    if LMA_task == 'LMA_classification':
                        batch_size = LMA_pred.shape[0]
                        _, LMA_pred_class = torch.max(LMA_pred, dim=1)
                        correct_pred_num = torch.sum(LMA_pred_class == LMA_labels)

                        epoch_val_data_num += batch_size
                        epoch_val_classification_correct_pred_num += correct_pred_num
            epoch_loss_dict['validation/total_loss'] = epoch_total_val_loss
            if use_wandb:
                wandb_experiment.log(epoch_loss_dict, step=epoch)
            if LMA_task == 'LMA_classification':
                epoch_val_classification_accuracy = epoch_val_classification_correct_pred_num / epoch_val_data_num
                epoch_loss_dict['validation/LMA_accuracy'] = epoch_val_classification_accuracy
            
            if early_stop:
                # save best model
                if epoch_total_val_loss < best_val_loss:
                    best_val_loss = epoch_total_val_loss
                    best_registration_model = copy.deepcopy(registration_model)
                    best_LMA_model = copy.deepcopy(LMA_model)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f'Epochs without improvement: {epochs_without_improvement} / {epochs_without_improvement_tolerance}')

                # early stopping
                if epochs_without_improvement >= epochs_without_improvement_tolerance:
                    print(f'Early stopping at epoch {epoch}')
                    return {
                        'cine_registraion': best_registration_model,
                        'LMA': best_LMA_model,
                    }

            
            if use_tensorboard:
                for loss_name, loss_value in epoch_loss_dict.items():
                    writer.add_scalar(loss_name, loss_value, epoch)
                # writer.add_scalar('validation?total_loss', epoch_total_val_loss, epoch)
                # writer.close()
                writer.flush()
        if use_tensorboard:
            writer.close()

        if early_stop:
            registration_model = best_registration_model
            LMA_model = best_LMA_model
        
        return {
            'cine_registraion': registration_model,
            'LMA': LMA_model,
        }
    
    # function inference: alias to test
    def inference(self, model, datasets, trainer_config=None, full_config=None, device=None):
        return self.test(model, datasets, trainer_config, full_config, device)
    
    def test(self, models, datasets, trainer_config=None, full_config=None, device=None):
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # build dataloaders
        test_loader = SliceDataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=True, num_workers=1)

        # other parameters
        # n_frames_to_use_for_regression = full_config['networks']['TOS_regression']['input_frame_num']
        n_frames_to_use_for_regression = 25
        
        # get models
        LMA_task = trainer_config.get('LMA_task', 'TOS_regression')
        registration_model = models['cine_registraion']
        LMA_model = models['LMA']
        # if LMA_task == 'TOS_regression':
        #     LMA_model = models['TOS_regression']
        #     self.LMA_loss_fn = torch.nn.MSELoss()
        # elif LMA_task == 'LMA_classification':
        #     LMA_model = models['LMA_classification']
        #     self.LMA_loss_fn = torch.nn.CrossEntropyLoss()
        
        registration_model.eval()
        LMA_model.eval()

        # build loss function
        self.regression_loss_fn = torch.nn.MSELoss()
        
        # print the device of model
        # print(f'Model is on {next(regression_model.parameters()).device}')
        # build progress bar
        test_preds = []
        progress_bar = tqdm(range(used_train_config['epochs']))
        for epoch in progress_bar:
            # train
            # for model_name, model in models.items():
            #     model.train()
            # registration_model.train()
            # LMA_model.train()
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
                    
                    pred_displacement_fields, TOS_curves, LMA_labels, batch_slice_full_ids = merge_data_of_same_slice_from_batch(batch, reg_pred_dict, n_frames_to_use_for_regression, used_device)

                    # compute regression loss
                    regression_pred = LMA_model(pred_displacement_fields)
                    # regression_loss = self.regression_loss_fn(regression_pred, TOS_curves)
                    

                    # train_total_loss = registration_loss + 1e-2 * regression_loss
                    test_total_loss = registration_loss

                    # update progress bar
                    progress_bar.set_description(f'Epoch {epoch} | Train Loss {test_total_loss.item():.3e}')

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
                        slice_idx_of_current_pair = batch_slice_full_ids.index(batch['slice_full_id'][i])
                        test_pred_dict['TOS_pred'] = regression_pred[slice_idx_of_current_pair].cpu().numpy()
                        
                        
                        # Add the test_pred_dict to test_preds
                        test_preds.append(test_pred_dict)

        return test_preds

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
    
    def visualize_pred_registraion(self, preds, n_vis=5, vis_indices=None):
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
        
        # plot the strain matrices and TOS curves
        # fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        # for plot_idx, vis_test_idx in enumerate(vis_indices):

        # axs[0].imshow(np.random.randn(128, 128), cmap='gray')

    def visualize_pred_regression(self, preds, n_vis=5):
        # plot the strain matrices and TOS curves
        vis_indices = np.random.randint(0, len(preds), n_vis)
        fig, axs = plt.subplots(1, n_vis, figsize=(n_vis*3, 3))
        for plot_idx, vis_test_idx in enumerate(vis_indices):
            axs[plot_idx].pcolor(preds[vis_test_idx]['strain_mat'][0], cmap='jet', vmin=-0.3, vmax=0.3)
            axs[plot_idx].plot(preds[vis_test_idx]['TOS']/17+1, np.arange(126), color='black')
            axs[plot_idx].plot(preds[vis_test_idx]['TOS_pred']/17+1, np.arange(126), color='red', linestyle='--')
            axs[plot_idx].set_xlim([0, 50])
            

        # return fig, axs
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
