import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import lagomorph as lm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class RegTrainer:
    def __init__(self, trainer_config, device=None, full_config=None):
        self.trainer_config = trainer_config
        self.full_config = full_config
        self.device = device if device is not None else torch.device('cpu')

    def train(self, model, datasets, trainer_config=None, full_config=None, device=None, skip_validation=False):
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # build dataloaders
        train_loader = DataLoader(datasets['train'], batch_size=used_train_config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0)

        # build optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=used_train_config['optimizers']['registration']['learning_rate'])

        # build loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # print the device of model
        print(f'Model is on {next(model.parameters()).device}')
        # build progress bar
        progress_bar = tqdm(range(used_train_config['epochs']))
        for epoch in progress_bar:
            # train
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # get data
                src, tar = batch['source_img'], batch['target_img']
                src = src.to(used_device)
                tar = tar.to(used_device)

                # forward
                pred_dict = model(src, tar)

                # compute loss
                train_loss = self.compute_training_loss(pred_dict, model, src, tar)

                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # update progress bar
                progress_bar.set_description(f'Epoch {epoch} | Train Loss {train_loss.item():.3e}')

            # validate
            if not skip_validation:
                model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        # get data
                        src, tar = batch['source_img'], batch['target_img']
                        src = src.to(used_device)
                        tar = tar.to(used_device)

                        # forward
                        pred_dict = model(src, tar)

                        # compute loss
                        val_loss = self.compute_training_loss(pred_dict, model, src, tar)

                        # update progress bar
                        progress_bar.set_description(
                            f'Epoch {epoch} | Train Loss {train_loss.item():.3e} | Val Loss {val_loss.item():.3e}')
            print("")
        return model
    
    def train_with_early_stopping(self, model, datasets, trainer_config=None, full_config=None, device=None):
        # similar to train, but with early stopping, 
        # i.e. record the best model based on validation loss and load the best model at the end of training
        # if the validation loss does not decrease for a certain number of epochs, stop training and load the best model
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # build dataloaders
        train_loader = DataLoader(datasets['train'], batch_size=used_train_config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(datasets['test'], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0)

        # build optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=used_train_config['optimizers']['registration']['learning_rate'])

        # build loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # print the device of model
        print(f'Model is on {next(model.parameters()).device}')
        # build progress bar
        progress_bar = tqdm(range(used_train_config['epochs']))
        best_val_loss = float('inf')
        best_model = None
        epochs_without_improvement = 0
        for epoch in progress_bar:
            # train
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # get data
                src, tar = batch['source_img'], batch['target_img']
                src = src.to(used_device)
                tar = tar.to(used_device)

                # forward
                pred_dict = model(src, tar)

                # compute loss
                train_loss = self.compute_training_loss(pred_dict, model, src, tar)

                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # update progress bar
                progress_bar.set_description(f'Epoch {epoch} | Train Loss {train_loss.item():.4f}')

            # validate
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # get data
                    src, tar = batch['source_img'], batch['target_img']
                    src = src.to(used_device)
                    tar = tar.to(used_device)

                    # forward
                    pred_dict = model(src, tar)

                    # compute loss
                    val_loss = self.compute_training_loss(pred_dict, model, src, tar)
                    
                    # save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    # update progress bar
                    progress_bar.set_description(
                        f'Epoch {epoch} | Train Loss {train_loss.item():.4f} | Val Loss {val_loss.item():.4f}')
                    
                    # early stopping
                    if epochs_without_improvement >= used_train_config['early_stopping_patience']:
                        print(f'Early stopping at epoch {epoch}')
                        return best_model
                    
        return best_model
    
    # function inference: alias to test
    def inference(self, model, datasets, trainer_config=None, full_config=None, device=None):
        return self.test(model, datasets, trainer_config, full_config, device)

    def test(self, model, datasets, target_dataset_name='test', trainer_config=None, full_config=None, device=None):
        # Use the trainer_config and full_config passed to the function if they are not None, otherwise use the ones
        used_train_config = trainer_config if trainer_config is not None else self.trainer_config
        used_full_config = full_config if full_config is not None else self.full_config
        used_device = device if device is not None else self.device

        # build dataloaders
        test_loader = DataLoader(datasets[target_dataset_name], batch_size=used_train_config['batch_size'], shuffle=False, num_workers=0)

        # build loss function
        self.loss_fn = torch.nn.MSELoss()

        # test
        test_preds = []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # get data
                src, tar = batch['source_img'], batch['target_img']
                src = src.to(used_device)
                tar = tar.to(used_device)

                # forward
                pred_dict = model(src, tar)

                # compute loss
                test_loss = self.compute_training_loss(pred_dict, model, src, tar)

                # update progress bar
                # print(f'Test Loss {test_loss.item():.3e}')

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
        loss1 = self.loss_fn(tar, Sdef)
        loss2 = (v*m).sum() / (src.numel())
        loss_regis = 0.5 * loss1/(model.sigma*model.sigma) + reg_loss_weight * loss2

        return loss_regis
    
    def visualize_pred(self, preds, n_vis=5):
        # visualize the data in preds, which is the output of self.test() or self.inference()

        # check n_vis random sample from preds
        # make (5, n_vis) subplot using matplotlib
        # where each column shows 
        # (1) source image, (2) deformed source image, (3) target image
        # (4) abs(source - target), (5) abs(deformed source - target)
        import matplotlib.pyplot as plt
        import numpy as np
        vis_indices = np.random.randint(0, len(preds), n_vis)
        fig, axs = plt.subplots(5, n_vis, figsize=(n_vis*3, 15))
        for plot_idx, vis_test_idx in enumerate(vis_indices):
            curr_vis_test_pred = preds[vis_test_idx]
            curr_vis_deformed_source = curr_vis_test_pred['deformed_source'][0, :, :]
            curr_vis_deformed_source[curr_vis_deformed_source < 0.5] = 0
            curr_vis_deformed_source[curr_vis_deformed_source > 0.5] = 1
            # source image
            axs[0, plot_idx].imshow(curr_vis_test_pred['source_img'][0, :, :], cmap='gray')
            axs[0, plot_idx].set_title(f"{curr_vis_test_pred['subject_id']}-{curr_vis_test_pred['slice_idx']}")

            # deformed source image
            axs[1, plot_idx].imshow(curr_vis_deformed_source, cmap='gray')

            # target image
            axs[2, plot_idx].imshow(curr_vis_test_pred['target_img'][0, :, :], cmap='gray')

            # abs(source - target)
            axs[3, plot_idx].imshow(np.abs(curr_vis_test_pred['source_img'][0, :, :] - curr_vis_test_pred['target_img'][0, :, :]), cmap='gray')

            # abs(deformed source - target)
            axs[4, plot_idx].imshow(np.abs(curr_vis_deformed_source - curr_vis_test_pred['target_img'][0, :, :]), cmap='gray')

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
