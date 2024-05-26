import torch
from modules.loss.registration_losses import RegistrationReconstructionLoss
class HardCodedLossCalculator:
    def __init__(self, losses_confs: dict, full_config: dict = None, device=None):
        self.losses = losses_confs
        self.full_config = full_config
        self.device = device if device is not None else torch.device('cpu')

        # get loss functions
        self.registration_loss_fn = RegistrationReconstructionLoss(sigma=losses_confs['registration_reconstruction']['sigma'], regularization_weight=losses_confs['registration_reconstruction']['regularization_weight'])
        self.registration_loss_weight = self.losses['registration_reconstruction']['weight']

        self.displacement_loss_fn = torch.nn.MSELoss()
        self.LMA_task = self.losses['LMA']['task']
        if self.LMA_task in ['TOS_regression']:
            self.LMA_loss_fn = torch.nn.MSELoss()
        elif self.LMA_task in ['TOS_classification', 'LMA_sector_classification', 'LMA_slice_classification']:
            self.LMA_loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"LMA task {self.LMA_task} not implemented")