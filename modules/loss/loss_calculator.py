# class lossCalculator
# Description: This class is used to calculate the loss of the model
import torch
from modules.loss.registration_losses import RegistrationReconstructionLoss
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientMagnitudeLayer(nn.Module):
    def __init__(self, device='cuda'):
        super(GradientMagnitudeLayer, self).__init__()
        # Define Sobel filters for x and y direction
        self.sobel_x = nn.Parameter(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view((1, 1, 3, 3)), requires_grad=False).to(device)
        self.sobel_y = nn.Parameter(torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view((1, 1, 3, 3)), requires_grad=False).to(device)

    def forward(self, x):
        # Check input dimensions
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Expected input shape (N, 1, H, W)")

        # Compute gradients
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return grad_magnitude

class GradientMagnitudeLoss(nn.Module):
    def __init__(self, config, **kwargs):
        super(GradientMagnitudeLoss, self).__init__()
        self.offset = config.get('offset', 0)
        self.prediction = config['prediction']
        self.target = config['target']
        self.gradient_magnitude_layer = GradientMagnitudeLayer()

    def forward(self, x, y):
        # y is not used
        # Check input dimensions
        x = x[self.prediction]
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Expected input shape (N, 1, H, W)")

        # Compute gradient magnitude
        grad_magnitude = self.gradient_magnitude_layer(x)

        # Calculate loss: Sum over each image's pixels, then mean over the batch
        loss_per_image = torch.abs(torch.sum(torch.abs(grad_magnitude), dim=[1, 2, 3]) - self.offset)  # Sum over H, W dimensions
        loss = torch.mean(loss_per_image)  # Mean over N (batch dimension)
        return loss

class MSELoss:
    """
    Custom MSE loss function 
    """
    def __init__(self, config, **kwargs):
        # self.weight = weight
        self.prediction = config['prediction']
        self.target = config['target']
        self.loss_function = torch.nn.MSELoss(**kwargs)
        
    def __call__(self, outputs, targets):
        loss = self.loss_function(outputs[self.prediction], targets[self.target])
        return loss

class CrossEntropyLoss:
    """
    Custom CrossEntropy loss function 
    """
    def __init__(self, config, **kwargs):
        # self.weight = weight
        self.prediction = config['prediction']
        self.target = config['target']
        self.loss_function = torch.nn.CrossEntropyLoss(**kwargs)
        
    def __call__(self, outputs, targets):
        loss = self.loss_function(outputs[self.prediction], targets[self.target])
        return loss


def get_loss_function(loss_conf, full_config=None):
    if loss_conf['criterion'] in ['cross_entropy', 'CrossEntropyLoss']:
        # return torch.nn.CrossEntropyLoss()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # class_weights_list = loss_conf.get('class_weights', [1, 1])
        # class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
        
        # print('cross entropy class_weights: {}'.format(class_weights))
        # return CrossEntropyLoss(loss_conf, weight=class_weights)
        return CrossEntropyLoss(loss_conf)
    elif loss_conf['criterion'] in ['mse', 'MSELoss']:
        # return torch.nn.MSELoss()
        return MSELoss(loss_conf)
    elif loss_conf['criterion'] in ['registration_reconstruction']:
        return RegistrationReconstructionLoss(sigma=loss_conf['sigma'], regularization_weight=loss_conf['regularization_weight'])
    elif loss_conf['criterion'] in ['gradient_magnitude']:
        return GradientMagnitudeLoss(loss_conf)
    else:
        raise NotImplementedError("Loss function {} not implemented".format(loss_conf['name']))

class LossCalculator:
    def __init__(self, losses_confs: dict, full_config: dict = None, device=None):
        self.losses = copy.deepcopy(losses_confs)
        self.full_config = copy.deepcopy(full_config)
        self.device = device if device is not None else torch.device('cpu')

        # self.losses_functions = {}
        for loss_name, loss_conf in self.losses.items():
            self.losses[loss_name]['function'] = get_loss_function(loss_conf, self.full_config)

    def __call__(self, outputs, targets):
        total_loss = 0
        losses_values = {}
        for loss_name, loss_conf in self.losses.items():
            if loss_conf['enable'] is False:
                continue
            # prediction = outputs[loss_conf['prediction']]
            # target = targets[loss_conf['target']]
            loss = loss_conf['function'](outputs, targets)
            losses_values[loss_name] = loss.item()
            total_loss += loss_conf['weight'] * loss
        losses_values['total_loss'] = total_loss.item()
        return total_loss, losses_values