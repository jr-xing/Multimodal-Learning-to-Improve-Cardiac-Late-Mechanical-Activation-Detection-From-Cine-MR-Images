import torch
# def registration_reconstruction_loss(pred_dict, model, src, tar):
#         reconst_loss_fn = torch.nn.MSELoss()
#         u = pred_dict['displacement']
#         v = pred_dict['velocity']
#         m = pred_dict['momentum']
#         Sdef = pred_dict['deformed_source']
#         regularization_weight = 1

#         # compute loss
#         loss1 = reconst_loss_fn(tar, Sdef)
#         loss2 = (v*m).sum() / (src.numel())
#         loss_regis = 0.5 * loss1/(model.sigma*model.sigma) + regularization_weight * loss2

#         return loss_regis
# def registration_reconstruction_loss
class RegistrationReconstructionLoss:
    def __init__(self, sigma, regularization_weight=1):
        self.sigma = sigma
        self.regularization_weight = regularization_weight

    def __call__(self, prediction, target):
        Sdef = prediction['deformed_source']
        tar = target['registration_target']
        recon_loss = torch.nn.MSELoss()(tar, Sdef)
        regularization = (prediction['velocity']*prediction['momentum']).sum() / (tar.numel())
        loss = 0.5 * recon_loss/(self.sigma*self.sigma) + regularization * self.regularization_weight
        return loss