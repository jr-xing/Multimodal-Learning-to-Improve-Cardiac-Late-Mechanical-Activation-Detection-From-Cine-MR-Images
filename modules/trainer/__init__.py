from modules.trainer.reg_trainer import RegTrainer
from modules.trainer.joint_registration_regression_trainer import JointRegistrationRegressionTrainer
from modules.trainer.LMA_trainer import LMATrainer
from modules.trainer.strainmat_pred_trainer import StrainmatPredTrainer
from modules.trainer.strainmat_LMA_trainer import StrainmatLMATrainer
from modules.trainer.joint_registration_strainmat_LMA import JointRegisterStrainmatLMATrainer
def build_trainer(trainer_config, device=None, full_config=None):
    trainer_scheme = trainer_config['scheme']
    if trainer_scheme == 'reg':
        return RegTrainer(trainer_config, device, full_config)
    elif trainer_scheme == 'LMA':
        return LMATrainer(trainer_config, device, full_config)
    elif trainer_scheme == 'joint_registration_regression':
        return JointRegistrationRegressionTrainer(trainer_config, device, full_config)
    elif trainer_scheme == 'strainmat_pred':
        return StrainmatPredTrainer(trainer_config, device, full_config)
    elif trainer_scheme == 'strainmat_LMA':
        return StrainmatLMATrainer(trainer_config, device, full_config)
    elif trainer_scheme == 'joint_registration_strainmat_LMA':
        return JointRegisterStrainmatLMATrainer(trainer_config, device, full_config)
    else:
        raise NotImplementedError(f"trainer scheme {trainer_scheme} not implemented")