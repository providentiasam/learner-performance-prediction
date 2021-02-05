import os
import shutil
from tensorboardX import SummaryWriter
import wandb


class Logger:
    """Logging with TensorboardX. """

    def __init__(self, logdir, verbose=True, \
        project_name=None, run_name='run', model_args={}):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError:
            pass

        if project_name is not None:
            wandb.init(project=project_name, name=run_name, config=model_args)
            print('wandb init')
            self.wandb = True
        else:
            self.wandb = False

        self.verbose = verbose
        self.writer = SummaryWriter(logdir)

    def log_histograms(self, dic, step):
        """Log dictionary of tensors as histograms. """
        for k, v in dic.items():
            self.writer.add_histogram(k, v, step)

    def log_scalars(self, dic, step):
        """Log dictionary of scalar values. """
        for k, v in dic.items():
            self.writer.add_scalar(k, v, step)
        if self.wandb:
            wandb.log(dic, step=step)

        if self.verbose:
            print(f"Step {step}, {dic}")

    def close(self):
        self.writer.close()
