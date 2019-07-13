import sys
import warnings
from tqdm import tqdm
from pathlib import Path
from numpy import inf
import numpy as np
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


class BaseTrainer:
    def __init__(self, model, loss, optimizer, args):
        self.args = args
        self.n_gpu_use = args.n_gpu_use
        self.device, self.data_type = self._prepare_device()
        self.model = model.to(self.device)

        if self.n_gpu_use > 1:
            self.model = torch.nn.DataParallel(model, device_ids=list(range(self.n_gpu_use)))

        self.loss = loss
        self.optimizer = optimizer

        self.start_epoch = 1
        self.epochs = args.epochs
        self.save_period = args.save_period
        self.early_stop = args.early_stop
        self.mnt_best = inf
        self.not_improved_count = 0

        ensure_dir(self.args.save_dir)
        self.checkpoint_dir = Path(args.save_dir)
        # self.loss_progress = {'loss': [], 'val_loss': []}
        self.record_loss = False
        self.loss_progress = {}

        if args.resume is not None:
            self.resume_path = Path(args.resume).resolve()
            self._resume_checkpoint()
        else:
            self._save_config()

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if not self.record_loss:
                for k, v in result.items():
                    self.loss_progress[k] = [result[k]]
                self.record_loss = True
            else:
                self.loss_progress[k].append(result[k])

            for k, v in result.items():
                print("  {}: {}".format(str(k), v))

            is_best = False
            try:
                improved = result['val_loss'] <= self.mnt_best

                if improved:
                    self.mnt_best = result['val_loss']
                    self.not_improved_count = 0
                    is_best = True
                else:
                    self.not_improved_count += 1

                if self.not_improved_count == self.early_stop:
                    print("Validation loss has not improved for {} epochs. Stop training.".format(self.early_stop))
                    break
            except Exception:
                pass

            is_period = epoch % self.save_period == 0
            if (is_period) or (is_best):
                self._save_checkpoint(epoch, is_period=is_period, is_best=is_best)

        print("The best loss: {}".format(self.mnt_best))
        if self.args.resume is not None:
            np.save(str(self.checkpoint_dir / 'loss_progress-{}.npy'.format(self.start_epoch)), self.loss_progress)
        np.save(str(self.checkpoint_dir / 'loss_progress.npy'), self.loss_progress)
        print("Loss of training progress saved.")

    def _save_checkpoint(self, epoch, is_period, is_best):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'args': self.args
        }
        if is_period:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch-{}.pth'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: %s ..." % filename)
        if is_best:
            filename = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, filename)
            print("Saving current best: %s ..." % filename)

    def _save_config(self):
        with open(str(self.checkpoint_dir / 'cmd.txt'), 'w') as f:
            f.write(' '.join(sys.argv))

        orig_stdout = sys.stdout
        with open(str(self.checkpoint_dir / 'arch_summary.txt'), 'w') as f:
            sys.stdout = f
            print(self.model)
        sys.stdout = orig_stdout

        with open(str(self.checkpoint_dir / 'args.txt'), 'w') as f:
            f.write(str(self.args))

    def _prepare_device(self):
        n_gpu = torch.cuda.device_count()
        if self.n_gpu_use > 0 and n_gpu == 0:
            warnings.warn("Warning: There is no GPU available, training will be performed on CPU.")
            self.n_gpu_use = 0
        if self.n_gpu_use > n_gpu:
            warnings.warn("Warning: N. of GPU specified to use: %d, only %d are availabe." % (self.n_gpu_use, n_gpu))
            self.n_gpu_use = n_gpu

        device = torch.device('cuda:0' if self.n_gpu_use > 0 else 'cpu')
        if device.type == 'cuda':
            default_dtype = 'torch.cuda.FloatTensor'
        else:
            default_dtype = 'torch.FloatTensor'
        torch.set_default_tensor_type(default_dtype)

        return device, default_dtype

    def _resume_checkpoint(self):
        resume_path = str(self.resume_path)
        print("Loading checkpoint: %s ..." % resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class DisentangleSeqVaeTrainer(BaseTrainer):
    def __init__(self, model, loss, optimizer, args, data_loader, valid_data_loader=None, lr_scheduler=None, fold=None):
        super(DisentangleSeqVaeTrainer, self).__init__(model, loss, optimizer, args)

        if args.cross_valid > 1:
            dir_fold = 'fold-%d' % fold
            result_subfolder = str(Path(self.args.save_dir) / dir_fold)
            ensure_dir(result_subfolder)
            self.checkpoint_dir = Path(result_subfolder)

        self.args = args
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_mse = 0
        total_kld_f = 0
        total_kld_z = 0
        bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, x in bar:
            x = x.type(self.data_type)

            self.optimizer.zero_grad()
            f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean, z_prior_logvar, recon_x = self.model(x)
            loss, mse, kld_f, kld_z = self.loss(x, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar,
                                                z_prior_mean, z_prior_logvar)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_kld_f += kld_f.item()
            total_kld_z += kld_z.item()

            bar.set_description("epoch: {}/{}".format(epoch, self.epochs))

        log = {
            'loss': total_loss / len(self.data_loader),
            'mse': total_mse / len(self.data_loader),
            'kld_f': total_kld_f / len(self.data_loader),
            'kld_z': total_kld_z / len(self.data_loader)
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()

        total_val_loss = 0
        total_val_mse = 0
        total_val_kld_f = 0
        total_val_kld_z = 0
        for batch_idx, x in enumerate(self.valid_data_loader):
            x = x.type(self.data_type)

            f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean, z_prior_logvar, recon_x = self.model(x)
            loss, mse, kld_f, kld_z = self.loss(x, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar,
                                                z_prior_mean, z_prior_logvar)
            total_val_loss += loss.item()
            total_val_mse += mse.item()
            total_val_kld_f += kld_f.item()
            total_val_kld_z += kld_z.item()

            """
            if epoch == chechkpoint:
                1. save reconstructed image
                2. do unconditional sampling
                3. do style transfer
            """
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_mse': total_val_mse / len(self.valid_data_loader),
            'val_kld_f': total_val_kld_f / len(self.valid_data_loader),
            'val_kld_z': total_val_kld_z / len(self.valid_data_loader)
        }
