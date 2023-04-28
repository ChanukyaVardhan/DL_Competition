import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method

import torchvision.transforms as transforms
import numpy as np

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return SimVP_Model(**config).to(self.device)

    def _predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        # print(f"Length of train_loader on device {self.rank} = {len(train_loader)}")
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for idx, (batch_x, batch_y) in enumerate(train_pbar):

            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y = self._predict(batch_x)
                loss = self.criterion(pred_y, batch_y)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())

            self.model_optim.step()
            # FIX : COMMENT OUT WHEN RUNNING ON CPU
            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        print(f"Length of train_loader on device {self.rank} = {idx + 1}")

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader."""
        # print(f"Length of vali_loader on device {self.rank} = {len(vali_loader)}")
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.eval()
        val_pbar = tqdm(vali_loader) if self.rank == 0 else vali_loader
        eval_loss = 0.0
        counter = 0
        results = []

        def unnormalize(img):
            mean = [0.5061, 0.5045, 0.5008]
            std = [0.0571, 0.0567, 0.0614]
            unnormalize_transform = transforms.Compose([
                transforms.Normalize(
                    mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
            ])
            to_pil = transforms.ToPILImage()

            unnormalized_image = unnormalize_transform(img)
            pil_images = to_pil(unnormalized_image)

            return pil_images

        end = time.time()
        for idx, (batch_x, batch_y) in enumerate(val_pbar):
            data_time_m.update(time.time() - end)
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                pred_y = self._predict(batch_x)
                loss = self.criterion(pred_y, batch_y)

                if self.dist:
                    reduced_loss = reduce_tensor(loss)
                else:
                    reduced_loss = loss

                losses_m.update(reduced_loss.item(), batch_x.size(0))                    

                if self.args.empty_cache:
                    torch.cuda.empty_cache()

                if self.rank == 0:
                    log_buffer = 'eval loss: {:.4f}'.format(reduced_loss.item())
                    log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                    val_pbar.set_description(log_buffer)

                if counter < 5: # Pick 5 images across the whole dataset
                    results.append(dict(zip(['preds', 'trues'],
                                            [unnormalize(pred_y[0, -1].detach()), unnormalize(batch_y[0, -1].detach())])))
                    counter += 1;

                end = time.time()  # end for

                eval_loss += reduced_loss.item()
        eval_loss /= (idx + 1)
        print(f"Length of vali_loader on device {self.rank} = {idx + 1}")

        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.stack([batch[k] for batch in results], axis=0)
        
        return results_all['preds'], results_all['trues'], eval_loss
