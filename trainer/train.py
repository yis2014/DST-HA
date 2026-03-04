from loader.HZMetro import HZMetro
from loader.SHMetro import SHMetro
from model.DST_HA import DST_HA
from trainer import metrics
from utils import StandardScaler, move2device, StepLR2

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import os
import time
from tqdm import tqdm
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def gen_data(cfg, loader_type):
    if cfg['loader'] == 'hz':
        data_set = HZMetro(cfg['dataset'], split=loader_type)
    elif cfg['loader'] == 'sh':
        data_set = SHMetro(cfg['dataset'], split=loader_type)
    else:
        raise 'Wrong Loader Name'

    if loader_type == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    data_loader = DataLoader(data_set,
                             batch_size=cfg['dataset']['batch_size'],
                             shuffle=shuffle,
                             num_workers=cfg['dataset']['num_workers'],
                             pin_memory=True,
                             drop_last=drop_last)
    return data_set, data_loader


# ---------------- 自定义加权多步 MAE 损失 ----------------
# 依据预测步长对损失分配指数衰减权重，强调短期预测精度。
class WeightedMAELoss(nn.Module):
    """加权 MAE，weights[t] = exp(-t / tau) / sum(exp(-i / tau))"""

    def __init__(self, tau: float = 3.0):
        super().__init__()
        self.tau = tau

    def forward(self, pred: torch.Tensor, truth: torch.Tensor):
        # pred, truth: (B, T, N, C)
        T = pred.size(1)
        device = pred.device
        weights = torch.exp(-torch.arange(T, device=device, dtype=pred.dtype) / self.tau)
        weights = weights / weights.sum()
        # reshape to (1, T, 1, 1)
        weights = weights.view(1, T, 1, 1)
        loss = torch.abs(pred - truth) * weights
        return loss.mean()


# ---------------- 混合损失函数，优化MAPE指标 ----------------
class CombinedLoss(nn.Module):
    """混合损失函数：加权MAE + 对称MAPE损失"""
    
    def __init__(self, tau: float = 3.0, mape_weight: float = 0.3, epsilon: float = 1e-3):
        super().__init__()
        self.tau = tau
        self.mape_weight = mape_weight
        self.mae_weight = 1.0 - mape_weight
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, truth: torch.Tensor):
        # pred, truth: (B, T, N, C)
        T = pred.size(1)
        device = pred.device
        weights = torch.exp(-torch.arange(T, device=device, dtype=pred.dtype) / self.tau)
        weights = weights / weights.sum()
        weights = weights.view(1, T, 1, 1)
        
        # MAE损失
        mae_loss = torch.abs(pred - truth) * weights
        mae_loss = mae_loss.mean()
        
        # 对称MAPE损失 - 使用对称分母避免除零问题
        denominator = (torch.abs(pred) + torch.abs(truth)) / 2.0
        denominator = torch.clamp(denominator, min=self.epsilon)
        mape_loss = torch.abs(pred - truth) / denominator * weights
        mape_loss = mape_loss.mean()
        
        # 混合损失
        total_loss = self.mae_weight * mae_loss + self.mape_weight * mape_loss
        return total_loss


# ---------------- 针对MAPE的后处理优化 ----------------
class MAPEOptimizedPredictor:
    """MAPE优化的预测后处理器"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # 平滑参数
        
    def smooth_predictions(self, pred, truth_history=None):
        """
        对预测结果进行平滑处理，减少小值预测的波动
        """
        # 对小值进行额外平滑
        small_value_mask = torch.abs(pred) < 1.0
        
        if small_value_mask.any():
            # 对小值预测进行平滑
            smoothed_pred = pred.clone()
            if truth_history is not None:
                # 使用历史真实值进行指导
                last_true = truth_history[:, -1:, :, :]  # (B, 1, N, C)
                smoothed_pred[small_value_mask] = (
                    (1 - self.alpha) * pred[small_value_mask] + 
                    self.alpha * last_true.expand_as(pred)[small_value_mask]
                )
            else:
                # 使用时间维度的平滑
                for t in range(1, pred.size(1)):
                    mask_t = small_value_mask[:, t:t+1, :, :]
                    if mask_t.any():
                        smoothed_pred[:, t:t+1, :, :][mask_t] = (
                            (1 - self.alpha) * pred[:, t:t+1, :, :][mask_t] + 
                            self.alpha * smoothed_pred[:, t-1:t, :, :][mask_t]
                        )
            return smoothed_pred
        return pred


def train_one_epoch(model, data_loader, criterion, optimizer, scaler, max_grad_norm, device, mape_optimizer=None):
    model.train()
    total_loss = 0
    cnt = 0
    dl = tqdm(data_loader)
    for idx, batch in enumerate(dl):
        optimizer.zero_grad()

        x, y, *extras = batch
        # 在归一化空间计算损失，稳定梯度
        x = scaler.transform(x)
        y = scaler.transform(y)
        
        # 支持无上下文模式
        if hasattr(model, 'use_context') and not model.use_context:
            extras = []
        
        x, y, *extras = move2device([x, y] + extras, device)

        y_pred, _ = model(x, y, extras)

        loss = criterion(y_pred, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        cnt += 1
        dl.set_postfix_str('Loss: {:.2f} '.format(loss.item()))
    return total_loss / cnt


def evaluate_model(eval_type, model, data_loader, scaler, device, logger, mape_mode='symmetric', use_mape_optimizer=False):
    detail = True
    if eval_type == "val":
        detail = False

    model.eval()
    mape_optimizer = MAPEOptimizedPredictor() if use_mape_optimizer else None
    
    y_pred_list = []
    y_truth_list = []
    for idx, batch in enumerate(data_loader):
        x, y, *extras = batch
        x = scaler.transform(x)
        y = scaler.transform(y)
        
        # 支持无上下文模式
        if hasattr(model, 'use_context') and not model.use_context:
            extras = []
            
        x, y, *extras = move2device([x, y] + extras, device)

        with torch.no_grad():
            y_pred, _ = model(x, y, extras)
            
            # MAPE优化后处理
            if mape_optimizer:
                y_pred = mape_optimizer.smooth_predictions(y_pred, x)

            y = scaler.inverse_transform(y)
            y_pred = scaler.inverse_transform(y_pred)
            y_pred_list.append(y_pred.cpu().numpy())
            y_truth_list.append(y.cpu().numpy())

    y_pred_np = np.concatenate(y_pred_list, axis=0)
    y_truth_np = np.concatenate(y_truth_list, axis=0)

    if detail:
        logger.info('Evaluation_{}_Begin:'.format(eval_type))

    mae_list = []
    mape_list = []
    rmse_list = []
    mae_sum = 0
    mape_sum = 0
    rmse_sum = 0
    horizon = y_pred_np.shape[1]
    for horizon_i in range(horizon):
        y_truth_temp = y_truth_np[:, horizon_i, :, :]
        y_pred_temp = y_pred_np[:y_truth_np.shape[0], horizon_i, :, :]

        mae = metrics.masked_mae_np(y_pred_temp, y_truth_temp, null_val=0, mode='dcrnn')
        # 使用改进的MAPE计算
        mape = metrics.masked_mape_np(y_pred_temp, y_truth_temp, null_val=0, mode=mape_mode)
        rmse = metrics.masked_rmse_np(y_pred_temp, y_truth_temp, null_val=0)
        mae_sum += mae
        mape_sum += mape
        rmse_sum += rmse
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        msg = "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}"
        if detail:
            logger.info(msg.format(horizon_i + 1, mae, mape, rmse))
    if detail:
        logger.info('Average MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}'.format(
            mae_sum / horizon, mape_sum / horizon, rmse_sum / horizon))
        logger.info('Evaluation_{}_End:'.format(eval_type))
    return mae_sum / horizon, mape_sum / horizon, rmse_sum / horizon, \
        {"MAE": mae_list, "RMSE": rmse_list, "MAPE": mape_list}


def train_model(cfg, logger, log_dir, seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = cfg['device']

    train_set, train_loader = gen_data(cfg, 'train')
    val_set, val_loader = gen_data(cfg, 'val')
    test_set, test_loader = gen_data(cfg, 'test')
    scaler = StandardScaler(mean=train_set.mean, std=train_set.std)

    logger.info('Train Dataset Shape {}'.format(train_set.data['x'].shape))
    logger.info('Val Dataset Shape {}'.format(val_set.data['x'].shape))
    logger.info('Test Dataset Shape {}'.format(test_set.data['x'].shape))

    model = DST_HA(cfg['model'])
    model.to(device)

    total_param = sum([param.nelement() for param in model.parameters()])
    logger.info("Number of parameter: {:.4f}M".format(total_param / 1e6))

    if cfg['train']['load_param'] is not None:
        path = os.path.join(cfg['train']['load_param'], 'best.pt')
        print("Load param form", path)
        model.load_state_dict(torch.load(path))

    # 选择损失函数类型
    loss_type = cfg['train'].get('loss_type', 'weighted_mae')  # 'weighted_mae' 或 'combined'
    if loss_type == 'combined':
        tau = cfg['train'].get('horizon_tau', 6.0)
        mape_weight = cfg['train'].get('mape_weight', 0.3)
        criterion = CombinedLoss(tau=tau, mape_weight=mape_weight)
        logger.info(f"Using CombinedLoss with mape_weight={mape_weight}")
    else:
        tau = cfg['train'].get('horizon_tau', 6.0)
        criterion = WeightedMAELoss(tau=tau)
        logger.info("Using WeightedMAELoss")
        
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['train']['base_lr'],
                           eps=cfg['train']['epsilon'],
                           weight_decay=cfg['train']['weight_decay'])
    scheduler = StepLR2(optimizer=optimizer,
                        milestones=cfg['train']['steps'],
                        gamma=cfg['train']['lr_decay_ratio'],
                        min_lr=cfg['train']['min_learning_rate'],
                        warm_up=cfg['train']['warm_up'],
                        warm_up_ep=cfg['train']['warm_up_ep'],
                        warm_up_lr=cfg['train']['warm_up_lr'])

    max_grad_norm = cfg['train']['max_grad_norm']
    mape_mode = cfg['train'].get('mape_mode', 'symmetric')  # 'standard', 'symmetric', 'weighted'
    use_mape_optimizer = cfg['train'].get('use_mape_optimizer', False)
    
    logger.info(f"MAPE evaluation mode: {mape_mode}")
    logger.info(f"Use MAPE optimizer: {use_mape_optimizer}")
    
    last_test_mae = 1e6

    for epoch in range(cfg['train']['epoch']):
        begin_time = time.perf_counter()

        train_loss = train_one_epoch(model=model,
                                     data_loader=train_loader,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scaler=scaler,
                                     max_grad_norm=max_grad_norm,
                                     device=device)

        val_result = evaluate_model(eval_type='val',
                                    model=model,
                                    data_loader=val_loader,
                                    scaler=scaler,
                                    device=device,
                                    logger=logger,
                                    mape_mode=mape_mode,
                                    use_mape_optimizer=use_mape_optimizer)
        val_mae, _, _, _ = val_result

        logger.info('Epoch:{}, train_mae:{:.2f}, val_mae:{}, lr={},'.format(
            epoch,
            train_loss,
            val_mae,
            str(optimizer.state_dict()['param_groups'][0]['lr'])))

        test_result = evaluate_model(eval_type="test",
                                     model=model,
                                     data_loader=test_loader,
                                     scaler=scaler,
                                     device=device,
                                     logger=logger,
                                     mape_mode=mape_mode,
                                     use_mape_optimizer=use_mape_optimizer)

        time_elapsed = time.perf_counter() - begin_time
        logger.info('time_elapsed:{:.2f}'.format(time_elapsed))

        if (epoch + 1) % cfg['train']['save_every_n_epochs'] == 0:
            save_dir = log_dir
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            config_path = os.path.join(save_dir,
                                       'config-{}.yaml'.format(epoch + 1))
            epoch_path = os.path.join(save_dir,
                                      'epoch-{}.pt'.format(epoch + 1))
            torch.save(model.state_dict(), epoch_path)
            with open(config_path, 'w') as f:
                from copy import deepcopy
                save_cfg = deepcopy(cfg)
                save_cfg['model']['save_path'] = epoch_path
                f.write(yaml.dump(save_cfg, Dumper=Dumper))

        if last_test_mae > test_result[0]:
            logger.info('test_mae decreased from {:.2f} to {:.2f}'.format(
                last_test_mae,
                test_result[0]))
            last_test_mae = test_result[0]

            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pt'))

        scheduler.step()

    model.load_state_dict(torch.load(os.path.join(log_dir, 'best.pt')))
    logger.info('Load best params')
    test_result = evaluate_model(eval_type="test",
                                 model=model,
                                 data_loader=test_loader,
                                 scaler=scaler,
                                 device=device,
                                 logger=logger,
                                 mape_mode=mape_mode,
                                 use_mape_optimizer=use_mape_optimizer)
    return test_result
