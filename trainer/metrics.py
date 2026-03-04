# part of this code are copied from DCRNN
import numpy as np

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan, mode='dcrnn'):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        if mode == 'dcrnn':
            return np.mean(mae)
        else:
            return np.mean(mae, axis=(0, 1))


def masked_mape_np(preds, labels, null_val=np.nan, epsilon=1e-3, mode='standard'):
    """
    改进的MAPE计算，支持多种模式
    Args:
        preds: 预测值
        labels: 真实值
        null_val: 需要忽略的值
        epsilon: 防止除零的小数
        mode: 'standard', 'symmetric', 'weighted'
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        
        if mode == 'standard':
            # 标准MAPE，添加epsilon避免除零
            mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), 
                                  np.maximum(np.abs(labels), epsilon)))
        elif mode == 'symmetric':
            # 对称MAPE，分母使用预测值和真实值的平均
            denominator = (np.abs(preds) + np.abs(labels)) / 2.0
            denominator = np.maximum(denominator, epsilon)
            mape = np.abs(np.subtract(preds, labels).astype('float32')) / denominator
        elif mode == 'weighted':
            # 加权MAPE，对大值给予更高权重
            weights = np.maximum(np.abs(labels), epsilon)
            mape = np.abs(np.subtract(preds, labels).astype('float32')) / weights
            # 按权重归一化
            mape = mape * weights / np.mean(weights * mask)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def masked_mape_by_threshold_np(preds, labels, null_val=np.nan, threshold=1.0):
    """
    基于阈值的MAPE计算，只计算真实值大于阈值的MAPE
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        
        # 添加阈值过滤
        threshold_mask = np.abs(labels) >= threshold
        mask = mask & threshold_mask
        
        if np.sum(mask) == 0:
            return 0.0
            
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)