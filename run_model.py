from trainer.train import train_model
from utils import get_logger

import yaml
import argparse
import os


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='运行DST_HA模型训练')
    parser.add_argument('--config', type=str, default='./config/hz.yaml', help='配置文件路径')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录路径')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型 (cuda/cpu)')
    parser.add_argument('--load_param', type=str, default=None, help='加载预训练模型路径')
    
    args = parser.parse_args()
    
    # 验证配置文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # 设置设备
    cfg['device'] = args.device
    
    # 处理预训练模型路径
    if args.load_param:
        cfg['train']['load_param'] = args.load_param
    elif cfg['train']['load_param'] == 'None':
        cfg['train']['load_param'] = None
    
    # 确定日志目录
    if args.log_dir:
        log_dir = args.log_dir
    else:
        # 从配置文件名自动生成日志目录
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        log_dir = f'log/{config_name}'
    
    # 创建日志目录和logger
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir)
    
    # 打印配置信息
    logger.info("="*60)
    logger.info("DST_HA 模型训练开始")
    logger.info("="*60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"设备: {cfg['device']}")
    logger.info(f"随机种子: {args.seed}")
    logger.info("="*60)
    logger.info("完整配置:")
    logger.info(cfg)
    logger.info("="*60)
    
    # 开始训练
    try:
        train_model(cfg, logger, log_dir, seed=args.seed)
        logger.info("训练完成!")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main()
