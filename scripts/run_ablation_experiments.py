#!/usr/bin/env python3
"""
DST_HA 消融实验运行脚本
自动运行所有8组消融实验，并生成对比结果表格

消融实验设计：
SE = HSA-Local (局部空间注意力)
TE = CA-GRU Context (外部上下文注入)  
HT = HSA-Global (全局空间注意力)

实验组：
0. Full:         SE✓ TE✓ HT✓ (完整模型)
1. w.o SE&TE&HT: SE✗ TE✗ HT✗ (最低基线)
2. w.o SE&TE:    SE✗ TE✗ HT✓ (仅全局)
3. w.o SE&HT:    SE✗ TE✓ HT✗ (仅上下文)
4. w.o TE&HT:    SE✓ TE✗ HT✗ (仅局部)
5. w.o SE:       SE✗ TE✓ HT✓ (全局+上下文)
6. w.o TE:       SE✓ TE✗ HT✓ (HSA双分支)
7. w.o HT:       SE✓ TE✓ HT✗ (局部+上下文)
"""

import os
import sys
import subprocess
import time
import yaml
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class AblationExperiments:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(__file__))
        self.config_dir = os.path.join(self.base_dir, 'config', 'ablation')
        self.results_dir = os.path.join(self.base_dir, 'results', 'ablation')
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 实验配置
        self.experiments = [
            {'name': 'Full', 'config': 'hz_full.yaml', 'description': 'SE✓ TE✓ HT✓'},
            {'name': 'w.o SE&TE&HT', 'config': 'hz_wo_se_te_ht.yaml', 'description': 'SE✗ TE✗ HT✗'},
            {'name': 'w.o SE&TE', 'config': 'hz_wo_se_te.yaml', 'description': 'SE✗ TE✗ HT✓'},
            {'name': 'w.o SE&HT', 'config': 'hz_wo_se_ht.yaml', 'description': 'SE✗ TE✓ HT✗'},
            {'name': 'w.o TE&HT', 'config': 'hz_wo_te_ht.yaml', 'description': 'SE✓ TE✗ HT✗'},
            {'name': 'w.o SE', 'config': 'hz_wo_se.yaml', 'description': 'SE✗ TE✓ HT✓'},
            {'name': 'w.o TE', 'config': 'hz_wo_te.yaml', 'description': 'SE✓ TE✗ HT✓'},
            {'name': 'w.o HT', 'config': 'hz_wo_ht.yaml', 'description': 'SE✓ TE✓ HT✗'},
        ]
        
        self.results = []
        
    def run_single_experiment(self, exp_config):
        """运行单个消融实验"""
        exp_name = exp_config['name']
        config_file = exp_config['config']
        
        print(f"\n{'='*60}")
        print(f"开始运行消融实验: {exp_name}")
        print(f"配置文件: {config_file}")
        print(f"描述: {exp_config['description']}")
        print(f"{'='*60}")
        
        # 构建命令
        config_path = os.path.join(self.config_dir, config_file)
        log_dir = os.path.join(self.results_dir, f"ablation_{exp_name.replace(' ', '_').replace('.', '').lower()}")
        
        cmd = [
            'python', 'run_model.py',
            '--config', config_path,
            '--log_dir', log_dir,
            '--seed', '42'
        ]
        
        # 运行实验
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.base_dir,
                capture_output=True, 
                text=True, 
                timeout=7200  # 2小时超时
            )
            
            if result.returncode == 0:
                print(f"✓ 实验 {exp_name} 完成成功")
                # 解析结果
                metrics = self.parse_results(log_dir)
                metrics['name'] = exp_name
                metrics['status'] = 'success'
                self.results.append(metrics)
            else:
                print(f"✗ 实验 {exp_name} 失败")
                print(f"错误输出: {result.stderr}")
                self.results.append({
                    'name': exp_name,
                    'status': 'failed',
                    'error': result.stderr[:200]
                })
                
        except subprocess.TimeoutExpired:
            print(f"✗ 实验 {exp_name} 超时")
            self.results.append({
                'name': exp_name,
                'status': 'timeout'
            })
        except Exception as e:
            print(f"✗ 实验 {exp_name} 异常: {e}")
            self.results.append({
                'name': exp_name,
                'status': 'error',
                'error': str(e)
            })
            
        end_time = time.time()
        print(f"实验耗时: {end_time - start_time:.2f} 秒")
        
    def parse_results(self, log_dir):
        """从日志文件中解析实验结果"""
        log_file = os.path.join(log_dir, 'log.txt')
        metrics = {}
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # 查找最终测试结果
            for line in reversed(lines):
                if 'Horizon' in line and 'MAE:' in line:
                    # 解析 "Horizon 01, MAE: 21.79, MAPE: 0.1313, RMSE: 35.32"
                    parts = line.strip().split(',')
                    horizon = parts[0].split()[-1]
                    mae = float(parts[1].split(':')[1].strip())
                    mape = float(parts[2].split(':')[1].strip())
                    rmse = float(parts[3].split(':')[1].strip())
                    
                    if f'mae_{horizon}' not in metrics:
                        metrics[f'mae_{horizon}'] = mae
                        metrics[f'mape_{horizon}'] = mape
                        metrics[f'rmse_{horizon}'] = rmse
                        
                elif 'Average MAE:' in line:
                    # 解析平均结果
                    parts = line.strip().split(',')
                    metrics['avg_mae'] = float(parts[0].split(':')[1].strip())
                    metrics['avg_mape'] = float(parts[1].split(':')[1].strip())
                    metrics['avg_rmse'] = float(parts[2].split(':')[1].strip())
                    break
                    
        except Exception as e:
            print(f"解析结果文件失败: {e}")
            metrics = {'avg_mae': 999.0, 'avg_mape': 999.0, 'avg_rmse': 999.0}
            
        return metrics
        
    def run_all_experiments(self, parse_only=False):
        """运行所有消融实验或仅解析结果"""
        print(f"开始运行 DST_HA 消融实验")
        print(f"共 {len(self.experiments)} 组实验")
        print(f"结果将保存到: {self.results_dir}")
        
        start_time = time.time()
        
        for i, exp_config in enumerate(self.experiments, 1):
            print(f"\n进度: {i}/{len(self.experiments)}")
            if parse_only:
                # 只解析，不训练
                exp_name = exp_config['name']
                log_dir = os.path.join(self.results_dir, f"ablation_{exp_name.replace(' ', '_').replace('.', '').lower()}")
                metrics = self.parse_results(log_dir)
                metrics['name'] = exp_name
                metrics['status'] = 'success' if metrics.get('avg_mae', 999.0) < 999.0 else 'failed'
                self.results.append(metrics)
            else:
                self.run_single_experiment(exp_config)
        
        end_time = time.time()
        print(f"\n所有实验完成! 总耗时: {(end_time - start_time) / 3600:.2f} 小时")
        
        # 生成结果报告
        self.generate_report()
        
    def generate_report(self):
        """生成消融实验结果报告"""
        print("\n生成消融实验结果报告...")
        
        # 创建结果表格
        df_data = []
        for result in self.results:
            if result['status'] == 'success':
                row = {
                    'Model': result['name'],
                    '15-min MAE': result.get('mae_01', 'N/A'),
                    '30-min MAE': result.get('mae_02', 'N/A'), 
                    '45-min MAE': result.get('mae_03', 'N/A'),
                    '60-min MAE': result.get('mae_04', 'N/A'),
                    'Avg MAE': result.get('avg_mae', 'N/A'),
                    'Avg MAPE': result.get('avg_mape', 'N/A'),
                    'Avg RMSE': result.get('avg_rmse', 'N/A')
                }
            else:
                row = {
                    'Model': result['name'],
                    'Status': result['status'],
                    'Error': result.get('error', '')[:50]
                }
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = os.path.join(self.results_dir, f'ablation_results_{timestamp}.csv')
        df.to_csv(csv_file, index=False)
        
        # 打印结果表格
        print("\n" + "="*80)
        print("DST_HA 消融实验结果")
        print("="*80)
        print(df.to_string(index=False))
        print(f"\n结果已保存到: {csv_file}")
        
        # 生成 LaTeX 表格
        self.generate_latex_table(df, timestamp)
        
    def generate_latex_table(self, df, timestamp):
        """生成 LaTeX 格式的表格"""
        latex_file = os.path.join(self.results_dir, f'ablation_table_{timestamp}.tex')
        
        # 只包含成功的实验
        success_df = df[df.get('Status', 'success') == 'success'].copy()
        
        if not success_df.empty:
            latex_content = """
\\begin{table}[htbp]
\\centering
\\caption{DST_HA消融实验结果}
\\label{tab:ablation}
\\begin{tabular}{l|cccc|c}
\\hline
Model & 15-min & 30-min & 45-min & 60-min & Avg MAE \\\\
\\hline
"""
            
            for _, row in success_df.iterrows():
                model_name = row['Model'].replace('&', '\\&')
                if row['Model'] == 'Full':
                    model_name = '\\textbf{' + model_name + '}'
                    
                latex_content += f"{model_name} & {row['15-min MAE']:.2f} & {row['30-min MAE']:.2f} & {row['45-min MAE']:.2f} & {row['60-min MAE']:.2f} & \\textbf{{{row['Avg MAE']:.2f}}} \\\\\n"
                
            latex_content += """\\hline
\\end{tabular}
\\end{table}
"""
            
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
                
            print(f"LaTeX表格已保存到: {latex_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行DST_HA消融实验')
    parser.add_argument('--config_dir', type=str, help='配置文件目录')
    parser.add_argument('--results_dir', type=str, help='结果保存目录')
    parser.add_argument('--experiment', type=str, help='运行单个实验 (如: Full, w.o_SE等)')
    parser.add_argument('--parse_only', action='store_true', help='只解析已有日志并生成表格，不训练')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    ablation = AblationExperiments()
    
    if args.experiment:
        # 运行单个实验
        exp_config = None
        for exp in ablation.experiments:
            if exp['name'] == args.experiment:
                exp_config = exp
                break
                
        if exp_config:
            if args.parse_only:
                exp_name = exp_config['name']
                log_dir = os.path.join(ablation.results_dir, f"ablation_{exp_name.replace(' ', '_').replace('.', '').lower()}")
                metrics = ablation.parse_results(log_dir)
                metrics['name'] = exp_name
                metrics['status'] = 'success' if metrics.get('avg_mae', 999.0) < 999.0 else 'failed'
                ablation.results.append(metrics)
                ablation.generate_report()
            else:
                ablation.run_single_experiment(exp_config)
                ablation.generate_report()
        else:
            print(f"未找到实验: {args.experiment}")
            print("可用实验:", [exp['name'] for exp in ablation.experiments])
    else:
        # 运行所有实验
        ablation.run_all_experiments(parse_only=args.parse_only)


if __name__ == '__main__':
    main() 