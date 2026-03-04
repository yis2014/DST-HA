@echo off
REM DST_HA 消融实验运行脚本 (Windows)
REM 用法: run_ablation.bat [experiment_name]

setlocal enabledelayedexpansion

REM 项目根目录
set "BASE_DIR=%~dp0.."
cd /d "%BASE_DIR%"

REM 配置
set "CONFIG_DIR=%BASE_DIR%\config\ablation"
set "RESULTS_DIR=%BASE_DIR%\results\ablation"

REM 确保结果目录存在
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo 开始运行 DST_HA 消融实验...

REM 实验1: Full Model
echo ==========================================
echo 运行实验: Full Model
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_full"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_full.yaml" --log_dir "%log_dir%" --seed 42

REM 实验2: w.o SE&TE&HT
echo ==========================================
echo 运行实验: w.o SE^&TE^&HT
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_wo_se_te_ht"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_wo_se_te_ht.yaml" --log_dir "%log_dir%" --seed 42

REM 实验3: w.o SE&TE
echo ==========================================
echo 运行实验: w.o SE^&TE
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_wo_se_te"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_wo_se_te.yaml" --log_dir "%log_dir%" --seed 42

REM 实验4: w.o SE&HT
echo ==========================================
echo 运行实验: w.o SE^&HT
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_wo_se_ht"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_wo_se_ht.yaml" --log_dir "%log_dir%" --seed 42

REM 实验5: w.o TE&HT
echo ==========================================
echo 运行实验: w.o TE^&HT
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_wo_te_ht"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_wo_te_ht.yaml" --log_dir "%log_dir%" --seed 42

REM 实验6: w.o SE
echo ==========================================
echo 运行实验: w.o SE
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_wo_se"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_wo_se.yaml" --log_dir "%log_dir%" --seed 42

REM 实验7: w.o TE
echo ==========================================
echo 运行实验: w.o TE
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_wo_te"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_wo_te.yaml" --log_dir "%log_dir%" --seed 42

REM 实验8: w.o HT
echo ==========================================
echo 运行实验: w.o HT
echo ==========================================
set "log_dir=%RESULTS_DIR%\ablation_wo_ht"
if not exist "%log_dir%" mkdir "%log_dir%"
python main.py --config "%CONFIG_DIR%\hz_wo_ht.yaml" --log_dir "%log_dir%" --seed 42

echo ==========================================
echo 所有消融实验完成!
echo 结果保存在: %RESULTS_DIR%
echo ==========================================
pause 