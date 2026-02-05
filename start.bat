@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:MENU
cls
echo ==================================================
echo           Auto-WZRY 启动菜单
echo ==================================================
echo 1. [录制] 录制专家数据 (Record Expert Data)
echo 2. [训练] 模仿学习 - 从头开始 (BC Training - Scratch)
echo 3. [训练] 模仿学习 - 断点续训 (BC Training - Resume)
echo 4. [微调] PPO 强化学习 - 加载BC模型 (PPO - From BC)
echo 5. [微调] PPO 强化学习 - 断点续训 (PPO - Resume)
echo 6. 退出 (Exit)
echo ==================================================
set /p choice=请输入选项 (1-6): 

if "%choice%"=="1" goto RECORD
if "%choice%"=="2" goto BC_SCRATCH
if "%choice%"=="3" goto BC_RESUME
if "%choice%"=="4" goto PPO_FROM_BC
if "%choice%"=="5" goto PPO_RESUME
if "%choice%"=="6" goto EXIT

echo 无效的选项，请重新输入。
pause
goto MENU

:RECORD
echo.
echo 正在启动录制脚本...
echo 请确保安卓设备已连接，且游戏已打开。
echo 结束录制请按 'q' 键。
echo.
conda run --no-capture-output -n wzry python main/record_expert.py
pause
goto MENU

:BC_SCRATCH
echo.
echo 正在启动 BC 训练 (从头开始)...
echo 将读取 data/expert_data 下的所有数据。
echo 默认训练 50 轮。
echo.
conda run --no-capture-output -n wzry python main/train_bc.py
pause
goto MENU

:BC_RESUME
echo.
echo 请将您的模型文件 (例如 models/bc_model_epoch_50.pth) 拖入此窗口，然后按回车。
set /p model_path=模型路径: 
if "%model_path%"=="" goto MENU
echo.
echo 正在启动 BC 训练 (续训)...
echo 将加载 %model_path% 并继续训练 20 轮。
echo.
conda run --no-capture-output -n wzry python main/train_bc.py --resume-path %model_path% --epochs 20
pause
goto MENU

:PPO_FROM_BC
echo.
echo 请将您的 BC 模型文件 (例如 models/bc_model_epoch_50.pth) 拖入此窗口，然后按回车。
set /p model_path=BC模型路径: 
if "%model_path%"=="" goto MENU
echo.
echo 正在启动 PPO 训练 (加载 BC 模型)...
echo.
conda run --no-capture-output -n wzry python main/train.py --pretrained-path %model_path%
pause
goto MENU

:PPO_RESUME
echo.
echo 请将您的 PPO 模型文件 (例如 models/ppo_wzry_epoch_10.pth) 拖入此窗口，然后按回车。
set /p model_path=PPO模型路径: 
if "%model_path%"=="" goto MENU
echo.
echo 正在启动 PPO 训练 (续训)...
echo.
conda run --no-capture-output -n wzry python main/train.py --resume-path %model_path%
pause
goto MENU

:EXIT
exit
