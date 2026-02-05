import os
import sys
import subprocess
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    clear_screen()
    print("==================================================")
    print("          Auto-WZRY 启动菜单")
    print("==================================================")
    print("1. [录制] 录制专家数据 (Record Expert Data)")
    print("   -> 场景: 刚开始玩，或想补充新数据时使用。")
    print("--------------------------------------------------")
    print("2. [训练] 模仿学习 - 从头开始 (BC Training - Scratch)")
    print("   -> 场景: 第一次训练，或想重新利用所有数据训练时使用。")
    print("3. [训练] 模仿学习 - 断点续训 (BC Training - Resume)")
    print("   -> 场景: 有了旧模型，想省时间只学新数据时使用。")
    print("--------------------------------------------------")
    print("4. [微调] PPO 强化学习 - 加载BC模型 (PPO - From BC)")
    print("   -> 场景: BC模型已经不错了，让AI自我进化时使用。")
    print("5. [微调] PPO 强化学习 - 断点续训 (PPO - Resume)")
    print("   -> 场景: PPO训练中断后继续训练时使用。")
    print("--------------------------------------------------")
    print("6. 退出 (Exit)")
    print("==================================================")

def run_command(cmd_args):
    """
    Run command using the current python interpreter (which should be in the conda env)
    """
    try:
        # Use sys.executable to ensure we use the same python interpreter (wzry env)
        # unless it's a direct conda call, but here we want to run python scripts
        # so we replace 'python' with sys.executable
        full_cmd = [sys.executable] + cmd_args
        print(f"\nExecuting: {' '.join(full_cmd)}\n")
        subprocess.run(full_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing command: {e}")
    except KeyboardInterrupt:
        print("\nCommand interrupted.")
    
    print("\nPress Enter to return to menu...")
    input()

def main():
    while True:
        print_menu()
        choice = input("请输入选项 (1-6): ").strip()
        
        if choice == '1':
            print("\n正在启动录制脚本...")
            print("请确保安卓设备已连接，且游戏已打开。")
            print("结束录制请按 'q' 键。\n")
            run_command(['main/record_expert.py'])
            
        elif choice == '2':
            print("\n正在启动 BC 训练 (从头开始)...")
            print("将读取 data/expert_data 下的所有数据。")
            print("默认训练 50 轮。\n")
            run_command(['main/train_bc.py'])
            
        elif choice == '3':
            print("\n请将您的模型文件 (例如 models/bc_model_epoch_50.pth) 拖入此窗口，然后按回车。")
            model_path = input("模型路径: ").strip().strip('"') # Remove quotes if added by drag-drop
            if not model_path:
                continue
            
            print("\n正在启动 BC 训练 (续训)...")
            print(f"将加载 {model_path} 并继续训练 20 轮。\n")
            run_command(['main/train_bc.py', '--resume-path', model_path, '--epochs', '20'])
            
        elif choice == '4':
            print("\n请将您的 BC 模型文件 (例如 models/bc_model_epoch_50.pth) 拖入此窗口，然后按回车。")
            model_path = input("BC模型路径: ").strip().strip('"')
            if not model_path:
                continue
                
            print("\n正在启动 PPO 训练 (加载 BC 模型)...\n")
            run_command(['main/train.py', '--pretrained-path', model_path])
            
        elif choice == '5':
            print("\n请将您的 PPO 模型文件 (例如 models/ppo_wzry_epoch_10.pth) 拖入此窗口，然后按回车。")
            model_path = input("PPO模型路径: ").strip().strip('"')
            if not model_path:
                continue
                
            print("\n正在启动 PPO 训练 (续训)...\n")
            run_command(['main/train.py', '--resume-path', model_path])
            
        elif choice == '6':
            print("Exiting...")
            sys.exit(0)
            
        else:
            print("无效的选项，请重新输入。")
            time.sleep(1)

if __name__ == "__main__":
    main()
