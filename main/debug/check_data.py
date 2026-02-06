
import os
import json
import sys
import numpy as np

def check_data_quality():
    data_dir = "data/expert_data"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not json_files:
        print("No data files found.")
        return

    print(f"Found {len(json_files)} episodes. Analyzing...\n")
    print(f"{'Episode':<15} | {'Frames':<8} | {'Actions':<8} | {'Active Ratio':<12} | {'Status'}")
    print("-" * 65)

    total_frames = 0
    total_active_frames = 0

    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            frames = sorted(data["frames"], key=lambda x: x["ts"])
            actions = sorted(data["actions"], key=lambda x: x["timestamp"])
            
            # 统计具体的动作类型
            num_down = sum(1 for a in actions if a["type"] == "down")
            num_move = sum(1 for a in actions if a["type"] == "move")
            num_up = sum(1 for a in actions if a["type"] == "up")
            
            # 模拟训练时的状态机逻辑
            action_idx = 0
            is_pressed = False
            active_count = 0
            
            # 检查是否有起始动作缺失 (Episode 开始前就已经按下)
            # 这种情况下，整个 Episode 可能都没有 DOWN 事件，只有 MOVE/UP，或者全空
            has_down_event = any(a["type"] == "down" for a in actions)
            
            for frame in frames:
                ts = frame["ts"]
                while action_idx < len(actions) and actions[action_idx]["timestamp"] <= ts:
                    act = actions[action_idx]
                    if act["type"] == "down" or act["type"] == "move":
                        is_pressed = True
                    elif act["type"] == "up":
                        is_pressed = False
                    action_idx += 1
                
                if is_pressed:
                    active_count += 1
            
            ratio = active_count / len(frames) if frames else 0
            
            status = "✅ OK"
            if ratio < 0.1:
                status = "⚠️ Low Activity"
            if not has_down_event and actions:
                status = "❌ No DOWN Event"
            if not actions:
                status = "💤 No Actions"

            # print(f"{filename:<15} | {len(frames):<8} | {len(actions):<8} | {ratio:.1%}      | {status}")
            print(f"{filename:<15} | {len(frames):<8} | D:{num_down} M:{num_move} U:{num_up} | {ratio:.1%}      | {status}")
            
            total_frames += len(frames)
            total_active_frames += active_count
            
        except Exception as e:
            print(f"{filename:<15} | Error: {e}")

    print("-" * 65)
    avg_ratio = total_active_frames / total_frames if total_frames else 0
    print(f"\nTotal Frames: {total_frames}")
    print(f"Overall Active Ratio: {avg_ratio:.1%}")
    print("\n[分析说明]")
    print("Active Ratio (有效操作率) = 按下状态持续时间 / 总时间")
    print("Check Data 逻辑: 只要检测到 DOWN 或 MOVE 事件，后续帧都视为 Active，直到遇到 UP。")
    print("因此，长按移动 (Hold) 会被正确计算为 100% Active。")
    
    if avg_ratio < 0.2:
        print("\n⚠️  警告: 有效操作比例过低！(< 20%)")
        print("可能原因：")
        print("1. **操作窗口错误**：请务必在 'Expert Recorder' 窗口（显示红点的那个）内操作鼠标，而不是在 Scrcpy 或手机上操作。")
        print("2. **丢失 Move 事件**：如果 Actions 显示 M:0，说明录制器完全没有收到鼠标移动信号。")
        print("3. **录制姿势**：必须先按 R 开始录制，再按下鼠标。")
        print("\n建议：")
        print("请删除 data/expert_data 下的旧数据，并在 'Expert Recorder' 窗口内重新录制。")

if __name__ == "__main__":
    check_data_quality()
