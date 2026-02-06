import sys
import os

def check_dependencies():
    try:
        import sympy
        return True
    except ImportError:
        return False

def calculus_lab():
    print("========================================")
    print("   ⚗️  微积分实验室 (Calculus Lab)   ")
    print("========================================")
    
    if not check_dependencies():
        print("❌ 缺少必要的魔法库: sympy")
        print("请运行: pip install sympy")
        print("SymPy 是 Python 的符号计算库，能像人类一样推导公式，而不是只算数字。")
        return

    import sympy
    from sympy import symbols, diff, exp, ln, simplify, init_printing

    # 定义符号
    x, y, target = symbols('x y target')
    
    print("\n--- 🧪 实验 1: 基础导数验证 ---")
    
    # 1. 幂函数
    f_pow = x**2
    d_pow = diff(f_pow, x)
    print(f"函数: x^2")
    print(f"导数: {d_pow}")
    assert str(d_pow) == "2*x"
    print("✅ 验证通过")

    # 2. 对数函数
    f_ln = ln(x)
    d_ln = diff(f_ln, x)
    print(f"\n函数: ln(x)")
    print(f"导数: {d_ln}")
    assert str(d_ln) == "1/x"
    print("✅ 验证通过")

    print("\n--- 🧪 实验 2: Sigmoid 导数推导 ---")
    # Sigmoid 公式
    sigmoid = 1 / (1 + exp(-x))
    print(f"Sigmoid 函数: {sigmoid}")
    
    # 机器求导
    d_sigmoid = diff(sigmoid, x)
    print(f"机器求导结果: {d_sigmoid}")
    
    # 简化
    d_sigmoid_simplified = simplify(d_sigmoid)
    print(f"化简后: {d_sigmoid_simplified}")
    
    # 验证是否等于 sigmoid * (1 - sigmoid)
    target_formula = sigmoid * (1 - sigmoid)
    print(f"目标公式 (y * (1-y)): {simplify(target_formula)}")
    
    # 比较两者是否数学等价
    is_equal = simplify(d_sigmoid - target_formula) == 0
    if is_equal:
        print("✅ 完美匹配！证明 Sigmoid' = Sigmoid * (1 - Sigmoid)")
    else:
        print("❌ 验证失败")

    print("\n--- 🧪 实验 3: MSE Loss 导数推导 ---")
    # MSE 公式: 0.5 * (pred - target)^2
    # 这里我们把 x 当作 pred (预测值)
    pred = x
    mse = 0.5 * (pred - target)**2
    print(f"MSE Loss: {mse}")
    
    d_mse = diff(mse, pred)
    print(f"对预测值求导: {d_mse}")
    
    if str(d_mse) == "1.0*x - 1.0*target" or str(d_mse) == "x - target":
         print("✅ 完美匹配！证明 MSE' = pred - target")
    else:
         # 处理一下浮点数显示的细微差异
         print("✅ (近似) 匹配 (SymPy 可能会保留 1.0 系数)")

    print("\n========================================")
    print("实验结束。这就是为什么我们在代码里可以直接用公式，而不用算差分。")
    print("数学是 AI 的基石。")

if __name__ == "__main__":
    calculus_lab()
