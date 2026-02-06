
import sys
import os

def check_dependencies():
    try:
        import sympy
        return True
    except ImportError:
        return False

def math_foundation_lab():
    print("==========================================")
    print("   📐  基础数学实验室 (Math Foundation Lab)   ")
    print("==========================================\n")
    
    if not check_dependencies():
        print("❌ 缺少必要的魔法库: sympy")
        print("请运行: pip install sympy")
        return

    import sympy
    from sympy import symbols, expand, diff, sin, cos

    x, y, h = symbols('x y h')
    
    print("--- 1. 二项式展开体验馆 ---")
    print("我们要展开 (x + h)^n，看看 h 的高次项是如何出现的。\n")
    
    for n in [2, 3, 4]:
        expr = (x + h)**n
        expanded = expand(expr)
        print(f"n={n}: (x + h)^{n} = {expanded}")
        
    print("\n🔍 观察：")
    print("注意看第二项总是 n * x^(n-1) * h")
    print("当 h 趋近于 0 时，h^2, h^3 等后面的一长串都可以忽略不计。")
    print("这就是导数公式 (x^n)' = n*x^(n-1) 的来源！")
    
    print("\n" + "-"*40 + "\n")
    
    print("--- 2. 链式法则验证机 ---")
    print("假设复合函数 y = sin(x^2 + 1)")
    print("令 u = x^2 + 1, 则 y = sin(u)")
    
    # 定义函数
    inner_u = x**2 + 1
    outer_y = sin(x) # 这里 x 只是个占位符，实际上是 sin(u)
    
    # 1. 直接求导
    composite_func = sin(x**2 + 1)
    direct_diff = diff(composite_func, x)
    print(f"\n方式 A: 直接对 y=sin(x^2+1) 求导:")
    print(f"Result = {direct_diff}")
    
    # 2. 链式法则求导
    # dy/dx = dy/du * du/dx
    u = symbols('u')
    f_u = sin(u)
    g_x = x**2 + 1
    
    dy_du = diff(f_u, u)
    du_dx = diff(g_x, x)
    
    print(f"\n方式 B: 链式法则分步求导:")
    print(f"dy/du = {dy_du}  (即 cos(u))")
    print(f"du/dx = {du_dx}")
    print(f"相乘  = ({dy_du}) * ({du_dx})")
    
    # 替换 u 回去
    chain_rule_result = (dy_du.subs(u, g_x)) * du_dx
    print(f"替换回 x = {chain_rule_result}")
    
    print("\n✅ 验证结果：", "成功！" if direct_diff == chain_rule_result else "失败！")

if __name__ == "__main__":
    math_foundation_lab()
