# 10. 基础数学补完计划：微积分的前置技能

> **写在前面**：
> 在推导神经网络的反向传播时，很多同学卡在了“二项式展开”和“链式法则”上。
> 这篇文档是微积分的“前传”，专门补全高中/大一数学中被遗忘的拼图。
> 掌握了这些，你再去读 [09_calculus_review.md](09_calculus_review.md) 就会如履平地。

---

## 1. 二项式定理 (Binomial Theorem)

### 1.1 什么是二项式展开？

简单来说，就是把 $(a+b)^n$ 这种式子展开成一长串多项式。

我们先看最简单的例子：

*   **1次方**: $(a+b)^1 = a + b$
*   **2次方**: $(a+b)^2 = a^2 + 2ab + b^2$
*   **3次方**: $(a+b)^3 = a^3 + 3a^2b + 3ab^2 + b^3$

### 1.2 通项公式

牛顿告诉我们要学会找规律。如果你把系数单独拿出来，会发现著名的 **杨辉三角 (Pascal's Triangle)**：

```
      1
     1 1
    1 2 1     (对应 n=2: 1*a^2 + 2*ab + 1*b^2)
   1 3 3 1    (对应 n=3: 1*a^3 + 3*a^2b + 3*ab^2 + 1*b^3)
  1 4 6 4 1
```

对于任意 $n$，公式如下（$C_n^k$ 是组合数）：

$$
(a+b)^n = C_n^0 a^n + C_n^1 a^{n-1}b + C_n^2 a^{n-2}b^2 + \dots + b^n
$$

### 1.3 为什么它对 AI 很重要？

在求导数 $x^n$ 时，我们需要计算 $(x + \Delta x)^n$。
让我们把 $a$ 换成 $x$，把 $b$ 换成 $\Delta x$：

$$
(x + \Delta x)^n = x^n + n \cdot x^{n-1} \cdot \Delta x + \frac{n(n-1)}{2} x^{n-2}(\Delta x)^2 + \dots
$$

**关键点来了**：
在微积分里，$\Delta x$ 是一个趋近于 0 的极小值。
*   $\Delta x$ 很小。
*   $(\Delta x)^2$ 比 $\Delta x$ 小得多（高阶无穷小）。
*   $(\Delta x)^3$ 更是微乎其微。

所以，在求极限时，我们通常**只关心前两项**：

$$
(x + \Delta x)^n \approx x^n + n x^{n-1} \Delta x
$$

这就是为什么 $(x^n)' = n x^{n-1}$ 的根本原因！

---

## 2. 复合函数 (Composite Function)

### 2.1 洋葱的比喻

复合函数就是“函数套函数”，像洋葱一样一层包一层。

设想有两个工厂：
*   **工厂 g**: 输入 $x$，产出 $u = g(x)$。
*   **工厂 f**: 输入 $u$，产出 $y = f(u)$。

把它们串联起来，就是复合函数：
$$
y = f(g(x))
$$

### 2.2 例子

假设：
*   内层 $g(x) = x^2 + 1$
*   外层 $f(u) = \sin(u)$

那么复合后：
$$
y = \sin(x^2 + 1)
$$

在神经网络中，每一层都在做复合：
$$
y = \text{Activation}(\text{Linear}(x))
$$

---

## 3. 链式法则 (Chain Rule)

### 3.1 核心直觉：齿轮传动

这是深度学习**最重要**的法则，没有之一。反向传播（Backpropagation）本质上就是链式法则的无限套娃。

想象三个齿轮咬合在一起：
*   **齿轮 A (x)** 转动。
*   **齿轮 B (u)** 被 A 带动。
*   **齿轮 C (y)** 被 B 带动。

如果我们知道：
1.  A 转 1 圈，B 转 2 圈 ($\frac{du}{dx} = 2$)
2.  B 转 1 圈，C 转 3 圈 ($\frac{dy}{du} = 3$)

那么请问：**A 转 1 圈，C 转几圈？**

答案显然是 $2 \times 3 = 6$ 圈。

### 3.2 数学公式

这就是链式法则：

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

**变化率是可以相乘的！**

### 3.3 神经网络中的应用

回顾我们的 Loss 计算过程：
$$
x \xrightarrow{w} h \xrightarrow{\text{sigmoid}} y_{pred} \xrightarrow{\text{MSE}} Loss
$$

如果你想求 $Loss$ 对 $w$ 的导数（梯度），你就需要把路径上所有的变化率乘起来：

$$
\frac{\partial Loss}{\partial w} = \frac{\partial Loss}{\partial y_{pred}} \cdot \frac{\partial y_{pred}}{\partial h} \cdot \frac{\partial h}{\partial w}
$$

这就是我们在 [08_hand_craft_backprop.md](08_hand_craft_backprop.md) 里做的事情。

---

## 4. 动手实验室

我们准备了一个 Python 脚本，用 SymPy 库来帮你：
1.  自动展开复杂的二项式。
2.  验证链式法则。

运行方式：
```bash
python main/debug/math_foundation_lab.py
```
