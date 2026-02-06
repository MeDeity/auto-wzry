# 09. 补课：微积分这一关得过 (Calculus Review)

> **"我不想听数学，我只想调包！"**
> 相信我，如果你不懂梯度（Gradient）是怎么算出来的，你永远只是一个"调参侠"，而不是 AI 工程师。
> 这一章，我们把神经网络最核心的数学工具——**导数（Derivative）**，一次性讲透。

---

## 1. 为什么需要求导？

在神经网络中，我们的终极目标是：**让 Loss（误差）变小**。

想象你在山上（Loss 很高），你想下山（Loss 变低）。
如果你蒙着眼睛，你怎么知道往哪个方向走是下坡？
这时候你需要用脚探一探：**"如果我往东走一步，高度是变高还是变低？"**

这个"探一探"的过程，数学上就叫**求导**。
*   导数 $> 0$：往这个方向走，Loss 变大（上坡）。
*   导数 $< 0$：往这个方向走，Loss 变小（下坡）。
*   导数 $= 0$：平地（可能是谷底，也可能是山顶）。

**梯度（Gradient）** 就是所有参数导数的集合，它直接告诉我们：**往哪里跑最快！**

---

## 2. 常用导数速查表 (Cheat Sheet)

不需要背诵整本微积分教材，深度学习常用的就这几个：

| 函数 $f(x)$ | 导数 $f'(x)$ | 备注 |
| :--- | :--- | :--- |
| **常数** $C$ | $0$ | 没有任何变化率 |
| **幂函数** $x^n$ | $n \cdot x^{n-1}$ | $x^2 \to 2x$, $x \to 1$ |
| **指数** $e^x$ | $e^x$ | 世界上最孤独的函数，求导还是自己 |
| **对数** $\ln(x)$ | $\frac{1}{x}$ | |
| **Sigmoid** $\sigma(x)$ | $\sigma(x)(1 - \sigma(x))$ | **反向传播的神技** |
| **ReLU** | $1$ (if $x>0$), $0$ (if $x<0$) | 简单粗暴，极其好用 |
| **加法** $u + v$ | $u' + v'$ | 各算各的 |
| **乘法** $uv$ | $u'v + uv'$ | 前导后不导 + 后导前不导 |

---

## 3. 核心武器：链式法则 (Chain Rule)

这是神经网络能训练的根本原因。
神经网络本质上是一个**复合函数**：

$$
y = f(g(h(x)))
$$

如果你想求 $y$ 对 $x$ 的导数，你不需要把公式展开（那会死人的）。你只需要**一层一层剥洋葱**：

$$
\frac{dy}{dx} = \frac{dy}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}
$$

**口诀**：整体求导 $\times$ 内部求导。

---

## 4. 基础导数推导 (Basic Proofs)

既然我们想把地基打牢，那就不能只记公式。这里我们用**导数的定义**来推导上述所有公式。

**导数的定义**:
函数 $f(x)$ 在 $x$ 处的瞬时变化率。

$$
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

### 4.1 常数法则 (Constant Rule)

证明：$f(x) = C$

$$
f'(x) = \lim_{\Delta x \to 0} \frac{C - C}{\Delta x} = \lim_{\Delta x \to 0} \frac{0}{\Delta x} = 0
$$

**直觉**：一条水平线，无论怎么走，高度都不变，所以坡度为 0。

### 4.2 幂函数法则 (Power Rule)

证明：$f(x) = x^n$

$$
f'(x) = \lim_{\Delta x \to 0} \frac{(x + \Delta x)^n - x^n}{\Delta x}
$$

利用二项式展开：

$$
(x + \Delta x)^n = x^n + n \cdot x^{n-1} \cdot \Delta x + \frac{n(n-1)}{2} x^{n-2} (\Delta x)^2 + \dots
$$

代入极限公式：

$$
= \lim_{\Delta x \to 0} \frac{x^n + n x^{n-1} \Delta x + O(\Delta x^2) - x^n}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} (n x^{n-1} + O(\Delta x))
$$

当 $\Delta x \to 0$，所有含 $\Delta x$ 的项都消失了。

$$
f'(x) = n x^{n-1}
$$

### 4.3 指数函数 (Exponential)

证明：$f(x) = e^x$

$$
f'(x) = \lim_{\Delta x \to 0} \frac{e^{x + \Delta x} - e^x}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{e^x \cdot e^{\Delta x} - e^x}{\Delta x}
$$

$$
= e^x \cdot \lim_{\Delta x \to 0} \frac{e^{\Delta x} - 1}{\Delta x}
$$

数学家告诉我们，自然常数 $e$ 的定义就是让这个极限为 1：$\lim_{h \to 0} \frac{e^h - 1}{h} = 1$。

$$
f'(x) = e^x \cdot 1 = e^x
$$

### 4.4 对数函数 (Logarithm)

证明：$y = \ln(x)$

我们可以利用反函数求导。$x = e^y$。
两边对 $x$ 求导：

$$
1 = e^y \cdot y'
$$

$$
y' = \frac{1}{e^y}
$$

因为 $e^y = x$，所以：

$$
y' = \frac{1}{x}
$$

### 4.5 加法法则 (Sum Rule)

证明：$f(x) = u(x) + v(x)$

$$
f'(x) = \lim_{\Delta x \to 0} \frac{[u(x+\Delta x) + v(x+\Delta x)] - [u(x) + v(x)]}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \left[ \frac{u(x+\Delta x) - u(x)}{\Delta x} + \frac{v(x+\Delta x) - v(x)}{\Delta x} \right]
$$

$$
= u'(x) + v'(x)
$$

### 4.6 乘法法则 (Product Rule)

证明：$f(x) = u(x)v(x)$

$$
f'(x) = \lim_{\Delta x \to 0} \frac{u(x+\Delta x)v(x+\Delta x) - u(x)v(x)}{\Delta x}
$$

**技巧**：加一项减一项 $u(x+\Delta x)v(x)$。

$$
= \lim_{\Delta x \to 0} \frac{u(x+\Delta x)v(x+\Delta x) - u(x+\Delta x)v(x) + u(x+\Delta x)v(x) - u(x)v(x)}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \left[ u(x+\Delta x) \frac{v(x+\Delta x) - v(x)}{\Delta x} + v(x) \frac{u(x+\Delta x) - u(x)}{\Delta x} \right]
$$

当 $\Delta x \to 0$， $u(x+\Delta x) \to u(x)$。

$$
= u(x)v'(x) + v(x)u'(x)
$$

### 4.7 ReLU 函数

定义：
$$
f(x) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}
$$

分段求导：
*   当 $x > 0$ 时，$f(x) = x$，导数是 1。
*   当 $x < 0$ 时，$f(x) = 0$，导数是 0。
*   当 $x = 0$ 时，导数不存在（不可导点），但在工程实现中通常人为定义为 0 或 0.5。

---

## 5. 神经网络核心推导 (Neural Network Proofs)

这里我们应用前面的基础，推导两个最常用的组件。

### 5.1 均方误差 (MSE) 的导数

Loss 函数通常长这样（为了方便，前面乘个 $1/2$）：

$$
L = \frac{1}{2}(y_{pred} - y_{target})^2
$$

我们想求 Loss 对 $y_{pred}$ 的变化率：
令 $u = y_{pred} - y_{target}$，则 $L = \frac{1}{2}u^2$。

根据链式法则：

$$
\frac{\partial L}{\partial y_{pred}} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial y_{pred}}
$$

**1. 外层导数**：$\frac{1}{2}u^2 \to u$

**2. 内层导数**：$(y_{pred} - y_{target}) \to 1$

所以：

$$
\frac{\partial L}{\partial y_{pred}} = u \cdot 1 = y_{pred} - y_{target}
$$

**结论**：MSE 的梯度就是**误差本身**（预测值 - 真实值）。这也太符合直觉了！误差越大，梯度越大，调整力度越大。

### 5.2 Sigmoid 函数的导数

Sigmoid 公式：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

这个函数看起来很复杂，但它的导数却异常优雅。
为了方便书写，记 $y = \sigma(x)$。

$$
y = (1 + e^{-x})^{-1}
$$

**推导步骤**：

**1. 链式法则外层** ($u^{-1}$):

$$
-1 \cdot (1 + e^{-x})^{-2}
$$

$$
= \frac{-1}{(1 + e^{-x})^2}
$$

**2. 链式法则内层** ($1 + e^{-x}$):

常数 1 导数是 0。
$e^{-x}$ 的导数是 $e^{-x} \cdot (-1) = -e^{-x}$。

**3. 合体**:

$$
\frac{dy}{dx} = \frac{-1}{(1 + e^{-x})^2} \cdot (-e^{-x})
$$

$$
= \frac{e^{-x}}{(1 + e^{-x})^2}
$$

**4. 化简魔法**（凑出 $y$ 的形式）:

$$
= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}}
$$

$$
= y \cdot \frac{e^{-x}}{1 + e^{-x}}
$$

注意分子 $e^{-x}$ 可以写成 $(1 + e^{-x}) - 1$：

$$
= y \cdot \frac{(1 + e^{-x}) - 1}{1 + e^{-x}}
$$

$$
= y \cdot (1 - \frac{1}{1 + e^{-x}})
$$

$$
= y \cdot (1 - y)
$$

**结论**：

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

这就是为什么我们在代码里可以直接写 `h * (1 - h)`，而不需要重新算一遍复杂的指数运算。数学家帮我们省了多少算力！

---

## 6. 动手验证

光看不练假把式。我们提供了一个 Python 脚本，使用 `SymPy` 库（Python 的符号计算库）来自动推导验证上述公式。

运行：
```bash
python main/debug/calculus_lab.py
```

你将看到计算机是如何像数学家一样推导公式的。
