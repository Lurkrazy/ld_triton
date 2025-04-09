# forward
## 元素表示
<p>
$f_{q,k}(x_m, m) = R^{d}_{m,Θ}W_{q,k}x_m$
</p>

<p>
$f_{q}(x_m, m) = R^{d}_{m,Θ}W_{q}x_m$
</p>

<p>
$f_{k}(x_n, n) = R^{d}_{n,Θ}W_{k}x_n$
</p>

矩阵乘法结合律

$f(q_m) = R^{d}_{m,Θ}q_m$

$f(k_n) = R^{d}_{n,Θ}k_n$

## 原始论文

$$
R_{n,Θ}^{d} = \begin{bmatrix}
cos(m\theta_1) & -sin(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
sin(m\theta_1) & cos(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & cos(m\theta_2) & -sin(m\theta_2) & \cdots & 0 & 0 \\
0 & 0 & sin(m\theta_2) & cos(m\theta_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & cos(m\theta_{d/2}) & -sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \cdots & sin(m\theta_{d/2}) & cos(m\theta_{d/2}) \\
\end{bmatrix}
$$


$$
R_{n,Θ}^{d}x =  \begin{bmatrix}
x_{1} \\
x_{2} \\
x_{3} \\
x_{4} \\
\vdots \\
x_{d-1} \\
x_{d} \\
\end{bmatrix} * \begin{bmatrix}
cos(m\theta_1) \\
cos(m\theta_1) \\
cos(m\theta_2) \\
cos(m\theta_2) \\
\vdots \\
cos(m\theta_{d/2}) \\
cos(m\theta_{d/2}) \\
\end{bmatrix} + \begin{bmatrix}
-x_{2} \\
x_{1} \\
-x_{4} \\
x_{3} \\
\vdots \\
-x_{d} \\
x_{d-1} \\
\end{bmatrix} * \begin{bmatrix}
sin(m\theta_1) \\
sin(m\theta_1) \\
sin(m\theta_2) \\
sin(m\theta_2) \\
\vdots \\
sin(m\theta_{d/2}) \\
sin(m\theta_{d/2}) \\
\end{bmatrix}
$$

$θ_i = 10000^{−2(i−1)/d}, i \in [1, 2, ..., d/2]$

## 原始论文等价
$$
R_{n,Θ}^{d} = \begin{bmatrix}
cos(m\theta_0) & -sin(m\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
sin(m\theta_0) & cos(m\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & cos(m\theta_2) & -sin(m\theta_2) & \cdots & 0 & 0 \\
0 & 0 & sin(m\theta_2) & cos(m\theta_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & cos(m\theta_{d-2}) & -sin(m\theta_{d-2}) \\
0 & 0 & 0 & 0 & \cdots & sin(m\theta_{d-2}) & cos(m\theta_{d-2}) \\
\end{bmatrix}
$$

$$
R_{n,Θ}^{d}x =  \begin{bmatrix}
x_{0} \\
x_{1} \\
x_{2} \\
x_{3} \\
\vdots \\
x_{d-2} \\
x_{d-1} \\
\end{bmatrix} * \begin{bmatrix}
cos(m\theta_0) \\
cos(m\theta_0) \\
cos(m\theta_2) \\
cos(m\theta_2) \\
\vdots \\
cos(m\theta_{d-2}) \\
cos(m\theta_{d-2}) \\
\end{bmatrix} + \begin{bmatrix}
-x_{1} \\
x_{0} \\
-x_{3} \\
x_{2} \\
\vdots \\
-x_{d-1} \\
x_{d-2} \\
\end{bmatrix} * \begin{bmatrix}
sin(m\theta_0) \\
sin(m\theta_0) \\
sin(m\theta_2) \\
sin(m\theta_2) \\
\vdots \\
sin(m\theta_{d-2}) \\
sin(m\theta_{d-2}) \\
\end{bmatrix}
$$

$θ_i = 10000^{−i/d}, i \in [0, 2, ..., d-2]$

## llama实现

$$
R_{n,Θ}^{d} = \begin{bmatrix}
cos(m\theta_0) & 0 & \cdots & 0 & 0  & \cdots & -sin(m\theta_0) & 0 & \cdots & 0 & 0  \\
0 & cos(m\theta_2) & \cdots & 0 & 0  & \cdots & 0 & -sin(m\theta_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
0 & 0 & \cdots & cos(m\theta_{d-4}) & 0  & \cdots & 0 & 0 & \cdots & -sin(m\theta_{d-4}) & 0 \\
0 & 0 & \cdots & 0 & cos(m\theta_{d-2})  & \cdots & 0 & 0 & \cdots & 0 & -sin(m\theta_{d-2}) \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
sin(m\theta_0) & 0 & \cdots & 0 & 0 & \cdots & cos(m\theta_{0}) & 0 & \cdots & 0 & 0 \\
0 & sin(m\theta_2) & \cdots & 0 & 0 & \cdots & 0 & cos(m\theta_{2}) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
0 & 0 & \cdots & sin(m\theta_{d-4}) & 0 & \cdots & 0 & 0 & \cdots & cos(m\theta_{d-4}) & 0 \\
0 & 0 & \cdots & 0 & sin(m\theta_{d-2}) & \cdots & 0 & 0 & \cdots & 0 & cos(m\theta_{d-2}) \\
\end{bmatrix}
$$

$$
R_{n,Θ}^{d}x =  \begin{bmatrix}
x_{0} \\
x_{1} \\
\vdots \\
x_{d/2-2} \\
x_{d/2-1} \\
x_{d/2} \\
x_{d/2+1} \\
\vdots \\
x_{d-2} \\
x_{d-1} \\
\end{bmatrix} * \begin{bmatrix}
cos(m\theta_0) \\
cos(m\theta_2) \\
\vdots \\
cos(m\theta_{d-4}) \\
cos(m\theta_{d-2}) \\
cos(m\theta_{0}) \\
cos(m\theta_{2}) \\
\vdots \\
cos(m\theta_{d-4}) \\
cos(m\theta_{d-2}) \\
\end{bmatrix} + \begin{bmatrix}
-x_{d/2} \\
-x_{d/2 + 1} \\
\vdots \\
-x_{d-2} \\
-x_{d-1} \\
x_{0} \\
x_{1} \\
\vdots \\
x_{d/2-2} \\
x_{d/2-1} \\
\end{bmatrix} * \begin{bmatrix}
sin(m\theta_0) \\
sin(m\theta_2) \\
\vdots \\
sin(m\theta_{d-4}) \\
sin(m\theta_{d-2}) \\
sin(m\theta_0) \\
sin(m\theta_2) \\
\vdots \\
sin(m\theta_{d-4}) \\
sin(m\theta_{d-2}) \\
\end{bmatrix}
$$

$θ_i = 10000^{−i/d}, i \in [0, 2, ..., d-2]$

## 矩阵表示

$f(Q) = Q(R^{d}_{m,Θ})^T$

$f(K) = K(R^{d}_{m,Θ})^T$

# 文献
[ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864)