
# forward

$silu(x) = \frac{x}{1 + e^{-x}}$

# 求导

$\frac{\partial silu(x)}{\partial x}$

$= \frac{(1 + e^{-x}) - x(-e^{-x})}{(1 + e^{-x})^2}$

$= \frac{1 + e^{-x} + xe^{-x}}{(1 + e^{-x})^2}$

$= \frac{1 + x}{1 + e^{-x}} - \frac{x}{(1 + e^{-x})^2}$

# 链式法则

$\frac{\partial f(silu(X))}{\partial x_{ij}}$

$= \frac{\partial f(silu(X))}{\partial silu(X)_{ij}} * \frac{\partial silu(X)_{ij}}{\partial x_{ij}}$

$= df_{ij} * (\frac{1 + x}{1 + e^{-x}} - \frac{x}{(1 + e^{-x})^2})$