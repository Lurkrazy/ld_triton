# forward

$Sigmoid(x) = \frac {1} {1 + exp(-x)}$

# backward
## 通用求导

$\frac {\partial exp(x)}{\partial x} = exp(x)$

$\frac {\partial \frac {1} {x}}{\partial x} = - \frac {1} {x^2}$

## x求导

$\frac {\partial Sigmoid(x)}{\partial x}$

$=- \frac {1} {(1 + exp(-x))^2} . \frac {\partial exp(-x)}{\partial x}$

$=\frac {exp(-x)} {(1 + exp(-x))^2}$

## 链式法则

$\frac{\partial f(Sigmoid(x))}{\partial x}$

$=\frac{\partial f(Sigmoid(x))}{\partial Sigmoid(x)} . \frac{\partial Sigmoid(x)}{\partial x}$

$=df . \frac {exp(-x)} {(1 + exp(-x))^2}$