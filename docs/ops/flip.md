
# forward
$input \in R^{M \times N} $

$output_{i,j} = input_{i,N-1-q}$

# 求导

$\frac {\partial output_{p,q}}{\partial input_{i, j}}$

## $p \neq i$

$\frac {\partial output_{p,q}}{\partial input_{i, j}} = 0$

## $p = i, j \neq N-1-q$

$\frac {\partial output_{p,q}}{\partial input_{i, j}} = \frac {\partial input_{i,N-1-q}}{\partial input_{i, j}} = 0$

## $p = i, j = N-1-q$

$\frac {\partial output_{p,q}}{\partial input_{i,j}} = \frac {\partial input_{i,N-1-q}}{\partial input_{i, j}} = 1$

# backward
## 元素形式
$\frac {\partial f(output)}{\partial input_{i,j}}$

$=\sum_{p=0}^{M-1} \sum_{q=0}^{N-1} \frac{\partial f(output)}{\partial output_{p,q}} . \frac{\partial output_{p,q}}{input_{i,j}}$

$=\sum_{p=0}^{M-1} \sum_{q=0}^{N-1} df_{p,q} . \frac{\partial output_{p,q}}{input_{i,j}}$

$=\sum_{q=0}^{N-1} df_{i,q} . \frac{\partial output_{i,q}}{input_{i,j}}$

$=df_{i,N-1-j} . \frac{\partial output_{i,N-1-j}}{input_{i,j}}$

$=df_{i,N-1-j}$

## 矩阵形式
$df_{input} = flip(df_{output})$