# forward
$x \in R^{MN}$

<p>
$max(x)_{ij} = max({x_{i,0},...,x_{i,n-1}}) = x_{i,k}$
</p>

# backward
## 通用求导

$\frac {\partial x}{\partial x} = 1$

$\frac {\partial {x}}{\partial y} = 0$

## x求导

<p>
$\frac {\partial max(x)_{pq}}{\partial x_{ij}}$
</p>

<p>
$=\frac {\partial max(x)_{p,k}}{\partial x_{ij}}$
</p>

### $p \neq i$
$\frac {\partial max(x_i)}{\partial x_{pq}}=0$

### $p = i, k \neq j$
$\frac {\partial max(x_i)}{\partial x_{pq}}=0$

### $p = i, k = j$
$\frac {\partial max(x_i)}{\partial x_{pq}}=1$

## 链式法则

$\frac{\partial f(max(x))}{\partial x_{ij}}$

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{N-1}\frac{\partial f(max(x))}{\partial max(x)_{pq}} . \frac{\partial max(x)_{pq}}{\partial x_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{N-1} df_{pq} . \frac{\partial max(x)_{pq}}{\partial x_{ij}}$
</p>

<p>
$=\sum_{q=0}^{N-1}df_{iq} . \frac{\partial max(x)_{iq}}{\partial x_{ij}}$
</p>

<p>
$=\sum_{q=0}^{N-1}df_{iq} . \frac{\partial max(x)_{ik}}{\partial x_{ij}}$
</p>

### $k \neq j$

$\frac{\partial f(max(x))}{\partial x_{ij}} = 0$

### $k = j$

$\frac{\partial f(max(x))}{\partial x_{ij}} = \sum_{q=0}^{N-1}df_{iq}$

