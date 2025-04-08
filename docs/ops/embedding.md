# forward

$weight \in R^{MN}$

$input \in R^{K}$

$output = weight[input]$

$output_{i} = weight_{input_{i}}$

$output_{i,j} = weight_{input_{i},j}$

# backward
## 求导
$input$ 不可导，只对$ weight$ 求导

$\frac {output_{p,q}}{\partial weight_{i,j}}$

$i \neq input_{p}$

$\frac {output_{p,q}}{\partial weight_{i,j}} = 0$

$i = input_{p} , q \neq j$

$\frac {output_{p,q}}{\partial weight_{i,j}} = 0$

$i = input_{p},q = j$

$\frac {output_{p,q}}{\partial weight_{i,j}} = 1$

## 链式法则

$\frac{f(embedding(weight))}{weight_{i,j}}$

$= \sum_{p=0}^{M-1} \sum_{q=0}^{N-1}\frac{f(embedding(weight))}{output_{p,q}} * \frac{output_{p,q}}{weight_{i,j}}$

$= \sum_{p=0}^{M-1} \sum_{q=0}^{N-1} df_{pq} * \frac{output_{p,q}}{weight_{i,j}}$

$= \sum_{p=0}^{M-1}  df_{pj} * \frac{output_{p,j}}{weight_{i,j}}$

$= \sum_{input_{p}=i}  df_{pj}$
