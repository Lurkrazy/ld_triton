
flashattentionv2 的理论推导有一些错误,重新推导了一遍。
第四页 $O^{(2)}$ 的公式是错误的应该是

$O^{(2)} = diag(l^{(2)}/(l^{(1)} * e^{m_{2}-m_{1}}))^{-1}O^{(1)} + \bar P^{(2)}V^{(2)}$

第5页，$O^{(2)}$ 的说法也是错误的

# forward

## attention矩阵形式
$Q, K, V \in R^{N \times d}$

$O \in R^{N \times d}$

$S = QK^{T} \in R^{N \times N}$

$P = softmax(S) \in R^{N \times N}$

$O = PV  \in R^{N \times d}$

## attention元素形式

$S_{ij} = \sum q_{ix}k_{jx}$

$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx})$

$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} \frac{exp(S_{ix})}{L_{i}}v_{xj} = \sum _{x} \frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}}v_{xj}$

## attention实现形式

$S_{ij} = \sum q_{ix}k_{jx}$

$M_{i} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1x}, ......, \sum_{x} q_{ix}k_{Nx})$

$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx}-M_{i})$

$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} \frac{exp(S_{ix}-M_{i})}{L_{i}}v_{xj} = \sum _{x} \frac{exp(\sum_{y} q_{iy}k_{xy}-M_{i})}{L_{i}}v_{xj}$

## flash形式

$S_{ij} = \sum q_{ix}k_{jx}$

$M_{in} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1x}, ......, \sum_{x} q_{ix}k_{nx})$

$L_{in} = \sum _{j=0}^{n}exp(S_{ij}) = \sum _{j=0}^{n} exp(\sum _{x}q_{ix}k_{jx}-M_{in})$

$\bar o_{i,j,n} = \sum _{x=0}^{n} \frac{exp(S_{ix}-M_{in})}{L_{in}}v_{xj} = \sum _{x=0}^{n} \frac{exp(\sum_{y} q_{iy}k_{xy}-M_{in})}{L_{in}}v_{xj}$

$M_{i,n+1} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1x}, ......, \sum_{x} q_{ix}k_{nx}, \sum_{x} q_{ix}k_{n+1,x}) = max(M_{in}, \sum_{x} q_{ix}k_{n+1,x})$

$L_{i,n+1} = \sum _{j=0}^{n+1}exp(S_{ij})$

$= \sum _{j=0}^{n+1} exp(\sum _{x}q_{ix}k_{jx}-M_{i,n+1})$

$= \sum _{j=0}^{n} exp(\sum _{x}q_{ix}k_{jx}-M_{i,n+1}) + exp(\sum _{x}q_{ix}k_{n+1,x}-M_{i,n+1}) $

$= exp(M_{i, n} - M_{i, n+1})\sum _{j=0}^{n}exp(\sum _{x}q_{ix}k_{jx}-M_{i,n}) + exp(\sum _{x}q_{ix}k_{n+1,x}-M_{i,n+1})$

$= exp(M_{i, n} - M_{i, n+1})L_{in}+ exp(\sum _{x}q_{ix}k_{n+1,x}-M_{i,n+1})$

$\bar o_{i,j,n+1}$

$= \sum _{x=0}^{n+1} \frac{exp(\sum_{y} q_{iy}k_{xy}-M_{i,n+1})}{L_{i,n+1}}v_{xj}$

$= \sum _{x=0}^{n} \frac{exp(\sum_{y} q_{iy}k_{xy}-M_{i,n+1})}{L_{i,n+1}}v_{xj} + \frac{exp(\sum_{y} q_{iy}k_{n+1,y}-M_{i,n+1})}{L_{i,n+1}}v_{n+1,j}$


$= \sum _{x=0}^{n} \frac{L_{i,n}}{L_{i,n+1}} \frac{exp(\sum_{y} q_{iy}k_{xy}-M_{i,n+1})}{L_{i,n}}v_{xj} + \frac{exp(\sum_{y} q_{iy}k_{n+1,y}-M_{i,n+1})}{L_{i,n+1}}v_{n+1,j}$

$= \sum _{x=0}^{n} \frac{L_{i,n}}{L_{i,n+1}} \frac{exp(M_{i,n}-M_{i,n+1})exp(\sum_{y} q_{iy}k_{xy}-M_{i,n})}{L_{i,n}}v_{xj} + \frac{exp(\sum_{y} q_{iy}k_{n+1,y}-M_{i,n+1})}{L_{i,n+1}}v_{n+1,j}$

$= \sum _{x=0}^{n} \frac{L_{i,n}exp(M_{i,n}-M_{i,n+1})}{L_{i,n+1}} \frac{exp(\sum_{y} q_{iy}k_{xy}-M_{i,n})}{L_{i,n}}v_{xj} + \frac{exp(\sum_{y} q_{iy}k_{n+1,y}-M_{i,n+1})}{L_{i,n+1}}v_{n+1,j}$

$= \frac{L_{i,n}exp(M_{i,n}-M_{i,n+1})}{L_{i,n+1}} \bar o_{i,j,n} + \frac{exp(\sum_{y} q_{iy}k_{n+1,y}-M_{i,n+1})}{L_{i,n+1}}v_{n+1,j}$