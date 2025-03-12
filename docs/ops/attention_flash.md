
flashattentionv2 的理论推导有一些错误,重新推导了一遍。
第四页 $O^{(2)}$ 的公式是错误的应该是

$O^{(2)} = diag(l^{(2)}/(l^{(1)} * e^{m_{2}-m_{1}}))^{-1}O^{(1)} + \bar P^{(2)}V^{(2)}$

第5页, $O^{(2)}$ 的说法也是错误的

# forward

## attention矩阵形式
$Q, K, V \in R^{N \times d}$

$O \in R^{N \times d}$

$S = QK^{T} \in R^{N \times N}$

$P = softmax(S) \in R^{N \times N}$

$O = PV  \in R^{N \times d}$

## attention元素形式

$S_{ij} = \sum q_{ix}k_{jx}$

<p>
$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx})$
</p>

<p>
$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} \frac{exp(S_{ix})}{L_{i}}v_{xj} = \sum _{x} \frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}}v_{xj}$
</p>

## attention实现形式

$S_{ij} = \sum q_{ix}k_{jx}$

$M_{i} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1x}, ......, \sum_{x} q_{ix}k_{Nx})$

<p>
$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx}-M_{i})$
</p>

<p>
$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} \frac{exp(S_{ix}-M_{i})}{L_{i}}v_{xj} = \sum _{x} \frac{exp(\sum_{y} q_{iy}k_{xy}-M_{i})}{L_{i}}v_{xj}$
</p>

## flash形式

$S_{ij} = \sum q_{ix}k_{jx}$

$M_{i,(a,b)} = max(\sum_{x} q_{ix}k_{ax}, \sum_{x} q_{ix}k_{a+1,x}, ......, \sum_{x} q_{ix}k_{b,x})$

<p>
$L_{i,(a,b)} = \sum _{j=a}^{b}e^{S_{ij}} = \sum _{j=a}^{b} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(a,b)}}$
</p>

$M_{i,(0,n-1)} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1,x}, ......, \sum_{x} q_{ix}k_{n-1,x})$

<p>
$L_{i,(0,n-1)} = \sum _{j=0}^{n-1}e^{S_{ij}} = \sum _{j=0}^{n-1} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n-1)}}$
</p>

$P_{i,x,(a,b)}=\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(a,b)}}}{L_{i,(a,b)}}$

<p>
$O_{i,j,n-1} = \sum _{x=0}^{n-1} \frac{e^{S_{ix}-M_{i,(0,n-1)}}}{L_{i,(0,n-1)}}v_{xj} = \sum _{x=0}^{n-1} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n-1)}}}{L_{i,(0,n-1)}}v_{xj}$
</p>

$M_{i,(0,n+m)} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1x}, ......, \sum_{x} q_{ix}k_{n+m,x}) = max(M_{i,(0,n-1)}, M_{i, (n, n+m)})$

<p>
$L_{i,(0,n+m)} = \sum _{j=0}^{n+m}e^{S_{ij}}$
</p>

<p>
$= \sum _{j=0}^{n+m} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n+m)}}$
</p>

<p>
$= \sum _{j=0}^{n-1} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n+m)}} + \sum _{j=n}^{n+m}e^{\sum _{x}q_{ix}k_{j,x}-M_{i,(0,n+m)}} $
</p>

<p>
$= e^{M_{i, (0,n-1)} - M_{i, (0,n+m)}}\sum _{j=0}^{n-1}e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n-1)}} + \sum _{j=n}^{n+m}e^{\sum _{x}q_{ix}k_{j,x}-M_{i,(0,n+m)}}$
</p>

<p>
$= e^{M_{i, (0,n-1)} - M_{i, (0,n+m)}}\sum _{j=0}^{n-1}e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n)}} + e^{M_{i, (n,n+m)} - M_{i, (0,n+m)}}\sum _{j=n}^{n+m}e^{\sum _{x}q_{ix}k_{j,x}-M_{i,(n,n+m)}}$
</p>

$= e^{M_{i, (0,n-1)} - M_{i, (0,n+m)}}L_{i,(0,n-1)} + e^{M_{i, (n,n+m)} - M_{i, (0,n+m)}}L_{i,(n,n+m)}$

<p>
$O_{i,j,n+m}= \sum _{x=0}^{n+m} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj}$
</p>

<p>
$= \sum _{x=0}^{n-1} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj} + \sum _{x=n}^{n+m}\frac{e^{\sum_{y} q_{iy}k_{x,y}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{x,j}$
</p>

<p>
$= \sum _{x=0}^{n-1} \frac{L_{i,(0,n-1)}}{L_{i,(0,n+m)}} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}}v_{xj} + \sum _{x=n}^{n+m} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj}$
</p>

<p>
$= \sum _{x=0}^{n-1} \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n-1)}}}{L_{i,(0,n-1)}}v_{xj} + \sum _{x=n}^{n+m} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj}$
</p>

<p>
$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \sum _{x=n}^{n+m} \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(n,n+m)}}}{L_{i,(n,n+m)}}v_{xj}$
</p>

<p>
$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \sum _{x=n}^{n+m} \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}P_{i,x,(n,n+m)}v_{xj}$
</p>

<p>
$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}\sum _{x=n}^{n+m} P_{i,x,(n,n+m)}v_{xj}$
</p>

$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}P_{i,(n,n+m)}v_{j}^{T}$

$n+m=N-1$时，
$O_{i,j,N-1} = O_{i,j}$

# backward
## 链式法则
### 矩阵形式
$\frac {\partial f(attention(q))}{\partial q} =(softmax(QK^{T})* (d_{f}V^{T} - sum(softmax(QK^{T}) * (d_{f}V^{T}), dim=-1, keepdim=True)))K$

### 元素形式

<p>
$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx})$
</p>

$\frac {\partial f(attention(q))}{\partial q_{ij}}=\sum_{w}\frac{{exp(\sum_{y} q_{iy}k_{wy})}}{L_{i}}.(\sum_{b} (df_{ib}v_{wb}) - \sum_{x}\frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}} . \sum_{b} (df_{ib}v_{xb})) .k_{wj}$

$\frac {\partial f(attention(k))}{\partial k_{ij}}=\sum_{a} \frac{{exp(\sum_{y} q_{ay}k_{iy})}}{L_{a}} . (\sum_{b} (df_{ab} .v_{xb}) - \sum_{x}{\frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}.\sum_{b} (df_{ab} . v_{xb})) .q_{aj}$

### 实现形式
$M_{i} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1x}, ......, \sum_{x} q_{ix}k_{Nx})$

<p>
$L_{i} = \sum _{j} e^{\sum _{x}q_{ix}k_{jx}-M_{i}}$
</p>

$P_{ij} = \frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i}}}}{L_{i}}$

$dP_{ij}= \sum_{b} df_{ib}v_{jb}$

$D_{i}=\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i}}}{L_{i}} . \sum_{b} (df_{ib}v_{xb})$

$=\sum_{b}(\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i}}}{L_{i}}v_{xb})df_{ib}$

$=\sum_{b}O_{ib}df_{ib}$

$dQ_{i,j} = \frac {\partial f(attention(q))}{\partial q_{ij}}=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i}}}}{L_{i}}.(\sum_{b} (df_{ib}v_{wb}) - \sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i}}}{L_{i}} . \sum_{b} (df_{ib}v_{xb})) .k_{wj}$

$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i}}}}{L_{i}}.(dP_{iw} - D_{i}).k_{wj}$


### flash形式
$M_{i,(c,d)} = max(\sum_{x} q_{ix}k_{cx}, ......, \sum_{x} q_{ix}k_{d,x})$

<p>
$L_{i,(c,d)} = \sum _{j=c}^{d} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(c,d)}}$
</p>

$P_{i,j,(c,d)} = \frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(c,d)}}}}{L_{i,(c,d)}}$

$dP_{i,j,(c,d)}= \sum_{b=c}^{d} df_{ib}v_{jb}$

$D_{i}=\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i}}}{L_{i}} . \sum_{b} (df_{ib}v_{xb}) = \sum_{b}O_{i,b}df_{i,b}$

$M_{i,(0,n-1)} = max(\sum_{x} q_{ix}k_{0x}, ......, \sum_{x} q_{ix}k_{n-1,x})$

<p>
$L_{i,(0,n-1)} = \sum _{j=0}^{n-1} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n-1)}}$
</p>

$P_{i,j,(0,n-1)} = \frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n-1)}}}}{L_{i,(0,n-1)}}$

$dQ_{i,j,n-1} =\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n-1)}}}}{L_{i,(0,n-1)}}.(\sum_{b} (df_{ib}v_{wb}) - \sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n-1)}}}{L_{i,(0,n-1)}} . \sum_{b} (df_{ib}v_{xb})) .k_{wj}$

$=\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n-1)}}}}{L_{i,(0, n-1)}}.(\sum_{b} df_{ib}v_{wb} - D_{i}) .k_{wj}$

$=\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n-1)}}}}{L_{i,(0, n-1)}}.(dP_{i,w,(0,n-1)} - D_{i}) .k_{wj}$

$M_{i,(0,n+m)} = max(\sum_{x} q_{ix}k_{0x}, \sum_{x} q_{ix}k_{1x}, ......, \sum_{x} q_{ix}k_{n-1,x},\sum_{x} q_{ix}k_{n,x}, ......, \sum_{x} q_{ix}k_{n+m,x})$

$=max(M_{i,(0,n-1)}, M_{i,(n,n+m)})$

<p>
$L_{i,(0,n+m)} = \sum _{j=0}^{n+m} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n+m)}}$
</p>

<p>
$=\sum _{j=0}^{n-1} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n+m)}}+\sum _{j=n}^{n+m} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n+m)}}$
</p>

<p>
$=e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}\sum _{j=0}^{n-1} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n-1)}}+\sum _{j=n}^{n+m} e^{\sum _{x}q_{ix}k_{jx}-M_{i,(0,n+m)}}$
</p>

$=e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}L_{i,(0,n-1)} + e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}L_{i,(n,n+m)}$

$dQ_{i,j,n+m}=\sum_{w=0}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n+m)}}}}{L_{i,(0,n+m)}}.(dP_{i,w} - D_{i}) .k_{wj}$

$=\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n+m)}}}}{L_{i,(0,n+m)}}.(dP_{i,w} - D_{i}) .k_{wj} + \sum_{w=n}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n+m)}}}}{L_{i,(0,n+m)}}.(dP_{i,w} - D_{i}) .k_{wj}$

$=\sum_{w=0}^{n-1}\frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n-1)}}}}{L_{i,(0,n-1)}}.(dP_{i,w} - D_{i}) .k_{wj} + \sum_{w=n}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n+m)}}}}{L_{i,(0,n+m)}}.(dP_{i,w} - D_{i}) .k_{wj}$

$=\frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}dQ_{i,j,n-1} + \sum_{w=n}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,n+m)}}}}{L_{i,(0,n+m)}}.(dP_{i,w} - D_{i}) .k_{wj}$

$=\frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}dQ_{i,j,n-1} + \sum_{w=n}^{n+m}\frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}} \frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(n,n+m)}}}}{L_{i,(n,n+m)}}.(dP_{i,w} - D_{i}) .k_{wj}$

#### $dq$ flash形式二

$dQ_{i,j,n-1} =\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}.(P_{iw} - D_{i}) .k_{wj}$

$dQ_{i,j,n+m} =\sum_{w=0}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}.(P_{iw} - D_{i}) .k_{wj}$

$dQ_{i,j,n+m} =\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}.(P_{iw} - D_{i}) .k_{wj} + \sum_{w=n}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}.(P_{iw} - D_{i}) .k_{wj}$

$dQ_{i,j,n+m} =dQ_{i,j,n-1} + \sum_{w=n}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}.(P_{iw} - D_{i}) .k_{wj}$

#### $dk$ flash形式二

$dK_{i,j,n-1} =\sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}} . (\sum_{b} (df_{ab} .v_{xb}) - \sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}-M_{i,(0,N-1)}}}{L_{i,,(0,N-1)}}}.\sum_{b} (df_{ab} . v_{xb})) .q_{aj}$

$=\sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}} . (\sum_{b} (df_{ab} .v_{xb}) - D_{i}) .q_{aj}$

$=\sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}} . (P_{iw} - D_{i}) .q_{aj}$

$dK_{i,j,n+m} = \sum_{a=0}^{n+m} \frac{{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}} . (P_{iw} - D_{i}) .q_{aj}$

$= \sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}} . (P_{iw} - D_{i}) .q_{aj} + \sum_{a=n}^{n+m} \frac{{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}} . (P_{iw} - D_{i}) .q_{aj}$

$= dK_{i,j,n-1} + \sum_{a=n}^{n+m} \frac{{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}} . (P_{iw} - D_{i}) .q_{aj}$


#### $dv$ flash形式二
$\frac {\partial f(attention(v))}{\partial v_{ij}}=\sum_{a} df_{aj} . \frac{exp(\sum_{y} q_{ay}k_{iy})}{L_{a}}$

$dV_{i,j,n-1} = \sum_{a=0}^{n-1} df_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}{L_{i,(0,N-1)}}$

$dV_{i,j,n+m}=\sum_{a=0}^{n+m} df_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}{L_{i,(0,N-1)}}$

$=\sum_{a=0}^{n-1} df_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}{L_{i,(0,N-1)}} + \sum_{a=n}^{n+m} df_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}{L_{i,(0,N-1)}} $

$=dV_{i,j,n-1} + \sum_{a=n}^{n+m} df_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}-M_{i,(0,N-1)}}}{L_{i,(0,N-1)}} $