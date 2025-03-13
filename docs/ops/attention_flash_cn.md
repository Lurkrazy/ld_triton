
flashattentionv2 的理论推导有一些错误,重新推导了一遍。
第四页 $O^{(2)}$ 的公式是错误的应该是

$O^{(2)} = diag(l^{(2)}/(l^{(1)} * e^{m_{2}-m_{1}}))^{-1}O^{(1)} + \bar P^{(2)}V^{(2)}$

第5页, $O^{(2)}$ 的说法也是错误的

# forward

## attention矩阵形式
$Q, K, V \in R^{N \times d}$

$O \in R^{N \times d}$

$S = QK^{T} \in R^{N \times N}$

$R = scale$

$P = softmax(S*R) \in R^{N \times N}$

$O = PV  \in R^{N \times d}$

## attention元素形式

$S_{ij} = \sum_{x} q_{ix}k_{jx}r_{ij}$

<p>
$L_{i} = \sum _{j}e^{S_{ij}} = \sum _{j} e^{\sum _{x}q_{ix}k_{jx}r_{ij}}$
</p>

<p>
$p_{ix}=softmax(QK^{T}*R)_{ix} = \frac{e^{S_{ij}}}{L_{i}}=\frac{e^{\sum_{x} q_{ix}k_{jx}r_{ij}}}{L_{i}}$
</p>

<p>
$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} \frac{e^{S_{ix}}}{L_{i}}v_{xj} = \sum _{x} \frac{e^{\sum_{y} q_{iy}k_{xy}}}{L_{i}}v_{xj}$
</p>

## attention实现形式

$S_{ij} = \sum_{x} q_{ix}k_{jx}r_{ij}$

$M_{i} = max(\sum_{x} q_{ix}k_{0x}r_{i,0}, \sum_{x} q_{ix}k_{1x}r_{i,1}, ......, \sum_{x} q_{ix}k_{N-1,x}r_{i,N-1})$

<p>
$L_{i} = \sum_{j}e^{S_{ij}-M_{i}} = \sum _{j} e^{\sum _{x}q_{ix}k_{jx}r_{ij}-M_{i}}$
</p>

<p>
$p_{ix}=softmax(QK^{T}*R)_{ix} = \frac{e^{S_{ij}-M_{i}}}{L_{i}}=\frac{e^{\sum_{x} q_{ix}k_{jx}r_{ij}-M_{i}}}{L_{i}}$
</p>

<p>
$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} \frac{e^{S_{ix}-M_{i}}}{L_{i}}v_{xj} = \sum _{x} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i}}}{L_{i}}v_{xj}$
</p>

## flash形式

$S_{ij} = \sum_{x} q_{ix}k_{jx}r_{ij}$

$M_{i,(a,b)} = max(\sum_{x} q_{ix}k_{ax}r_{ia}, \sum_{x} q_{ix}k_{a+1,x}r_{i,a+1}, ......, \sum_{x} q_{ix}k_{b,x}r_{ib})$

<p>
$L_{i,(a,b)} = \sum _{j=a}^{b}e^{S_{ij}-M_{i,(a,b)}} = \sum _{j=a}^{b} e^{\sum _{x}q_{ix}k_{jx}r_{ij}-M_{i,(a,b)}}$
</p>

$P_{i,x,(a,b)}=\frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(a,b)}}}{L_{i,(a,b)}}$

$M_{i,(0,n-1)} = max(\sum_{x} q_{ix}k_{0x}r_{i0}, \sum_{x} q_{ix}k_{1x}r_{i1}, ......, \sum_{x} q_{ix}k_{n-1,x}r_{i,n-1})$

<p>
$L_{i,(0,n-1)} = \sum _{j=0}^{n-1}e^{S_{ij}-M_{i,(0,n-1)}} = \sum _{j=0}^{n-1} e^{\sum _{x}q_{ix}k_{jx}r_{ij}-M_{i,(0,n-1)}}$
</p>

<p>
$O_{i,j,n-1} = \sum _{x=0}^{n-1} \frac{e^{S_{ix}-M_{i,(0,n-1)}}}{L_{i,(0,n-1)}}v_{xj} = \sum _{x=0}^{n-1} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,n-1)}}}{L_{i,(0,n-1)}}v_{xj}$
</p>

$M_{i,(0,n+m)} = max(\sum_{x} q_{ix}k_{0x}r_{i0}, \sum_{x} q_{ix}k_{1x}r_{i1}, ......, \sum_{x} q_{ix}k_{n+m,x}r_{i,n+m}) = max(M_{i,(0,n-1)}, M_{i, (n, n+m)})$

<p>
$L_{i,(0,n+m)} = \sum _{j=0}^{n+m}e^{S_{ij}-M_{i,(0,n+m)}}$
</p>

<p>
$= \sum _{j=0}^{n+m} e^{\sum _{x}q_{ix}k_{jx}r_{ij}-M_{i,(0,n+m)}}$
</p>

<p>
$= \sum _{j=0}^{n-1} e^{\sum _{x}q_{ix}k_{jx}r_{ij}-M_{i,(0,n+m)}} + \sum _{j=n}^{n+m}e^{\sum _{x}q_{ix}k_{j,x}r_{ij}-M_{i,(0,n+m)}} $
</p>

<p>
$= e^{M_{i, (0,n-1)} - M_{i, (0,n+m)}}\sum _{j=0}^{n-1}e^{\sum _{x}q_{ix}k_{jx}r_{ij}-M_{i,(0,n-1)}} + \sum _{j=n}^{n+m}e^{\sum _{x}q_{ix}k_{j,x}r_{ij}-M_{i,(0,n+m)}}$
</p>

<p>
$= e^{M_{i, (0,n-1)} - M_{i, (0,n+m)}}\sum _{j=0}^{n-1}e^{\sum _{x}q_{ix}k_{jx}r_{ij}-M_{i,(0,n)}} + e^{M_{i, (n,n+m)} - M_{i, (0,n+m)}}\sum _{j=n}^{n+m}e^{\sum _{x}q_{ix}k_{j,x}r_{ij}-M_{i,(n,n+m)}}$
</p>

<p>
$= e^{M_{i, (0,n-1)} - M_{i, (0,n+m)}}L_{i,(0,n-1)} + e^{M_{i, (n,n+m)} - M_{i, (0,n+m)}}L_{i,(n,n+m)}$
</p>

<p>
$O_{i,j,n+m}= \sum _{x=0}^{n+m} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj}$
</p>

<p>
$= \sum _{x=0}^{n-1} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj} + \sum _{x=n}^{n+m}\frac{e^{\sum_{y} q_{iy}k_{x,y}r_{ix}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{x,j}$
</p>

<p>
$= \sum _{x=0}^{n-1} \frac{L_{i,(0,n-1)}}{L_{i,(0,n+m)}} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}}v_{xj} + \sum _{x=n}^{n+m} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj}$
</p>

<p>
$= \sum _{x=0}^{n-1} \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} \frac{e^{\sum_{y} q_{iy}k_{xy}-M_{i,(0,n-1)}}}{L_{i,(0,n-1)}}v_{xj} + \sum _{x=n}^{n+m} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}v_{xj}$
</p>

<p>
$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \sum _{x=n}^{n+m} \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(n,n+m)}}}{L_{i,(n,n+m)}}v_{xj}$
</p>

<p>
$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \sum _{x=n}^{n+m} \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}P_{i,x,(n,n+m)}v_{xj}$
</p>

<p>
$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}\sum _{x=n}^{n+m} P_{i,x,(n,n+m)}v_{xj}$
</p>

$= \frac{L_{i,(0,n-1)}e^{M_{i,(0,n-1)}-M_{i,(0,n+m)}}}{L_{i,(0,n-1)}} O_{i,j,n-1} + \frac{L_{i,(n,n+m)}e^{M_{i,(n,n+m)}-M_{i,(0,n+m)}}}{L_{i,(0,n+m)}}P_{i,(n,n+m)}v_{j}^{T}$

$n+m=N-1$时，

$M_{i,(0, N-1)} = M_{i}$

$L_{i,(0, N-1)} = L_{i}$

$O_{i,j,N-1} = O_{i,j}$

# backward
## 链式法则
### 矩阵形式
$P=softmax(QK^{T}*R)$

$O=softmax(QK^{T}*R)V$

$dP=dOV^{T}$

<p>
$\frac {\partial f(o(q))}{\partial q} =(softmax(QK^{T}*R)* (dOV^{T} - sum(softmax(QK^{T}*R)V * dO, dim=-1, keepdim=True))*R)K$
</p>

<p>
$=(softmax(QK^{T}*R)* (dOV^{T} - sum(softmax(QK^{T}*R)V * dO))*R)K$
</p>

$= (P*(dP - sum(O*dO))*R)K$


$\frac {\partial f(o(k))}{\partial k}$

<p>
$= (softmax(QK^{T}*R)*(dOV^{T} - sum(softmax(QK^{T}*R)V * dO, dim=-1, keepdim=True))*R)^{T}Q$
</p>

<p>
$= (softmax(QK^{T}*R)*(dOV^{T} - sum(softmax(QK^{T}*R)V * dO))*R)^{T}Q$
</p>

$= (P*(dP - sum(O * dO))*R)^{T}Q$

$\frac {\partial f(o(v))}{\partial v}$

$= softmax(QK^T*R)^{T}dO$

$=P^TdO$

### 元素形式
$\frac {\partial f(o(q))}{\partial q_{ij}}=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i}}}}{L_{i}}(\sum_{b}do_{ib}v_{wb} - \sum_{b}(\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i}}}{L_{i}}v_{xb})do_{ib})r_{iw}k_{wj}$

$\frac {\partial f(o(k))}{\partial k_{ij}}=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a}}}}{L_{a}} (\sum_{b} do_{ab}v_{xb} - \sum_{b}(\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}-M_{a}}}{L_{a}}}v_{xb})do_{ab})r_{ai}q_{aj}$

$\frac {\partial f(o(v))}{\partial v_{ij}}$

$=\sum_{a} do_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}{L_{a}}$

### 实现形式

$P_{ij} = \frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i}}}}{L_{i}}$

$dP_{ij}= \sum_{b} do_{ib}v_{jb}$

$O_{ij} = \sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i}}}{L_{i}}v_{xj}$

$D_{i}=\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i}}}{L_{i}}\sum_{b}do_{ib}v_{xb}$

$=\sum_{b}(\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i}}}{L_{i}}v_{xb})do_{ib}$

$=\sum_{b}O_{ib}do_{ib}$

$dQ_{ij} = \frac {\partial f(o(q))}{\partial q_{ij}}$

$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i}}}}{L_{i}}(\sum_{b}do_{ib}v_{wb} - \sum_{b}(\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i}}}{L_{i}}v_{xb})do_{ib})r_{iw}k_{wj}$

$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}-M_{i}}}}{L_{i}}(dP_{iw} - D_{i})r_{iw}k_{wj}$

$dK_{ij}=\frac {\partial f(o(k))}{\partial k_{ij}}$

$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a}}}}{L_{a}} (\sum_{b} do_{ab}v_{xb} - \sum_{b}(\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}-M_{a}}}{L_{a}}}v_{xb})do_{ab})r_{ai}q_{aj}$

$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a}}}}{L_{a}} (dP_{ax} - D_{i})r_{ai}q_{aj}$

$dV_{ij} = \frac {\partial f(o(v))}{\partial v_{ij}}$

$=\sum_{a} do_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}{L_{a}}$

### flash形式
#### $dq$ flash形式
$dQ_{i,j,n-1}=\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i,(0,N-1)}}}}{L_{i,(0, N-1)}}(\sum_{b}do_{ib}v_{wb} - \sum_{b}(\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,N-1)}}}{L_{i,(0,N-1)}}v_{xb})do_{ib})r_{iw}k_{wj}$

$=\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}(dP_{iw} - D_{i})r_{iw}k_{wj}$

$dQ_{i,j,n+m}=\sum_{w=0}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}(\sum_{b}do_{ib}v_{wb} - \sum_{b}(\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}-M_{i,(0,N-1)}}}{L_{i,(0,N-1)}}v_{xb})do_{ib})r_{iw}k_{wj}$

$=\sum_{w=0}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}(dP_{iw} - D_{i})r_{iw}k_{wj}$

$=\sum_{w=0}^{n-1}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}(dP_{iw} - D_{i})r_{iw}k_{wj} + \sum_{w=n}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}(dP_{iw} - D_{i})r_{iw}k_{wj}$

$=dQ_{i,j,n-1} + \sum_{w=n}^{n+m}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}-M_{i,(0,N-1)}}}}{L_{i,(0,N-1)}}(dP_{iw} - D_{i})r_{iw}k_{wj}$

#### $dk$ flash形式

$dK_{i,j,n-1}=\sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}}{L_{a,(0,N-1)}} (\sum_{b} do_{ab}v_{xb} - \sum_{b}(\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}-M_{a,(0,N-1)}}}{L_{a,(0,N-1)}}}v_{xb})do_{ab})r_{ai}q_{aj}$

$=\sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}}{L_{a,(0,N-1)}} (dP_{ax} - \sum_{b}o_{ab}do_{ab})r_{ai}q_{aj}$

$=\sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}}{L_{a,(0,N-1)}} (dP_{ax} - D_{a})r_{ai}q_{aj}$

$dK_{i,j,n+m}=\sum_{a=0}^{n+m} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}}{L_{a,(0,N-1)}} (dP_{ax} - D_{a})r_{ai}q_{aj}$

$=\sum_{a=0}^{n-1} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}}{L_{a,(0,N-1)}} (dP_{ax} - D_{a})r_{ai}q_{aj} + \sum_{a=n}^{n+m} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}}{L_{a,(0,N-1)}} (dP_{ax} - D_{a})r_{ai}q_{aj} $

$=dK_{i,j,n-1} + \sum_{a=n}^{n+m} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}}{L_{a,(0,N-1)}} (dP_{ax} - D_{a})r_{ai}q_{aj} $

#### $dv$ flash形式
$dV_{i,j}=\frac {\partial f(o(v))}{\partial v_{ij}}=\sum_{a} do_{aj} . \frac{exp(\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a})}{L_{a}}$

$dV_{i,j,n-1} = \sum_{a=0}^{n-1} do_{aj}\frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}{L_{a,(0,N-1)}}$

$dV_{i,j,n+m}= \sum_{a=0}^{n+m} do_{aj}\frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}{L_{a,(0,N-1)}}$

$= \sum_{a=0}^{n-1} do_{aj}\frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}{L_{a,(0,N-1)}} + \sum_{a=n}^{n+m} do_{aj}\frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}{L_{a,(0,N-1)}}$

$= dV_{i,j,n-1} + \sum_{a=n}^{n+m} do_{aj}\frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}-M_{a,(0,N-1)}}}{L_{a,(0,N-1)}}$
