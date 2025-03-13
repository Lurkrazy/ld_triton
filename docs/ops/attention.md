
 The theory part of paper [flash attention v1](https://arxiv.org/abs/2205.14135) and [flash attention v2](https://tridao.me/publications/flash2/flash2.pdf) have some errors and skip step serious. It makes me very confuse. So, i  elements view to derive again. Includes differentiation and the chain rule of forward and backward

# forward

## Matrix Representation
$Q, K, V \in R^{N \times d}$

$O \in R^{N \times d}$

$S = QK^{T} \in R^{N \times N}$

$R = scale$

$P = softmax(S*R) \in R^{N \times N}$

$O = PV  \in R^{N \times d}$

## Element Representation
$S_{ij} = \sum_{x} q_{ix}k_{jx}r_{ij}$

<p>
$L_{i} = \sum _{j}e^{S_{ij}r_{ij}} = \sum _{j} e^{\sum _{x}q_{ix}k_{jx}r_{ij}}$
</p>

$p_{ij}=\frac{e^{\sum_{x} q_{ix}k_{jx}r_{ij}}}{L_{i}}$

<p>
$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} softmax(S*R)_{ix}v_{xj} = \sum _{x} \frac{e^{S_{ix}r_{ix}}}{L_{i}}v_{xj} = \sum _{x} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}}v_{xj}$
</p>

# backward
## Differentiation

### General Differentiation

$\frac {\partial \frac{a(x)}{b(x)}}{\partial x} = \frac{\frac{\partial a(x)}{\partial x}.b(x) - a(x).\frac{\partial b(x)}{\partial x}}{b(x)^{2}}$

### $q$ Differentiation

$\frac {\partial o_{ab}}{\partial q_{ij}}$

<p>
$=\frac {\sum _{x} \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}v_{xb}}{\partial q_{ij}}$
</p>

#### $a \neq i$

$\frac {\partial o_{ab}}{\partial q_{ij}} = 0$

### $a = i$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial q_{ij}} . v_{xb} + \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}} . \frac{\partial v_{xb}}{\partial q_{ij}}$

because $v_{xb}$ is free with $q_{ij}$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial q_{ij}} . v_{xb}$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}}}{\partial q_{ij}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{q_{ij}}.L_{i} - e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{L_i}{\partial q_{ij}}}{L_{i}^{2}}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{q_{ij}}.L_{i} - e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{\sum_{w} e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{\partial q_{ij}}}{L_{i}^{2}}} . v_{xb}$

$={\frac{\sum_{x} \frac {e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{q_{ij}}.L_{i}.v_{xb} - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{\sum_{w} e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{\partial q_{ij}} . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}r_{ix}.e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}.L_{i}.v_{xb} - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{\sum_{w} e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{\partial q_{ij}} . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}r_{ix}.e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}.L_{i}.v_{xb} - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\sum_{w}k_{wj}r_{iw}e^{\sum_{z}q_{iz}k_{wz}r_{iw}} . v_{xb}}{L_{i}^{2}}}$

### $k$ Differentiation

$\frac {\partial o_{ab}}{\partial k_{ij}}$

<p>
$=\frac {\sum _{x} \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}v_{xb}}{\partial k_{ij}}$
</p>

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial k_{ij}} . v_{xb} + \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}} . \frac{\partial v_{xb}}{\partial k_{ij}}$

because $v_{xb}$ is free with $k_{ij}$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial k_{ij}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{k_{ij}}.L_{a} - e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{L_a}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{k_{ij}}.L_{a} - e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{\sum_{w} e^{\sum_{z}q_{az}k_{wz}r_{aw}}}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$={\frac{\sum_{x} \frac {e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}{k_{ij}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{\sum_{w} e^{\sum_{z}q_{az}k_{wz}r_{aw}}}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{\sum_{w} e^{\sum_{z}q_{az}k_{wz}r_{aw}}}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{e^{\sum_{z}q_{az}k_{iz}r_{ai}}}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.q_{aj}r_{ai}.{e^{\sum_{z}q_{az}k_{iz}r_{ai}}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - q_{aj}.{e^{\sum_{z}q_{az}k_{iz}r_{ai}}}.\sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.v_{xb}}{L_{a}^{2}}}$

## $v$ Differentiation

$\frac {\partial o_{ab}}{\partial v_{ij}}$

<p>
$=\frac {\sum _{x} \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}v_{xb}}{\partial v_{ij}}$
</p>

# $b \neq j$

$\frac {\partial o_{ab}}{\partial v_{ij}} = 0$

# $b = j$

$\frac {\partial o_{ab}}{\partial v_{ij}}$

<p>
$=\frac {\sum _{x} \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}v_{xj}}{\partial v_{ij}}$
</p>

$= \frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}{L_{a}}$

# Chain Rule

## $q$ Chain Rule
$\frac {\partial f(o(q))}{\partial q_{ij}}$

<p>
$=\sum_{a}\sum_{b} \frac {\partial f(o(q))}{\partial o(q)_{ab}} . \frac {\partial o(q)_{ab}}{\partial q_{ij}}$
</p>

<p>
$=\sum_{a}\sum_{b} do_{ab} . \frac {\partial o(q)_{ab}}{\partial q_{ij}}$
</p>

<p>
$=\sum_{b} do_{ib} . \frac {\partial o(q)_{ib}}{\partial q_{ij}}$
</p>

$=\sum_{b} do_{ib} . {\frac{{\sum_{x}k_{xj}r_{ix}.e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}.L_{i}.v_{xb} - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\sum_{w}k_{wj}r_{iw}e^{\sum_{z}q_{iz}k_{wz}r_{iw}} . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}r_{ix}.e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}.L_{i}.\sum_{b} (do_{ib}v_{xb}) - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\sum_{w}k_{wj}r_{iw}e^{\sum_{z}q_{iz}k_{wz}r_{iw}} . \sum_{b} (do_{ib}v_{xb})}{L_{i}^{2}}}$

$={\sum_{x}\frac{{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}.\sum_{b} (do_{ib}v_{xb}).k_{xj}r_{ix}}{L_{i}}} - \sum_{x}\sum_{w}{\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.e^{\sum_{z}q_{iz}k_{wz}r_{iw}} . \sum_{b} (do_{ib}v_{xb}).k_{wj}r_{iw}}{L_{i}^{2}}}$

$={\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}.\sum_{b} (do_{ib}v_{wb}).k_{wj}r_{iw}}{L_{i}}} - \sum_{x}\sum_{w}{\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.e^{\sum_{z}q_{iz}k_{wz}r_{iw}} . \sum_{b} (do_{ib}v_{xb}).k_{wj}r_{iw}}{L_{i}^{2}}}$

$={\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}.\sum_{b} (do_{ib}v_{wb}).k_{wj}r_{iw}}{L_{i}}} - \sum_{w}\sum_{x}{\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.e^{\sum_{z}q_{iz}k_{wz}r_{iw}} . \sum_{b} (do_{ib}v_{xb}).k_{wj}r_{iw}}{L_{i}^{2}}}$

$={\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}}{L_{i}}.\sum_{b} (do_{ib}v_{wb}).k_{wj}r_{iw}} - \sum_{w}\sum_{x}{\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}} . \sum_{b} (do_{ib}v_{xb}). \frac{e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{L_{i}} .k_{wj}}r_{iw}$

$={\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}}{L_{i}}.\sum_{b} (do_{ib}v_{wb}).k_{wj}r_{iw}} - \sum_{w}\frac{e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{L_{i}} \sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}} . \sum_{b} (do_{ib}v_{xb}) .k_{wj}r_{iw}$

$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}}{L_{i}}.(\sum_{b} (do_{ib}v_{wb}) - \sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}} . \sum_{b} (do_{ib}v_{xb})) .k_{wj}r_{iw}$

#### Chain Rule One
$=\sum_{w} p_{iw}.(\sum_{b} (do_{ib}v_{wb}) - \sum_{x}p_{ix} . \sum_{b} (do_{ib}v_{xb})) .k_{wj}r_{iw}$

#### Chain Rule Two
$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}}{L_{i}}.(\sum_{b} (do_{ib}v_{wb}) - \sum_{x}\frac{e_{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}} . \sum_{b} v_{xb} do_{ib}) .k_{wj}r_{iw}$

$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}}{L_{i}}.(\sum_{b} (do_{ib}v_{wb}) - \sum_{b} (\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}}v_{xb}) do_{ib}) .k_{wj}r_{iw}$

$=\sum_{w}p_{iw}.(\sum_{b}do_{ib}v_{wb} - \sum_{b}o_{ib} do_{ib}) .k_{wj}r_{iw}$

### Row Representation
#### Representation One
$dp_{ij} = \sum_{x}do_{ix}v_{jx} = do_{i}v_{j}^T$

$\frac {\partial f(o(q))}{\partial q_{ij}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^Tr_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{x}\frac{e^{q_{i}k_{x}^T r_{ix}}}{L_{i}} .do_{i}v_{x}^T) .k_{wj}r_{iw}$

$=\sum_{w}p_{iw}.(do_{i}v_{w}^{T} - \sum_{x}p_{ix} .do_{i}v_{x}^T) .k_{wj}r_{iw}$

$= \sum_{w}p_{iw}.(dp_{iw} - \sum_{x}p_{ix} .dp_{ix}) .k_{wj}r_{iw}$

$\frac {\partial f(o(q))}{\partial q_{i}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^T r_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{x}\frac{e^{q_{i}k_{x}^T r_{ix}}}{L_{i}} .do_{i}v_{x}^T)r_{iw}.k_{w}$

$=\sum_{w} p_{iw}.(do_{i}v_{w}^{T} - \sum_{x}p_{iw} .do_{i}v_{x}^T)r_{iw}.k_{w}$

$=\sum_{w} p_{iw}.(dp_{iw} - \sum_{x}p_{iw} .dp_{ix})r_{iw}.k_{w}$

### Representation Two
$\frac {\partial f(o(q))}{\partial q_{ij}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^Tr_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{b}o_{ib}do_{ib})r_{iw}.k_{wj}$

$=\sum_{w}p_{iw}.(do_{i}v_{w}^{T} - o_ido_{i}^T)r_{iw} .k_{wj}$

$=\sum_{w}p_{iw}.(dp_{iw} - o_ido_{i}^T)r_{iw} .k_{wj}$

$\frac {\partial f(o(q))}{\partial q_{i}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^Tr_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{b}o_{ib}do_{ib})r_{iw}.k_{w}$

$=\sum_{w} p_{iw}.(do_{i}v_{w}^{T} - o_{i}do_{i}^T)r_{iw}.k_{w}$

### Matrix Representation

$P=softmax(QK^{T}*R)$

$dP=dOV^{T}$

#### Representation One
$\frac {\partial f(o(q))}{\partial q}$


$=((softmax(QK^{T}*R)* (dOV^{T} - sum(softmax(QK^{T}*R) * (dOV^{T}), dim=-1, keepdim=True)))*R)K$


$=(P * (dOV^{T} - sum(P * dP, dim=-1, keepdim=True))*R)K$

#### Representation Two
$\frac {\partial f(o(q))}{\partial q}$


$= (softmax(QK^{T}*R)* (dOV^{T} - sum(softmax(QK^{T}*R)V*dO, dim=-1, keepdim=True))*R)K$


$= (P* (dP - sum(O.dO, dim=-1, keepdim=True))*R)K$

## $k$ Chain Rule
### Element Representation
$\frac {\partial f(o(k))}{\partial k_{ij}}$

<p>
$=\sum_{a}\sum_{b} \frac {\partial f(o(q))}{\partial o(q)_{ab}} . \frac {\partial o(q)_{ab}}{\partial k_{ij}}$
</p>

<p>
$=\sum_{a}\sum_{b} do_{ab}  \frac {\partial o(q)_{ab}}{\partial k_{ij}}$
</p>

$=\sum_{a}\sum_{b} do_{ab} . {\frac{{q_{aj}r_{ai}.e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.q_{aj}r_{ai}.{e^{\sum_{z}q_{az}k_{iz}r_{ai}}} . v_{xb}}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{q_{aj}r_{ai}.e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}.L_{a}.\sum_{b} (do_{ab} .v_{xb}) - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.q_{aj}.{e^{\sum_{z}q_{az}k_{iz}r_{ai}}} . \sum_{b} (do_{ab} . v_{xb})}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{q_{aj}r_{ai}.e^{\sum_{y} q_{ay}k_{iy}r_{ai}}} .\sum_{b} (do_{ab} .v_{xb})}{L_{a}}} - \sum_{a} {\frac{\sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.q_{aj}r_{ai}.{e^{\sum_{z}q_{az}k_{iz}r_{ai}}} . \sum_{b} (do_{ab} . v_{xb})}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} .\sum_{b} (do_{ab} .v_{xb}).q_{aj}r_{ai}} - \sum_{a} \frac{e^{\sum_{z}q_{az}k_{iz}r_{ai}}}{L_{a}} . \sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}.\sum_{b} (do_{ab} . v_{xb}).q_{aj}r_{ai}$

#### Representation One
$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b} (do_{ab} .v_{xb}) - \sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}.\sum_{b} (do_{ab} . v_{xb})) .q_{aj}r_{ai}$

#### Representation Two
$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b} do_{ab} .v_{xb} - \sum_{b}\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}v_{xb}do_{ab}) .q_{aj}r_{ai}$

$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b}do_{ab} .v_{xb} - \sum_{b}(\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}v_{xb})do_{ab}) .q_{aj}r_{ai}$

### Row Representation
#### Representation One
$\frac {\partial f(o(k))}{\partial k_{ij}}$

$= \sum_{a} \frac{{e^{q_{a}k_{i}^{T}r_{ai}}}}{L_{a}} . (do_{a}v_{i}^{T} - \sum_{x}{\frac{e^{q_{a}k_{x}^{T}r_{ax}}}{L_{a}}}.do_{a}v_{x}^{T})r_{ai} .q_{aj}$

$= \sum_{a} p_{ai} . (do_{a}v_{i}^{T} - \sum_{x} p_{ax}.do_{a}v_{x}^{T})r_{ai} .q_{aj}$

$= \sum_{a} p_{ai} . (dp_{ai} - \sum_{x} p_{ax}.dp_{ax})r_{ai} .q_{aj}$

$\frac {\partial f(o(k))}{\partial k_{i}}$

$= \sum_{a} p_{ai} . (dp_{ai} - \sum_{x} p_{ax}.dp_{ax})r_{ai} .q_{a}$

#### Representation Two
$\frac {\partial f(o(k))}{\partial k_{ij}}$

$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b}do_{ab} .v_{xb} - \sum_{b}(\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}v_{xb})do_{ab})r_{ai} .q_{aj}$

$=\sum_{a} p_{ai}. (dp_{ax} - \sum_{b}o_{ab}do_{ab})r_{ai} .q_{aj}$

### Matrix Representation
#### Representation One
$\frac {\partial f(o(k))}{\partial k}$

$= (P*(dP - sum(P * dP, dim=-1, keepdim=True))*R)^{T}Q$

#### Representation Two

$\frac {\partial f(o(k))}{\partial k}$

$= (P*(dP - sum(O * dO, dim=-1, keepdim=True))*R)^{T}Q$

## $v$ Chain Rule
### Element Representation
$\frac {\partial f(o(v))}{\partial v_{ij}}$

<p>
$=\sum_{a}\sum_{b} \frac {\partial f(o(v))}{\partial o(v)_{ab}} . \frac {\partial o(v)_{ab}}{\partial v_{ij}}$
</p>

<p>
$=\sum_{a}\sum_{b} do_{ab} . \frac {\partial o(v)_{ab}}{\partial v_{ij}}$
</p>

$=\sum_{a} do_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}{L_{a}}$

$=\sum_{a} do_{aj} . p_{ai}$

### Matrix Representation
$\frac {\partial f(o(v))}{\partial v} = softmax(QK^T*R)^{T}dO=P^TdO$
