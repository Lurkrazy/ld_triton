
# forward
flasho中，好像是使用列表示进行推到，与torch表示不符，令人困惑，本文使用行表示推导

## 矩阵表示
$Q, K, V \in R^{N \times d}$

$O \in R^{N \times d}$

$S = QK^{T} \in R^{N \times N}$

$R = scale$

$P = softmax(S*R) \in R^{N \times N}$

$O = PV  \in R^{N \times d}$

## 元素表示
$S_{ij} = \sum_{x} q_{ix}k_{jx}r_{ij}$

<p>
$L_{i} = \sum _{j}e^{S_{ij}r_{ij}} = \sum _{j} e^{\sum _{x}q_{ix}k_{jx}r_{ij}}$
</p>

$p_{ij}=\frac{e^{\sum_{x} q_{ix}k_{jx}r_{ij}}}{L_{i}}$

<p>
$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} softmax(S*R)_{ix}v_{xj} = \sum _{x} \frac{e^{S_{ix}r_{ix}}}{L_{i}}v_{xj} = \sum _{x} \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}}v_{xj}$
</p>

# backward
## 求导

### 通用求导

$\frac {\partial \frac{a(x)}{b(x)}}{\partial x} = \frac{\frac{\partial a(x)}{\partial x}.b(x) - a(x).\frac{\partial b(x)}{\partial x}}{b(x)^{2}}$

### $q$求导

$\frac {\partial o_{ab}}{\partial q_{ij}}$

<p>
$=\frac {\sum _{x} \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}v_{xb}}{\partial q_{ij}}$
</p>

#### $a \neq i$

$\frac {\partial o_{ab}}{\partial q_{ij}} = 0$

### $a = i$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial q_{ij}} . v_{xb} + \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}} . \frac{\partial v_{xb}}{\partial q_{ij}}$

$由于v_{xb} 与 q_{ij}无关$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial q_{ij}} . v_{xb}$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}}}{\partial q_{ij}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{q_{ij}}.L_{i} - e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{L_i}{\partial q_{ij}}}{L_{i}^{2}}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{q_{ij}}.L_{i} - e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{\sum_{w} e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{\partial q_{ij}}}{L_{i}^{2}}} . v_{xb}$

$={\frac{\sum_{x} \frac {e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{q_{ij}}.L_{i}.v_{xb} - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{\sum_{w} e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{\partial q_{ij}} . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}r_{ix}.e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}.L_{i}.v_{xb} - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\frac{\sum_{w} e^{\sum_{z}q_{iz}k_{wz}r_{iw}}}{\partial q_{ij}} . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}r_{ix}.e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}.L_{i}.v_{xb} - \sum_{x}e^{\sum_{y} q_{iy}k_{xy}r_{ix}}.\sum_{w}k_{wj}r_{iw}e^{\sum_{z}q_{iz}k_{wz}r_{iw}} . v_{xb}}{L_{i}^{2}}}$

### $k$求导

$\frac {\partial o_{ab}}{\partial k_{ij}}$

<p>
$=\frac {\sum _{x} \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}v_{xb}}{\partial k_{ij}}$
</p>

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial k_{ij}} . v_{xb} + \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}} . \frac{\partial v_{xb}}{\partial k_{ij}}$

$由于v_{xb} 与 k_{ij}无关$

$=\sum_{x} \frac {\partial \frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}{\partial k_{ij}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{k_{ij}}.L_{a} - e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{L_a}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$=\sum_{x} {\frac{\frac {e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{k_{ij}}.L_{a} - e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{\sum_{w} e^{\sum_{z}q_{az}k_{wz}r_{aw}}}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$={\frac{\sum_{x} \frac {e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}{k_{ij}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{\sum_{w} e^{\sum_{z}q_{az}k_{wz}r_{aw}}}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{\sum_{w} e^{\sum_{z}q_{az}k_{wz}r_{aw}}}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.\frac{e^{\sum_{z}q_{az}k_{iz}r_{ai}}}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - \sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.q_{aj}r_{ai}.{e^{\sum_{z}q_{az}k_{iz}r_{ai}}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}r_{ax}.e^{\sum_{y} q_{ay}k_{iy}r_{ax}}}.L_{a}.v_{xb} - q_{aj}.{e^{\sum_{z}q_{az}k_{iz}r_{ai}}}.\sum_{x}e^{\sum_{y} q_{ay}k_{xy}r_{ax}}.v_{xb}}{L_{a}^{2}}}$

## $v$求导

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

# 链式法则

## $q$链式法则
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

#### 变形一
$=\sum_{w} p_{iw}.(\sum_{b} (do_{ib}v_{wb}) - \sum_{x}p_{ix} . \sum_{b} (do_{ib}v_{xb})) .k_{wj}r_{iw}$

#### 变形二
$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}}{L_{i}}.(\sum_{b} (do_{ib}v_{wb}) - \sum_{x}\frac{e_{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}} . \sum_{b} v_{xb} do_{ib}) .k_{wj}r_{iw}$

$=\sum_{w}\frac{{e^{\sum_{y} q_{iy}k_{wy}r_{iw}}}}{L_{i}}.(\sum_{b} (do_{ib}v_{wb}) - \sum_{b} (\sum_{x}\frac{e^{\sum_{y} q_{iy}k_{xy}r_{ix}}}{L_{i}}v_{xb}) do_{ib}) .k_{wj}r_{iw}$

$=\sum_{w}p_{iw}.(\sum_{b}do_{ib}v_{wb} - \sum_{b}o_{ib} do_{ib}) .k_{wj}r_{iw}$

### 行形式
#### 变形一
$dp_{ij} = \sum_{x}do_{ix}v_{jx} = do_{i}v_{j}^T$

$\frac {\partial f(o(q))}{\partial q_{ij}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^Tr_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{x}\frac{e^{q_{i}k_{x}^T r_{ix}}}{L_{i}} .do_{i}v_{x}^T) .k_{wj}r_{iw}$

$=\sum_{w}p_{iw}.(do_{i}v_{w}^{T} - \sum_{x}p_{ix} .do_{i}v_{x}^T) .k_{wj}r_{iw}$

$= \sum_{w}p_{iw}.(dp_{iw} - \sum_{x}p_{ix} .dp_{ix}) .k_{wj}r_{iw}$

$\frac {\partial f(o(q))}{\partial q_{i}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^T r_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{x}\frac{e^{q_{i}k_{x}^T r_{ix}}}{L_{i}} .do_{i}v_{x}^T)r_{iw}.k_{w}$

$=\sum_{w} p_{iw}.(do_{i}v_{w}^{T} - \sum_{x}p_{iw} .do_{i}v_{x}^T)r_{iw}.k_{w}$

$=\sum_{w} p_{iw}.(dp_{iw} - \sum_{x}p_{iw} .dp_{ix})r_{iw}.k_{w}$

### 变形二
$\frac {\partial f(o(q))}{\partial q_{ij}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^Tr_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{b}o_{ib}do_{ib})r_{iw}.k_{wj}$

$=\sum_{w}p_{iw}.(do_{i}v_{w}^{T} - o_ido_{i}^T)r_{iw} .k_{wj}$

$=\sum_{w}p_{iw}.(dp_{iw} - o_ido_{i}^T)r_{iw} .k_{wj}$

$\frac {\partial f(o(q))}{\partial q_{i}}$

$=\sum_{w}\frac{{e^{q_{i}k_{w}^Tr_{iw}}}}{L_{i}}.(do_{i}v_{w}^{T} - \sum_{b}o_{ib}do_{ib})r_{iw}.k_{w}$

$=\sum_{w} p_{iw}.(do_{i}v_{w}^{T} - o_{i}do_{i}^T)r_{iw}.k_{w}$

### 矩阵形式


$P=softmax(QK^{T}*R)$

$dP=dOV^{T}$

#### 变形一
$\frac {\partial f(o(q))}{\partial q}$


$=((softmax(QK^{T}*R)* (dOV^{T} - sum(softmax(QK^{T}*R) * (dOV^{T}), dim=-1, keepdim=True)))*R)K$


$=(P * (dOV^{T} - sum(P * dP, dim=-1, keepdim=True))*R)K$

#### 变形二
$\frac {\partial f(o(q))}{\partial q}$


$= (softmax(QK^{T}*R)* (dOV^{T} - sum(softmax(QK^{T}*R)V*dO, dim=-1, keepdim=True))*R)K$


$= (P* (dP - sum(O.dO, dim=-1, keepdim=True))*R)K$

## $k$链式法则
### 元素形式
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

#### 变形一
$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b} (do_{ab} .v_{xb}) - \sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}.\sum_{b} (do_{ab} . v_{xb})) .q_{aj}r_{ai}$

#### 变形二
$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b} do_{ab} .v_{xb} - \sum_{b}\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}v_{xb}do_{ab}) .q_{aj}r_{ai}$

$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b}do_{ab} .v_{xb} - \sum_{b}(\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}v_{xb})do_{ab}) .q_{aj}r_{ai}$

### 行形式
#### 变形一
$\frac {\partial f(o(k))}{\partial k_{ij}}$

$= \sum_{a} \frac{{e^{q_{a}k_{i}^{T}r_{ai}}}}{L_{a}} . (do_{a}v_{i}^{T} - \sum_{x}{\frac{e^{q_{a}k_{x}^{T}r_{ax}}}{L_{a}}}.do_{a}v_{x}^{T})r_{ai} .q_{aj}$

$= \sum_{a} p_{ai} . (do_{a}v_{i}^{T} - \sum_{x} p_{ax}.do_{a}v_{x}^{T})r_{ai} .q_{aj}$

$= \sum_{a} p_{ai} . (dp_{ai} - \sum_{x} p_{ax}.dp_{ax})r_{ai} .q_{aj}$

$\frac {\partial f(o(k))}{\partial k_{i}}$

$= \sum_{a} p_{ai} . (dp_{ai} - \sum_{x} p_{ax}.dp_{ax})r_{ai} .q_{a}$

#### 变形二
$\frac {\partial f(o(k))}{\partial k_{ij}}$

$=\sum_{a} \frac{{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}}{L_{a}} . (\sum_{b}do_{ab} .v_{xb} - \sum_{b}(\sum_{x}{\frac{e^{\sum_{y} q_{ay}k_{xy}r_{ax}}}{L_{a}}}v_{xb})do_{ab})r_{ai} .q_{aj}$

$=\sum_{a} p_{ai}. (dp_{ax} - \sum_{b}o_{ab}do_{ab})r_{ai} .q_{aj}$

### 矩阵形式
#### 变形一
$\frac {\partial f(o(k))}{\partial k}$

$= (P*(dP - sum(P * dP, dim=-1, keepdim=True))*R)^{T}Q$

#### 变形二

$\frac {\partial f(o(k))}{\partial k}$

$= (P*(dP - sum(O * dO, dim=-1, keepdim=True))*R)^{T}Q$

## $v$链式法则
### 元素形式
$\frac {\partial f(o(v))}{\partial v_{ij}}$

<p>
$=\sum_{a}\sum_{b} \frac {\partial f(o(v))}{\partial o(v)_{ab}} . \frac {\partial o(v)_{ab}}{\partial v_{ij}}$
</p>

<p>
$=\sum_{a}\sum_{b} do_{ab} . \frac {\partial o(v)_{ab}}{\partial v_{ij}}$
</p>

$=\sum_{a} do_{aj} . \frac{e^{\sum_{y} q_{ay}k_{iy}r_{ai}}}{L_{a}}$

$=\sum_{a} do_{aj} . p_{ai}$

### 矩阵形式
$\frac {\partial f(o(v))}{\partial v} = softmax(QK^T*R)^{T}dO=P^TdO$
