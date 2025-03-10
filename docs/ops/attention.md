
# forward
flashattention中，好像是使用列表示进行推到，与torch表示不符，令人困惑，本文使用行表示推导

## 矩阵表示
$Q, K, V \in R^{N \times d}$

$O \in R^{N \times d}$

$S = QK^{T} \in R^{N \times N}$

$P = softmax(S) \in R^{N \times N}$

$O = PV  \in R^{N \times d}$

## 元素表示
$S_{ij} = \sum q_{ix}k_{jx}$

<p>
$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx})$
</p>

<p>
$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} softmax(S)_{ix}v_{xj} = \sum _{x} \frac{exp(S_{ix})}{L_{i}}v_{xj} = \sum _{x} \frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}}v_{xj}$
</p>

## 行表示
<p>
$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx}) = exp(\sum _{j}q_{i}k_{j}^{T}) $
</p>

<p>
$o_{i} = \sum _{x} \frac{exp(q_{i}k_{x}^{T})}{L_{i}}v_{x}$
</p>

# backward
## 求导

### 通用求导

$\frac {\partial \frac{a(x)}{b(x)}}{\partial x} = \frac{\frac{\partial a(x)}{\partial x}.b(x) - a(x).\frac{\partial b(x)}{\partial x}}{b(x)^{2}}$

### $q$求导

$\frac {\partial o_{ab}}{\partial q_{ij}}$

<p>
$=\frac {\sum _{x} \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}v_{xb}}{\partial q_{ij}}$
</p>

#### $a \neq i$

$\frac {\partial o_{ab}}{\partial q_{ij}} = 0$

### $ a = i$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial q_{ij}} . v_{xb} + \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}} . \frac{\partial v_{xb}}{\partial q_{ij}}$

$由于v_{xb} 与 q_{ij}无关$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial q_{ij}} . v_{xb}$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}}}{\partial q_{ij}} . v_{xb}$

$=\sum_{x} {\frac{\frac {exp(\sum_{y} q_{iy}k_{xy})}{q_{ij}}.L_{a} - exp(\sum_{y} q_{iy}k_{xy}).\frac{L_i}{\partial q_{ij}}}{L_{i}^{2}}} . v_{xb}$

$=\sum_{x} {\frac{\frac {exp(\sum_{y} q_{iy}k_{xy})}{q_{ij}}.L_{i} - exp(\sum_{y} q_{iy}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{iz}k_{wz})}{\partial q_{ij}}}{L_{i}^{2}}} . v_{xb}$

$={\frac{\sum_{x} \frac {exp(\sum_{y} q_{iy}k_{xy})}{q_{ij}}.L_{i}.v_{xb} - \sum_{x}exp(\sum_{y} q_{iy}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{iz}k_{wz})}{\partial q_{ij}} . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}.exp(\sum_{y} q_{iy}k_{xy})}.L_{i}.v_{xb} - \sum_{x}exp(\sum_{y} q_{iy}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{iz}k_{wz})}{\partial q_{ij}} . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}.exp(\sum_{y} q_{iy}k_{xy})}.L_{i}.v_{xb} - \sum_{x}exp(\sum_{y} q_{iy}k_{xy}).\sum_{w}k_{wj}exp(\sum_{z}q_{iz}k_{wz}) . v_{xb}}{L_{i}^{2}}}$

### $k$求导

$\frac {\partial o_{ab}}{\partial k_{ij}}$

<p>
$=\frac {\sum _{x} \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}v_{xb}}{\partial k_{ij}}$
</p>

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial k_{ij}} . v_{xb} + \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}} . \frac{\partial v_{xb}}{\partial k_{ij}}$

$由于v_{xb} 与 k_{ij}无关$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial k_{ij}} . v_{xb}$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial k_{ij}} . v_{xb}$

$=\sum_{x} {\frac{\frac {exp(\sum_{y} q_{ay}k_{xy})}{k_{ij}}.L_{a} - exp(\sum_{y} q_{ay}k_{xy}).\frac{L_a}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$=\sum_{x} {\frac{\frac {exp(\sum_{y} q_{ay}k_{xy})}{k_{ij}}.L_{a} - exp(\sum_{y} q_{ay}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{az}k_{wz})}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$={\frac{\sum_{x} \frac {exp(\sum_{y} q_{ay}k_{iy})}{k_{ij}}.L_{a}.v_{xb} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{az}k_{wz})}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{xb} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{az}k_{wz})}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{xb} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).\frac{exp(\sum_{z}q_{az}k_{iz})}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{xb} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{xb} - q_{aj}.{exp(\sum_{z}q_{az}k_{iz})}.(\sum_{x}exp(\sum_{y} q_{ay}k_{xy}).v_{xb})}{L_{a}^{2}}}$

## $v$求导

$\frac {\partial o_{ab}}{\partial v_{ij}}$

<p>
$=\frac {\sum _{x} \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}v_{xb}}{\partial v_{ij}}$
</p>

# $b \neq j$

$\frac {\partial o_{ab}}{\partial v_{ij}} = 0$

# $b = j$

$\frac {\partial o_{ab}}{\partial v_{ij}}$

<p>
$=\frac {\sum _{x} \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}v_{xj}}{\partial v_{ij}}$
</p>

$= \frac{exp(\sum_{y} q_{ay}k_{iy})}{L_{a}}$

# 链式法则

## $q$链式法则
$\frac {\partial f(attention(q))}{\partial q_{ij}}$

<p>
$=\sum_{a}\sum_{b} \frac {\partial f(attention(q))}{\partial attention(q)_{ab}} . \frac {\partial attention(q)_{ab}}{\partial q_{ij}}$
</p>

<p>
$=\sum_{a}\sum_{b} df_{ab} . \frac {\partial attention(q)_{ab}}{\partial q_{ij}}$
</p>

<p>
$=\sum_{b} df_{ib} . \frac {\partial attention(q)_{ib}}{\partial q_{ij}}$
</p>

$=\sum_{b} df_{ib} . {\frac{{\sum_{x}k_{xj}.exp(\sum_{y} q_{iy}k_{xy})}.L_{i}.v_{xb} - \sum_{x}exp(\sum_{y} q_{iy}k_{xy}).\sum_{w}k_{wj}exp(\sum_{z}q_{iz}k_{wz}) . v_{xb}}{L_{i}^{2}}}$

$={\frac{{\sum_{x}k_{xj}.exp(\sum_{y} q_{iy}k_{xy})}.L_{i}.\sum_{b} (df_{ib}v_{xb}) - \sum_{x}exp(\sum_{y} q_{iy}k_{xy}).\sum_{w}k_{wj}exp(\sum_{z}q_{iz}k_{wz}) . \sum_{b} (df_{ib}v_{xb})}{L_{i}^{2}}}$

$={\sum_{x}\frac{{exp(\sum_{y} q_{iy}k_{xy})}.\sum_{b} (df_{ib}v_{xb}).k_{xj}}{L_{i}}} - \sum_{x}\sum_{w}{\frac{exp(\sum_{y} q_{iy}k_{xy}).exp(\sum_{z}q_{iz}k_{wz}) . \sum_{b} (df_{ib}v_{xb}).k_{wj}}{L_{i}^{2}}}$

$={\sum_{w}\frac{{exp(\sum_{y} q_{iy}k_{wy})}.\sum_{b} (df_{ib}v_{wb}).k_{wj}}{L_{i}}} - \sum_{x}\sum_{w}{\frac{exp(\sum_{y} q_{iy}k_{xy}).exp(\sum_{z}q_{iz}k_{wz}) . \sum_{b} (df_{ib}v_{xb}).k_{wj}}{L_{i}^{2}}}$

$={\sum_{w}\frac{{exp(\sum_{y} q_{iy}k_{wy})}.\sum_{b} (df_{ib}v_{wb}).k_{wj}}{L_{i}}} - \sum_{w}\sum_{x}{\frac{exp(\sum_{y} q_{iy}k_{xy}).exp(\sum_{z}q_{iz}k_{wz}) . \sum_{b} (df_{ib}v_{xb}).k_{wj}}{L_{i}^{2}}}$

$={\sum_{w}\frac{{exp(\sum_{y} q_{iy}k_{wy})}}{L_{i}}.\sum_{b} (df_{ib}v_{wb}).k_{wj}} - \sum_{w}\sum_{x}{\frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}} . \sum_{b} (df_{ib}v_{xb}). \frac{exp(\sum_{z}q_{iz}k_{wz})}{L_{i}} .k_{wj}}$

$={\sum_{w}\frac{{exp(\sum_{y} q_{iy}k_{wy})}}{L_{i}}.\sum_{b} (df_{ib}v_{wb}).k_{wj}} - \sum_{w}\frac{exp(\sum_{z}q_{iz}k_{wz})}{L_{i}}\sum_{x}{\frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}} . \sum_{b} (df_{ib}v_{xb}) .k_{wj}}$

$=\sum_{w}\frac{{exp(\sum_{y} q_{iy}k_{wy})}}{L_{i}}.(\sum_{b} (df_{ib}v_{wb}) - \sum_{x}\frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}} . \sum_{b} (df_{ib}v_{xb})) .k_{wj}$

### 行形式
$\frac {\partial f(attention(q))}{\partial q_{ij}} =\sum_{w}\frac{{exp(q_{i}k_{w}^T)}}{L_{i}}.(df_{i}v_{w}^{T} - \sum_{x}\frac{exp(q_{i}k_{x}^T)}{L_{i}} .df_{i}v_{x}^T) .k_{wj}$

$\frac {\partial f(attention(q))}{\partial q_{i}} =\sum_{w}\frac{{exp(q_{i}k_{w}^T)}}{L_{i}}.(df_{i}v_{w}^{T} - \sum_{x}\frac{exp(q_{i}k_{x}^T)}{L_{i}} .df_{i}v_{x}^T) .k_{w}$

### 矩阵形式
$\frac {\partial f(attention(q))}{\partial q} =(softmax(Q@K^{T})* (d_{f}@V^{T} - sum(softmax(Q@K_{T}) * (d_{f}@V^{T}), dim=-1, keepdim=True)))@K$

## $k$链式法则
### 元素形式
$\frac {\partial f(attention(k))}{\partial k_{ij}}$

<p>
$=\sum_{a}\sum_{b} \frac {\partial f(attention(q))}{\partial attention(q)_{ab}} . \frac {\partial attention(q)_{ab}}{\partial k_{ij}}$
</p>

<p>
$=\sum_{a}\sum_{b} df_{ab} . \frac {\partial attention(q)_{ab}}{\partial k_{ij}}$
</p>

$=\sum_{a}\sum_{b} df_{ab} . {\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{xb} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . v_{xb}}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.\sum_{b} (df_{ab} .v_{xb}) - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . \sum_{b} (df_{ab} . v_{xb})}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})} .\sum_{b} (df_{ab} .v_{xb})}{L_{a}}} - \sum_{a} {\frac{\sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . \sum_{b} (df_{ab} . v_{xb})}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{exp(\sum_{y} q_{ay}k_{iy})}}{L_{a}} .\sum_{b} (df_{ab} .v_{xb}).q_{aj}} - \sum_{a} \frac{exp(\sum_{z}q_{az}k_{iz})}{L_{a}} . \sum_{x}{\frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}.\sum_{b} (df_{ab} . v_{xb}).q_{aj}$

$=\sum_{a} \frac{{exp(\sum_{y} q_{ay}k_{iy})}}{L_{a}} . (\sum_{b} (df_{ab} .v_{xb}) - \sum_{x}{\frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}.\sum_{b} (df_{ab} . v_{xb})) .q_{aj}$

### 行形式

$\frac {\partial f(attention(k))}{\partial k_{ij}} = \sum_{a} \frac{{exp(q_{a}k_{i}^{T})}}{L_{a}} . (df_{a}v_{i}^{T} - {\frac{\sum_{x}exp(q_{a}k_{x}^{T})}{L_{a}}}.df_{a}v_{x}^{T}) .q_{aj}$

$\frac {\partial f(attention(k))}{\partial k_{i}} = \sum_{a} \frac{{exp(q_{a}k_{i}^{T})}}{L_{a}} . (df_{a}v_{i}^{T} - {\frac{\sum_{x}exp(q_{a}k_{x}^{T})}{L_{a}}}.df_{a}v_{x}^{T}) .q_{a}$

### 矩阵形式
$\frac {\partial f(attention(k))}{\partial k} = (softmax(Q@K^{T})*(df@V^{T} - sum(softmax(Q@K^{T}) * df@V^{T}, dim=-1, keepdim=True)))^{T}@Q$

## $v$链式法则
### 元素形式
$\frac {\partial f(attention(q))}{\partial v_{ij}}$

<p>
$=\sum_{a}\sum_{b} \frac {\partial f(attention(q))}{\partial attention(q)_{ab}} . \frac {\partial attention(q)_{ab}}{\partial v_{ij}}$
</p>

<p>
$=\sum_{a}\sum_{b} df_{ab} . \frac {\partial attention(q)_{ab}}{\partial v_{ij}}$
</p>

$=\sum_{a} df_{aj} . \frac{exp(\sum_{y} q_{ay}k_{iy})}{L_{a}}$

### 行形式

$\frac {\partial f(attention(q))}{\partial v_{ij}} = \sum_{a} df_{aj} . \frac{exp(\sum_{y} q_{a}k_{i}^T)}{L_{a}} = \sum_{a} \frac{exp(\sum_{y} q_{a}k_{i}^T)}{L_{a}}.df_{aj}$

$\frac {\partial f(attention(q))}{\partial v_{i}} = \sum_{a} \frac{exp(\sum_{y} q_{a}k_{i}^T)}{L_{a}}.df_{a}$


