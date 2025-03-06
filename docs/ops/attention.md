
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

$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx})$

$o_{ij} = \sum _{x} p_{ix}v_{xj} = \sum _{x} softmax(S)_{ix}v_{xj} = \sum _{x} \frac{exp(S_{ix})}{L_{i}}v_{xj} = \sum _{x} \frac{exp(\sum_{y} q_{iy}k_{xy})}{L_{i}}v_{xj}$

## 行表示
$L_{i} = \sum _{j}exp(S_{ij}) = \sum _{j} exp(\sum _{x}q_{ix}k_{jx}) = exp(\sum _{j}q_{i}k_{j}^{T}) $

$o_{i} = \sum _{x} \frac{exp(q_{i}k_{x}^{T})}{L_{i}}v_{x}$

# backward
## 求导

### 通用求导

$\frac {\partial \frac{a(x)}{b(x)}}{\partial x} = \frac{\frac{\partial a(x)}{\partial x}.b(x) - a(x).\frac{\partial b(x)}{\partial x}}{b(x)^{2}}$

### $k$求导

$\frac {\partial o_{ab}}{\partial k_{ij}}$

$=\frac {\sum _{x} \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}v_{xb}}{\partial k_{ij}}$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial k_{ij}} . v_{xb} + \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}} . \frac{\partial v_{xb}}{\partial k_{ij}}$

$由于v_{xb} 与 k_{ij}无关$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial k_{ij}} . v_{xb}$

$=\sum_{x} \frac {\partial \frac{exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}{\partial k_{ij}} . v_{xb}$

$=\sum_{x} {\frac{\frac {exp(\sum_{y} q_{ay}k_{xy})}{k_{ij}}.L_{a} - exp(\sum_{y} q_{ay}k_{xy}).\frac{L_a}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$=\sum_{x} {\frac{\frac {exp(\sum_{y} q_{ay}k_{xy})}{k_{ij}}.L_{a} - exp(\sum_{y} q_{ay}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{az}k_{wz})}{\partial k_{ij}}}{L_{a}^{2}}} . v_{xb}$

$={\frac{\sum_{x} \frac {exp(\sum_{y} q_{ay}k_{xy})}{k_{ij}}.L_{a}.v_{xb} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{az}k_{wz})}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{ib} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).\frac{\sum_{w} exp(\sum_{z}q_{az}k_{wz})}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{ib} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).\frac{exp(\sum_{z}q_{az}k_{iz})}{\partial k_{ij}} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{ib} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . v_{xb}}{L_{a}^{2}}}$

$={\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{ib} - q_{aj}.{exp(\sum_{z}q_{az}k_{iz})}.(\sum_{x}exp(\sum_{y} q_{ay}k_{xy}).v_{xb})}{L_{a}^{2}}}$

# 链式法则

## $k$链式法则
### 元素形式
$\frac {\partial f(attention(q))}{\partial k_{ij}}$

$=\sum_{a}\sum_{b} \frac {\partial f(attention(q))}{\partial attention(q)_{ab}} . \frac {\partial attention(q)_{ab}}{\partial k_{ij}}$

$=\sum_{a}\sum_{b} df_{ab} . \frac {\partial attention(q)_{ab}}{\partial k_{ij}}$

$=\sum_{a}\sum_{b} df_{ab} . {\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.v_{ib} - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . v_{xb}}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})}.L_{a}.\sum_{b} (df_{ab} .v_{ib}) - \sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . \sum_{b} (df_{ab} . v_{xb})}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{q_{aj}.exp(\sum_{y} q_{ay}k_{iy})} .\sum_{b} (df_{ab} .v_{ib})}{L_{a}}} - \sum_{a} {\frac{\sum_{x}exp(\sum_{y} q_{ay}k_{xy}).q_{aj}.{exp(\sum_{z}q_{az}k_{iz})} . \sum_{b} (df_{ab} . v_{xb})}{L_{a}^{2}}}$

$=\sum_{a} {\frac{{exp(\sum_{y} q_{ay}k_{iy})}}{L_{a}} .\sum_{b} (df_{ab} .v_{ib}).q_{aj}} - \sum_{a} \frac{exp(\sum_{z}q_{az}k_{iz})}{L_{a}} . {\frac{\sum_{x}exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}.\sum_{b} (df_{ab} . v_{xb}).q_{aj}$

$=\sum_{a} \frac{{exp(\sum_{y} q_{ay}k_{iy})}}{L_{a}} . (\sum_{b} (df_{ab} .v_{ib}) - {\frac{\sum_{x}exp(\sum_{y} q_{ay}k_{xy})}{L_{a}}}.\sum_{b} (df_{ab} . v_{xb})) .q_{aj}$

### 行形式

$\frac {\partial f(attention(q))}{\partial k_{ij}} = \sum_{a} \frac{{exp(q_{a}k_{i}^{T})}}{L_{a}} . (\sum_{b} (df_{a}v_{i}^{T}) - {\frac{\sum_{x}exp(q_{a}k_{x}^{T})}{L_{a}}}.(df_{a}v_{x}^{T})) .q_{aj}$

$\frac {\partial f(attention(q))}{\partial k_{i}} = \sum_{a} \frac{{exp(q_{a}k_{i}^{T})}}{L_{a}} . (\sum_{b} (df_{a}v_{i}^{T}) - {\frac{\sum_{x}exp(q_{a}k_{x}^{T})}{L_{a}}}.(df_{a}v_{x}^{T})) .q_{a}$