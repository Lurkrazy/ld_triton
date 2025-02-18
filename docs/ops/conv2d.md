# forward

[tensorflow conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)

$output_{b,c_{out},h_{out},w_{out}} = \sum_{c_{in}, di, dj} input_{b,c_{in},strh * h_{out} + di, strw * w_{out} + dj} * weight_{c_{out}, c_{in}, di,dj}$

# backward

## 求导
## 通用求导

$\frac{\partial ax}{\partial x} = a$

### input求导

$\frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial input_{b,c_{in},h_{in},w_{in}}}$


$= \frac {\partial \sum_{c_{in1}, di, dj} input_{b1,c_{in1},strh * h_{out1} + di, strw * w_{out1} + dj} * weight_{c_{out1}, c_{in1}, di,dj}}{\partial input_{b,c_{in},h_{in},w_{in}}}$

##### $b \neq b1$ or 
<p> $h_{in} \notin (strh*h_{out1},strh*h_{out1}+R) or w_{in} \notin (strw*w_{out1},strw*h_{out1}+S)$  </p>

$\frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial input_{b,c_{in},h_{in},w_{in}}} = 0$ 

##### $b = b1$ and 
<p> $h_{in} \in (strh*h_{out1},strh*h_{out1}+R), w_{in} \in (strw*w_{out1},strw*h_{out1}+S)$ </p>

$\frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial input_{b,c_{in},h_{in},w_{in}}}$


$= \frac {\partial \sum_{c_{in1}, di, dj} input_{b,c_{in1},strh * h_{out1} + di, strw * w_{out1} + dj} * weight_{c_{out1}, c_{in1}, di,dj}}{\partial input_{b,c_{in},h_{in},w_{in}}}$

$= \frac {\partial \sum_{di, dj} input_{b,c_{in},strh * h_{out1} + di, strw * w_{out1} + dj} * weight_{c_{out1}, c_{in}, di,dj}}{\partial input_{b,c_{in},h_{in},w_{in}}}$

$= weight_{c_{out1}, c_{in}, di,dj}$

### weight求导

$\frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial weight{c_{out},c_{in},d_{i},d_{j}}}$

$= \frac {\partial \sum_{c_{in1}, di1, dj1} input_{b1,c_{in1},strh * h_{out1} + di1, strw * w_{out1} + dj1} * weight_{c_{out1}, c_{in1}, di1,dj1}}{\partial weight{c_{out},c_{in},d_{i},d_{j}}}$

#### $c_{out_1} \neq c_{out}$ or $c_{in_1} \neq c_{in}$ 

$\frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial weight{c_{out},c_{in},d_{i},d_{j}}} = 0$

#### $c_{out_1} = c_{out}, c_{in_1} = c_{in}$

$\frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial weight{c_{out},c_{in},d_{i},d_{j}}}$


$= \frac {\partial \sum_{di1, dj1} input_{b1,c_{in},strh * h_{out} + di1, strw * w_{out} + dj1} * weight_{c_{out}, c_{in}, di1,dj1}}{\partial weight{c_{out},c_{in},d_{i},d_{j}}}$

$=  input_{b1,c_{in},strh * h_{out1} + di, strw * w_{out1} + dj}$

### bias求导

$\frac{\partial out(N_p, C_{out_q})}{\partial bias(C_{out_j})}$

$=\frac{\partial bias(C_{out_q})}{\partial bias(C_{out_j})}$

#### $out_q \neq out_j$

$\frac{\partial out(N_p, C_{out_q})}{\partial bias(C_{out_j})}=0$

#### $out_q = out_j$

$\frac{\partial out(N_p, C_{out_q})}{\partial bias(C_{out_j})}=\frac{\partial out(N_p, C_{out_j})}{\partial bias(C_{out_j})}=1$

## 链式法则

### input链式法则

#### 元素形式

$\frac{\partial f(convolution(input))}{\partial {\partial input_{b,c_{in},h_{in},w_{in}}}}$

$=\sum_{b1,c_{out1},h_{out1},w_{out1}}\frac{\partial f(convolution(input))}{\partial output_{b1,c_{out1},h_{out1},w_{out1}}} * \frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial input_{b,c_{in},h_{in},w_{in}}}$

$=\sum_{b1,c_{out1},h_{out1},w_{out1}} df_{b1,c_{out1},h_{out1},w_{out1}} * \frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial input_{b,c_{in},h_{in},w_{in}}}$

$=\sum_{c_{out1},h_{out1},w_{out1}} df_{b,c_{out1},h_{out1},w_{out1}} * \frac{\partial output_{b,c_{out1},h_{out1},w_{out1}}}{\partial input_{b,c_{in},h_{in},w_{in}}}$

$=\sum_{c_{out1},h_{out1},w_{out1}} df_{b,c_{out1},h_{out1},w_{out1}} * \frac {\partial \sum_{di1, dj1} input_{b,c_{in},strh * h_{out1} + di1, strw * w_{out1} + dj1} * weight_{c_{out1}, c_{in}, di1,dj1}}{\partial input_{b,c_{in},h_{in},w_{in}}}$

$=\sum_{c_{out1},h_{out1},w_{out1}} df_{b,c_{out1},h_{out1},w_{out1}} * \frac {\partial \sum_{di1, dj1} input_{b,c_{in},strh * h_{out1} + di1, strw * w_{out1} + dj1} * weight_{c_{out1}, c_{in}, di1,dj1}}{\partial input_{b,c_{in},strh * h_{out} + di,strw * w_{out} + dj}}$


#### 矩阵形式

### weight链式法则

#### 元素形式

$\frac{\partial f(convolution(weight))}{\partial weight_{c_{out},c_{in},d_{i},d_{j}}}$

<!-- \frac{\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial weight{c_{out},c_{in},d_{i},d_{j}}} -->

$=\sum_{b1,c_{out1},h_{out1},w_{out1}}\frac{\partial f(convolution(weight))}{\partial output_{b1,c_{out1},h_{out1},w_{out1}}} * \frac {\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial weight_{c_{out},c_{in},d_{i},d_{j}}}$


$=\sum_{b1,c_{out1},h_{out1},w_{out1}}  df_{b1,c_{out1},h_{out1},w_{out1}} * \frac {\partial output_{b1,c_{out1},h_{out1},w_{out1}}}{\partial weight_{c_{out},c_{in},d_{i},d_{j}}}$

$=\sum_{b1,h_{out1},w_{out1}}  df_{b1,c_{out},h_{out1},w_{out1}} * \frac {\partial output_{b1,c_{out},h_{out1},w_{out1}}}{\partial weight_{c_{out},c_{in},d_{i},d_{j}}}$


### bias链式法则

#### 元素形式

<p>
$\frac{\partial f(convolution(bias)_{(N_k,C_{out_l})})}{\partial bias(C_{out_j})}$
</p>

<p>
$=\sum_{N_p}\sum_{C_{out_q}}\frac{\partial f(convolution(bias)_{(N_k,C_{out_l})})}{\partial convolution(bias)_{(N_p,C_{out_q})}} * \frac {convolution(bias)_{(N_p,C_{out_q})}}{\partial bias(C_{out_j})}$
</p>

<p>
$=\sum_{N_p}\sum_{C_{out_q}} df(N_p,C_{out_q}) * \frac {convolution(bias)_{(N_p,C_{out_q})}}{\partial bias(C_{out_j})}$
</p>

<p>
$=\sum_{N_p} df(N_p,C_{out_j}) * \frac {convolution(bias)_{(N_p,C_{out_j})}}{\partial bias(C_{out_j})}$
</p>

<p>
$=\sum_{N_p} df(N_p,C_{out_j})$
</p>

#### 矩阵形式

$\frac{\partial f(convolution(bias))}{\partial bias}$

$=sum(df, (dim(N_p)...))$


# reference

https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf

https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cpp

https://github.com/pjreddie/darknet/blob/master/src/convolutional_layer.c

https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md
