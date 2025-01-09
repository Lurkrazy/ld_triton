# forward

$out(N_i, C_{out_j}) = bias(C_{out_j}) + \sum_{k=0}^{C_{in}-1} weight(C_{out_j, k}) * input(N_i, k)$

# backward

## 求导
## 通用求导

$\frac{\partial ax}{\partial x} = a$

### input求导

$\frac{\partial out(N_p, C_{out_q})}{\partial input(N_i, j)}$

<p>
$= \frac {\partial \sum_{k=0}^{C_{in}-1} weight(C_{out_q, k}) * input(N_p, k)}{\partial input(N_i,j)}$
</p>

##### $N_p \neq N_i$

$\frac{\partial out(N_p, C_{out_q})}{\partial input(N_i,j)} = 0$

##### $N_p = N_i$

$\frac{\partial out(N_p, C_{out_q})}{\partial input(N_i,j)}$

<p>
$= \frac {\partial \sum_{k=0}^{C_{in}-1} weight(C_{out_q, k}) * input(N_i, k)}{\partial input(N_i,j)}$
</p>

$= weight(C_{out_q, j})$

### weight求导

$\frac{\partial out(N_p, C_{out_q})}{\partial weight(C_{out_i},j)}$

<p>
$= \frac {\partial \sum_{k=0}^{C_{in}-1} weight(C_{out_q, k}) * input(N_p, k)}{\partial weight(C_{out_i},j)}$
</p>

#### $C_{out_q} \neq C_{out_i}$

$\frac{\partial out(N_p, C_{out_q})}{\partial weight(C_{out_i},j)}=0$

#### $C_{out_q} = C_{out_i}$

$\frac{\partial out(N_p, C_{out_q})}{\partial weight(C_{out_i},j)}$

<p>
$= \frac {\partial \sum_{k=0}^{C_{in}-1} weight(C_{out_i, k}) * input(N_p, k)}{\partial weight(C_{out_i},j)}$
</p>

$= input(N_p, j)$

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

<p>
$\frac{\partial f(convolution(input)_{(N_k,C_{out_l})})}{\partial input(N_i,j)}$
</p>

<p>
$=\sum_{N_p}\sum_{C_{out_q}}\frac{\partial f(convolution(input)_{(N_k,C_{out_l})})}{\partial convolution(input)_{(N_p,C_{out_q})}} * \frac {convolution(input)_{(N_p,C_{out_q})}}{\partial input(N_i,j)}$
</p>

<p>
$=\sum_{N_p}\sum_{C_{out_q}}df{(N_p,C_{out_q})} * \frac {convolution(input)_{(N_p,C_{out_q})}}{\partial input(N_i,j)}$
</p>

$=\sum_{C_{out_q}}df{(N_i,C_{out_q})}  * weight(C_{out_q}, j)$

$=\sum_{C_{out_q}} weight(C_{out_q}, j) * df{(N_i,C_{out_q})}$

#### 矩阵形式

### weight链式法则

#### 元素形式

<p>
$\frac{\partial f(convolution(weight)_{(N_k,C_{out_l})})}{\partial weight(C_{out_i},j)}$
</p>

<p>
$=\sum_{N_p}\sum_{C_{out_q}}\frac{\partial f(convolution(weight)_{(N_k,C_{out_l})})}{\partial convolution(weight)_{(N_p,C_{out_q})}} * \frac {convolution(weight)_{(N_p,C_{out_q})}}{\partial weight(C_{out_i},j)}$
</p>

<p>
$=\sum_{N_p}\sum_{C_{out_q}}df(N_p,C_{out_q}) * \frac {convolution(weight)_{(N_p,C_{out_q})}}{\partial weight(C_{out_i},j)}$
</p>

<p>
$=\sum_{N_p}df(N_p,C_{out_i}) * \frac {convolution(weight)_{(N_p,C_{out_i})}}{\partial weight(C_{out_i},j)}$
</p>

$=\sum_{N_p}df(N_p,C_{out_i}) * input(N_p,j)$
