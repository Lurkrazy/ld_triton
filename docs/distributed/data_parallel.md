
# sgd

for simple assum $X \in R^{1}$

$L(θ;X,Y) = \frac{1}{N}\sum_{i=0}^{N-1}J(θ;x_{i}, y_{i})$

$∇_{θ}L(θ;X,Y) = \frac{1}{N}\sum_{i=0}^{N-1}∇_{θ}J(θ;x_{i}, y_{i})$

$θ=θ–η∇_{θ}L(θ) = θ–\frac{η}{N}\sum_{i=0}^{N-1}∇_{θ}J(θ;x_{i}, y_{i})$

# chain rule
$L(θ;X,Y) = \frac{1}{N}\sum_{i=0}^{N-1}J(θ;x_{i}, y_{i})$

$L(f(θ;X,Y)) = \frac{1}{N}\sum_{i=0}^{N-1}J(f(θ;x_{i}, y_{i}))$

$\frac{\partial L(f(θ;X,Y))}{\partial θ}$

$=\frac{1}{N}\sum_{i=0}^{N-1}\frac{\partial J(f(θ;x_{i}, y_{i}))}{\partial θ}$

$=\frac{1}{N}\sum_{i=0}^{N-1}\frac{\partial J(f(θ;x_{i}, y_{i}))}{\partial f(θ;x_{i}, y_{i})}\frac{\partial f(θ;x_{i}, y_{i})}{\partial θ}$

# distributed sgd

$L^{nodes_{i}}(θ;X^{nodes_{i}},Y^{nodes_{i}}) = \frac{1}{N}\sum_{j=0}^{N-1}J(θ;x_{j}^{nodes_{i}}, y_{j}^{nodes_{i}})$

<p>
$L(θ;X,Y) = \frac{1}{nodes*N}\sum_{j=0}^{nodes*N-1}J(θ;x_{j}, y_{j})$
</p>

$= \frac{1}{nodes*N}(\sum_{i=0}^{N-1}J(θ;x_{i}, y_{i}) + \sum_{i=N}^{2N-1}J(θ;x_{i}, y_{i}) + ... + \sum_{i=(nodes-1) * N}^{nodes * N-1}J(θ;x_{i}, y_{i}))$

$= \frac{1}{nodes}(\frac{1}{N}\sum_{i=0}^{N-1}J(θ;x_{i}, y_{i}) + \frac{1}{N}\sum_{i=N}^{2N-1}J(θ;x_{i}, y_{i}) + ... + \frac{1}{N}\sum_{i=(nodes-1) * N}^{nodes * N-1}J(θ;x_{i}, y_{i}))$

$= \frac{1}{nodes}(L^{nodes_{0}}(θ;X^{nodes_{0}},Y^{nodes_{0}}) + L^{nodes_{1}}(θ;X^{nodes_{1}},Y^{nodes_{1}}) + ... + L^{nodes-1}(θ;X^{nodes-1},Y^{nodes-1}))$

$= \frac{1}{nodes}\sum_{i=0}^{nodes-1}L^{nodes_{i}}(θ;X^{nodes_{i}},Y^{nodes_{i}})$

$∇_{θ}L(θ;X,Y) = \frac{1}{nodes}\sum_{i=0}^{nodes-1}∇_{θ}L^{nodes_{i}}(θ;X^{nodes_{i}},Y^{nodes_{i}})$

# distributed chain rule

$\frac{\partial L^{nodes_{i}}(f(θ;X^{nodes_{i}},Y^{nodes_{i}}))}{\partial θ}$

$=\frac{1}{N}\sum_{j=0}^{N-1} \frac{\partial J(f(θ;x_{j}^{nodes_{i}},x_{j}^{nodes_{i}}))}{\partial θ}$

$=\frac{1}{N}\sum_{j=0}^{N-1} \frac{\partial J(f(θ;x_{j}^{nodes_{i}},x_{j}^{nodes_{i}}))}{\partial f(θ;x_{j}^{nodes_{i}},x_{j}^{nodes_{i}})}\frac{f(θ;x_{j}^{nodes_{i}},x_{j}^{nodes_{i}})}{\partial θ}$

$\frac{\partial L(f(θ;X,Y))}{\partial θ}$

<p>
$=\frac{1}{nodes*N}\sum_{j=0}^{nodes*N-1}\frac{\partial J(f(θ;x_{j}, y_{j}))}{\partial θ}$
</p>

<p>
$=\frac{1}{nodes*N}\sum_{j=0}^{nodes*N-1}\frac{\partial J(f(θ;x_{j}, y_{j}))}{\partial f(θ;x_{j}, y_{j})}\frac{\partial f(θ;x_{j}, y_{j})}{\partial θ}$
</p>

<p>
$=\frac{1}{nodes*N}\sum_{j=0}^{N-1}(\frac{\partial J(f(θ;x_{j}, y_{j}))}{\partial f(θ;x_{j}, y_{j})}\frac{\partial f(θ;x_{j}, y_{j})}{\partial θ} + ...+\sum_{j=(nodes-1)*N}^{nodes*N-1}\frac{\partial J(f(θ;x_{j}, y_{j}))}{\partial f(θ;x_{j}, y_{j})}\frac{\partial f(θ;x_{j}, y_{j})}{\partial θ})$
</p>

<p>
$=\frac{1}{nodes}(\frac{1}{N}\sum_{j=0}^{N-1}\frac{\partial J(f(θ;x_{j}, y_{j}))}{\partial f(θ;x_{j}, y_{j})}\frac{\partial f(θ;x_{j}, y_{j})}{\partial θ} + ...+\frac{1}{N}\sum_{j=(nodes-1)*N}^{nodes*N-1}\frac{\partial J(f(θ;x_{j}, y_{j}))}{\partial f(θ;x_{j}, y_{j})}\frac{\partial f(θ;x_{j}, y_{j})}{\partial θ})$
</p>

<p>
$=\frac{1}{nodes}(\frac{\partial L^{nodes_{0}}(f(θ;X^{nodes_{0}},Y^{nodes_{0}}))}{\partial θ} + ...+\frac{\partial L^{nodes-1}(f(θ;X^{nodes-1},Y^{nodes-1}))}{\partial θ})$
</p>

# AllReduce

# References

[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704)

[Data-Parallel Distributed Training of Deep Learning Models](https://siboehm.com/articles/22/data-parallel-training)