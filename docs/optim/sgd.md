
# sgd

$L(θ) = \frac{1}{N}\sum_{i=0}^{N-1}J(θ;x_{i}, y_{i})$

$∇_{θ}L(θ) = \frac{1}{N}\sum_{i=0}^{N-1}∇_{θ}J(θ;x_{i}, y_{i})$

$θ=θ–η∇_{θ}L(θ) = θ–\frac{η}{N}\sum_{i=0}^{N-1}∇_{θ}J(θ;x_{i}, y_{i})$

## 
# distributed sgd

$L_{nodes_{i}}(θ) = \frac{1}{N}\sum_{j=0}^{N-1}J(θ;x_{j}^{nodes_{i}}, y_{j}^{nodes_{i}})$

<p>
$L(θ) = \frac{1}{nodes*N}\sum_{j=0}^{nodes*N-1}J(θ;x_{j}, y_{j})$
</p>

$= \frac{1}{nodes*N}(\sum_{i=0}^{N-1}J(θ;x_{i}, y_{i}) + \sum_{i=N}^{2N-1}J(θ;x_{i}, y_{i}) + ... + \sum_{i=(nodes-1) * N}^{nodes * N-1}J(θ;x_{i}, y_{i}))$

$= \frac{1}{nodes}(\frac{1}{N}\sum_{i=0}^{N-1}J(θ;x_{i}, y_{i}) + \frac{1}{N}\sum_{i=N}^{2N-1}J(θ;x_{i}, y_{i}) + ... + \frac{1}{N}\sum_{i=(nodes-1) * N}^{nodes * N-1}J(θ;x_{i}, y_{i}))$

$= \frac{1}{nodes}(L_{nodes_{0}}(θ) + L_{nodes_{1}}(θ) + ... + L_{nodes_{nodes-1}}(θ))$

$= \frac{1}{nodes}\sum_{i=0}^{nodes-1}L_{nodes_{i}}$

$∇_{θ}L(θ) = \frac{1}{nodes}\sum_{i=0}^{nodes-1}∇_{θ}L_{nodes_{i}}$

# AllReduce

# References

[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704)

[Data-Parallel Distributed Training of Deep Learning Models](https://siboehm.com/articles/22/data-parallel-training)