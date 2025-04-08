# global gradient

for simple assum $X \in R^{1}$

$GBS = global\_batch\_size$

$MBS = micro\_batch\_size$

$MB = num\_micro\_batch = \frac{GBS}{MBS}$

$L^{global}(θ;X,Y) = \frac{1}{GBS}\sum_{i=0}^{GBS-1}J(θ;x_{i}, y_{i})$

$∇_{θ}L^{global}(θ;X,Y) = \frac{1}{GBS}\sum_{i=0}^{GBS-1}∇_{θ}J(θ;x_{i}, y_{i})$



# micro gradient

$L^{micro}(θ;X,Y) = \frac{1}{MBS}\sum_{i=0}^{MBS-1}J(θ;x_{i}, y_{i})$

$∇_{θ}L^{micro}(θ;X,Y) = \frac{1}{MBS}\sum_{i=0}^{MBS-1}∇_{θ}J(θ;x_{i}, y_{i})$

# sgd
$∇_{θ}L^{global}(θ;X,Y)$

$= \frac{1}{GBS}\sum_{i=0}^{GBS-1}∇_{θ}J(θ;x_{i}, y_{i})$

$= \frac{1}{MBS * MB}\sum_{i=0}^{MBS * MB-1}∇_{θ}J(θ;x_{i}, y_{i})$

$= \frac{1}{MB} (\frac{1}{MBS} \sum_{i=0}^{MBS * MB-1}∇_{θ}J(θ;x_{i}, y_{i}))$

$= \frac{1}{MB} (\frac{1}{MBS} \sum_{i=0}^{MBS-1}∇_{θ}J(θ;x_{i}, y_{i}) + \frac{1}{MBS} \sum_{i=MBS}^{2 * MBS-1}∇_{θ}J(θ;x_{i}, y_{i}) + ...... + \frac{1}{MBS} \sum_{i=(MB-1)*MBS}^{MB * MBS-1}∇_{θ}J(θ;x_{i}, y_{i}))$

$= \frac{1}{MB} (∇_{θ}L_{0}^{micro}(θ;X,Y) + ∇_{θ}L_{1}^{micro}(θ;X,Y) + ...... + ∇_{θ}L_{MB-1}^{micro}(θ;X,Y))$

$= \frac{1}{MB} \sum_{i=0}^{MB-1}∇_{θ}L_{i}^{micro}(θ;X,Y)$


$θ=θ–η∇_{θ}L^{global}(θ)$

$= θ–\frac{η}{GBS}\sum_{i=0}^{GBS-1}∇_{θ}J(θ;x_{i}, y_{i})$

$= θ–\frac{η}{MB} \sum_{i=0}^{MB-1}∇_{θ}L_{i}^{micro}(θ;X,Y)$