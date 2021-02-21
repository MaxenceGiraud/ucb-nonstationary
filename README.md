# On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems


Implementation of the paper by Aurélien Garivier and Eric Moulines, On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems [1]. We also try some variants of the algorithms and compare them together.

Our experiments with the different algorithms are compiled in the notebook [experiements.ipynb](./experiments.ipynb)/

## Installation
To install simply clone the project : 
```bash
git clone https://github.com/MaxenceGiraud/ucb-nonstationary
cd ucb-nonstationary/
```
## Usage
```python
import numpy as np
import nsucb
from bandit_env import *

# Arms sequence
def arm_f(t):
    arms = [Bernoulli(0.5),Bernoulli(0.1),Bernoulli(0.4)]
    if t> 300 and t<500 :
        arms[1] = Bernoulli(0.9)
    return arms 

n=3 # nb of arms
mab = MAB_NS(3,arm_f)

# Algorithms
ucb = nsucb.UCB(n)
d= nsucb.DiscountedUCB(n)
sw= nsucb.SlidingUCB(n)

# Run simulations
RunExpes([ucb,d,sw],mab,50,T,non_stationary=True,quantiles=False)
```

To compile the report, you will need latex installed and an appropriate compiler, then you can simply :
```bash
cd report/
pdflatex main.tex
```

## TODO
- [x] Implement non stationary Bandit
- [x] Discounted UCB
- [x] Sliding-Window UCB

## References
[1] Garivier, Aurélien & Moulines, Eric. (2008). On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems. 
