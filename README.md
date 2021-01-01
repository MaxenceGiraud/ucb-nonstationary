# On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems


Implementation of the paper by Aurélien Garivier and Eric Moulines, On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems [1].

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

gaussian_arm_list = [Gaussian(x) for x in np.random.uniform(0,10,100)]
nonstat_bandit = MAB_NS(n_arms = 5,arms_list = gaussian_arm_list,p=0.05)

# (Algos are upcoming features)
```

## TODO
- [x] Implement non stationary Bandit
- [x] Discounted UCB
- [ ] Sliding-Window UCB

## References
[1] Garivier, Aurélien & Moulines, Eric. (2008). On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems. 