# Bayes Polynomial Test

This is a simple python module using the [emcee](http://dan.iel.fm/emcee/current/) 
MCMC implementation to estimate the Bayes factor between polynomial fits to
data. 

## Example

For a simple example, see the file `test.py`:

```python 
from BayesPolynomialTest import BayesPolynomialTest
import numpy as np

N = 10
m = 3
c = 10
x = np.linspace(0, 1, N)
y = m*x + c + np.random.normal(0, 0.5, N)
test = BayesPolynomialTest(x, y, degrees=[1, 2], ntemps=120, betamin=-6,
                           nburn0=50, nburn=50, nprod=50,
                           unif_lim=100)
test.diagnostic_plot()
test.BayesFactor()
```
