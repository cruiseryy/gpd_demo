import numpy as np
from scipy.stats import genpareto as gpd
from scipy.stats import expon 
import matplotlib.pyplot as plt

# set random seed
# np.random.seed(1)

# fit a generalized pareto distribution to the tail of some data samples X
def fit_gpd(X, threshold=0, cflag = False):
    # select observations greater than threshold
    X = X[X > threshold]
    # estimate parameters
    if not cflag:
        params = gpd.fit(X, loc = threshold)
        dist = gpd(*params)
    else:
        # fit with c = 0
        params = expon.fit(X, loc = threshold)
        dist = expon(*params)
    return dist, params

# generate samples from a normal distribution
N = 10000
xx = 5 * np.random.randn(N) + 20
xx.sort()
ecdf = np.arange(1, N+1) / (N+1)
fig, ax = plt.subplots()
ax.plot(1 / (1-ecdf), xx, marker='.', linestyle='none', label = 'Sample Data')

ax.set_xscale('log')
ax.set_xlabel('Return Period')
ax.set_ylabel('Value')

# get the qth percentile of xx
qthre = 75
threshold = np.percentile(xx, qthre)
# fit model
genp, _ = fit_gpd(xx, threshold)
genp2, _ = fit_gpd(xx, threshold, True)
xxf = np.linspace(threshold, xx.max() + 2, 1000)
yyf = 1 - genp.cdf(xxf)
yyf2 = 1 - genp2.cdf(xxf)
pause = 1

ax.plot(1/(1-qthre/100)/yyf, xxf, color='red', label='GPD Fit c != 0')
ax.plot(1/(1-qthre/100)/yyf2, xxf, color='green', label='GPD Fit c = 0')
plt.show()
fig.savefig('sample_data.png', dpi=300)
pause = 1



