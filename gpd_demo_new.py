import numpy as np
from scipy.stats import genpareto as gpd
from scipy.stats import expon 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

def fit_gpd2(X, qthre = 75):
    threshold = np.percentile(X, qthre)
    X = X[X > threshold]
    X.sort()
    ecdf = np.arange(X.shape[0]) / X.shape[0] 
    pars, cov = curve_fit(lambda x, ksi, sigma: gpd.cdf(X, c = ksi, loc=threshold, scale=sigma), X, ecdf, p0 = [-1.137, 159.56], maxfev = 10000)

    dist2 = gpd(c = pars[0], loc = threshold, scale = pars[1])
    return dist2


rainc = -np.loadtxt('rainc.txt')
rainc.sort()
# xx = rainc
xx = rainc[:-1]

genp = fit_gpd2(xx, qthre = 75)
N = xx.shape[0]
ecdf = np.arange(1, N+1) / (N+1)
qthre = 75
threshold = np.percentile(xx, qthre)
# genp, _ = fit_gpd(xx, threshold)

xxf = np.linspace(threshold, -440, 1000)
yyf = 1 - genp.cdf(xxf)

fig, ax = plt.subplots()
ax.plot(1 / (1-ecdf), -xx, marker='.', linestyle='none', label = 'Sample Data')
ax.plot(1/(1-qthre/100)/yyf, -xxf, color='red', label='GPD Fit c != 0')
ax.plot([0, 2000], [-rainc[-2], -rainc[-2]], color='grey', linestyle= 'dashed')
ax.plot([0, 2000], [-rainc[-1], -rainc[-1]], color='grey', linestyle= 'dashed')
ax.set_xlim([0.9, 1500])
ax.set_xscale('log')
pause = 1



# # generate samples from a normal distribution
# N = 10000
# xx = 5 * np.random.randn(N) + 20
# xx.sort()
# ecdf = np.arange(1, N+1) / (N+1)
# fig, ax = plt.subplots()
# ax.plot(1 / (1-ecdf), xx, marker='.', linestyle='none', label = 'Sample Data')

# ax.set_xscale('log')
# ax.set_xlabel('Return Period')
# ax.set_ylabel('Value')

# # get the qth percentile of xx
# qthre = 75
# threshold = np.percentile(xx, qthre)
# # fit model
# genp, _ = fit_gpd(xx, threshold)
# genp2, _ = fit_gpd(xx, threshold, True)
# xxf = np.linspace(threshold, xx.max() + 2, 1000)
# yyf = 1 - genp.cdf(xxf)
# yyf2 = 1 - genp2.cdf(xxf)
# pause = 1

# ax.plot(1/(1-qthre/100)/yyf, xxf, color='red', label='GPD Fit c != 0')
# ax.plot(1/(1-qthre/100)/yyf2, xxf, color='green', label='GPD Fit c = 0')
# plt.show()
# fig.savefig('sample_data.png', dpi=300)
# pause = 1



