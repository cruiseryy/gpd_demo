from typing import Any
import numpy as np
from scipy.stats import genpareto as gpd
from scipy.stats import expon 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class gpd_fit:
    def __init__(self, X) -> None:
        self.X = X
        pass

    def fit_gpd2(self, qthre = 75):
        threshold = np.percentile(self.X, qthre)
        X = self.X[self.X > threshold]
        X.sort()
        ecdf = np.arange(X.shape[0]) / X.shape[0] 
        para0 = gpd.fit(X, loc = threshold)
        pars, cov = curve_fit(lambda x, ksi, sigma: gpd.cdf(X, c = ksi, loc=threshold, scale=sigma), X, ecdf, p0 = [para0[0], para0[2]], maxfev = 10000)
        dist = gpd(c = pars[0], loc = threshold, scale = pars[1])
        return dist