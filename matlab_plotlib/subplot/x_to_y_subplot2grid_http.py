import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.style as style


style.use("bmh")


def moving_average(values, window):
    weights = np.repeat(1.0 / (window * 1.0), window)
    res = np.convolve(values, weights, "valid")
    return res


