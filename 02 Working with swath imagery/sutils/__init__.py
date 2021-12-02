from dataclasses import dataclass

import numpy as np
from matplotlib import ticker
from scipy import stats


@dataclass
class ReturnValue:
    """
    Function return values
     r:  float
        correlation coefficient
     p:  float
        p-value for the test of the correlation coefficient
    """
    xp: np.array
    yp: np.array
    desc: str


def regress(x, y, scale: str = 'log-log'):
    """
    https://en.wikipedia.org/wiki/Log%E2%80%93log_plot
    :param x:
    :param y:
    :param scale:
    :return:
    """

    if scale == 'log-log':
        xi = np.log10(x)
        yi = np.log10(y)
    else:
        xi = np.asarray(x)
        yi = np.asarray(y)

    f = np.poly1d(np.polyfit(xi, yi, 1))
    m, b = f.coeffs

    if scale == 'log-log':
        xx = np.logspace(np.min(xi), np.max(xi), 100)
        yy = (xx ** m) * (10 ** b)
        bias = np.power(10, np.sum(yi - xi) / xi.size)
    else:
        xx = np.linspace(np.min(xi), np.max(xi), 100)
        yy = xx * m + b
        bias = np.sum(yi - xi) / xi.size

    r, p = stats.spearmanr(xi, yi)
    p = 0.001 if p < 0.001 else (
        0.01 if p < 0.01 else (
            0.05 if p < 0.05 else p))
    p = float(f'{p:.3f}')

    if b > 0:
        s = f'+ {b:.3f}'
    else:
        s = f'{b:.3f}'

    # predict y values of origional data using the fit
    p_y = f(xi)
    # calculate the y-error (residuals)
    yp = yi - p_y
    # sum of the squares of the residuals | SSE
    sse = np.sum(np.power(yp, 2))
    # sum of the squares total | SST
    sst = np.sum(np.power(yi - yi.mean(), 2))
    # r-squared
    rsq = 1 - sse / sst

    base = r'\log_{10}'
    if scale == 'log-log':
        txt = f'${base} (y)={m:.3f} \\times {base} (x) {s}$\n' \
              f'$N={xi.size}$\n$R^{2}={rsq:.2f}$\n$r={r:.2f}$\n' \
              f'$p<{p:g}$\n$\\delta={bias:.2f}$'
        return ReturnValue(xp=xx, yp=yy, desc=txt)

    txt = f'$y={m:.3f} x {s}$\n' \
          f'$N={xi.size}$\n$R^{2}={rsq:.2f}$\n$r={r:.2f}$\n' \
          f'$p<{p:g}$\n$\\delta={bias:.2f}$'
    return ReturnValue(xp=xx, yp=yy, desc=txt)


def logticks(ax, n, axes: str = 'both'):
    # set y ticks
    subs = np.hstack((np.arange(2, 10) * 10 ** -2,
                      np.arange(2, 10) * 10 ** -1,
                      np.arange(2, 10) * 10 ** 0,
                      np.arange(2, 10) * 10 ** 1))
    major = ticker.LogLocator(base=10, numticks=5)
    minor = ticker.LogLocator(base=10, subs=subs, numticks=10)

    if axes == 'both':
        ax.xaxis.set_major_locator(major)
        ax.yaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_xticks())])
        ax.set_yticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_yticks())])

    if axes == 'x':
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_xticks())])
    if axes == 'y':
        ax.yaxis.set_major_locator(major)
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_yticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_yticks())])
    return


def underestimated(x, y, ratio):
    idx = np.where(x > y * ratio)
    return x[idx], y[idx]


def overestimated(x, y, ratio):
    idx = np.where(y > x * ratio)
    return x[idx], y[idx]


def identity_line(ax, lines: list):
    # Add 1:1, 1:2, 1:3 or 2:1, 3:1 lines
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    for r in lines:
        if r == 1:
            ax.plot([x0, x1], [y0, y1], '-r', linewidth=2, label='$x = y$')
        if r == 2:
            ax.plot([x0 * r, x1], [y0, y1 / r], ':k', linewidth=2)
            ax.plot([x0, x1 / r], [y0 * r, y1], ':k', linewidth=2)
        if r == 3:
            ax.plot([x0 * r, x1], [y0, y1 / r], '-', color='#2F5597', linewidth=2)
            ax.plot([x0, x1 / r], [y0 * r, y1], '-', color='#2F5597', linewidth=2)

    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    return


# def in_water(df, sensor: str):
#     if sensor in ('MODIS-Aqua', 'aqua'):
#         # https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/
#         a0, a1, a2, a3, a4 = 0.2424, -2.7423, 1.8017, 0.0015, -1.2280
#
#         idx = df['Variable [Units]'].isin(['Rrs_443 [sr^-1]'])
#         rrs443 = df.loc[idx, 'Median']
#         idx = df['Variable [Units]'].isin(['Rrs_488 [sr^-1]'])
#         rrs488 = df.loc[idx, 'Median']
#         idx = df['Variable [Units]'].isin(['Rrs_547 [sr^-1]'])
#         rrs547 = df.loc[idx, 'Median']
#         r = np.log10(np.maximum(rrs443.to_numpy(), rrs488.to_numpy()) / rrs547.to_numpy())
#         print(f'r: {r}')
#         return a0 + a1 * r + a2 * (r ** 2) + a3 * (r ** 3) + a4 * (r ** 4)
#
#     if sensor == 'sgli':
#         return
