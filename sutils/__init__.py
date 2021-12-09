import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests
import scipy.io as sio
from matplotlib import ticker
from scipy import stats
from tqdm import tqdm


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


def coastline(file: Path = None):
    key = 'ncst'
    if file is None:
        file = '{}/{}'.format(Path().parent,
                              'sample_data/nowpap_sea.mat')
        # print(file)
    cline = sio.loadmat(file, variable_names=[key], squeeze_me=True).get(key)
    return cline[:, 0], cline[:, 1]


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
    pv = f"p<{float(f'{p:.3f}')}" if p < 0.05 else f"p={float(f'{p:.3f}')}"

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
              f'${pv}$\n$\\delta={bias:.2f}$'
        return ReturnValue(xp=xx, yp=yy, desc=txt)

    txt = f'$y={m:.3f} x {s}$\n' \
          f'$N={xi.size}$\n$R^{2}={rsq:.2f}$\n$r={r:.2f}$\n' \
          f'${pv}$\n$\\delta={bias:.2f}$'
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


def validation(x, y, plt, xlabel: str, ylabel: str, xscale: str = 'log',
               yscale: str = 'log', xlim=None, ylim=None, id_lines=None,
               s: int = 100, alpha: float = 0.4, marker: str = 'o',
               figure_name: str = None):
    if id_lines is None:
        id_lines = [1, 2, 3]
    if ylim is None:
        ylim = [0.01, 100]
    if xlim is None:
        xlim = [0.01, 100]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Display scatter
    ax.scatter(
        x, y,
        c='k',
        marker=marker,
        alpha=alpha,
        s=s,
    )

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the scale to log
    ax.set_xscale(xscale)
    ax.set_yscale(xscale)

    # Set axis limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Identity line
    identity_line(ax=ax, lines=id_lines)

    # Configure ticks
    logticks(ax=ax, n=2)

    # overestimate
    xov, yov = overestimated(x=x.to_numpy(),
                             y=y.to_numpy(),
                             ratio=id_lines[-1])
    ax.scatter(xov, yov, s=s, c='r', marker='o', alpha=0.5)

    # underestimate
    xun, yun = underestimated(x=x.to_numpy(),
                              y=y.to_numpy(),
                              ratio=id_lines[-1])
    ax.scatter(xun, yun, s=s, c='b', marker='o', alpha=0.5)

    # Regression line
    mask = np.isnan(x.to_numpy()) | np.isnan(y.to_numpy())
    result = regress(x=x.to_numpy()[~mask].flatten(),
                     y=y.to_numpy()[~mask].flatten(),
                     scale=f'{xscale}-{yscale}')
    rl = ax.plot(result.xp, result.yp, '-g', label='linear regression', lw=6)

    plt.legend(loc='lower right')
    t = ax.text(.015, 3, result.desc)

    if figure_name:
        plt.tight_layout()
        plt.savefig(figure_name, dpi=200)
    return


def mpl_custom(mpl):
    # custom image settings
    mpl.rcParams.update({
        'font.size': 25,
        # 'axes.grid': True,
        'axes.linewidth': 2,
        'grid.linewidth': 1,

        'xtick.major.size': 8,
        'xtick.major.width': 2,

        'xtick.minor.visible': True,
        'xtick.minor.size': 4,
        'xtick.minor.width': 1,

        'ytick.major.size': 8,
        'ytick.major.width': 2,

        'ytick.minor.visible': True,
        'ytick.minor.size': 4,
        'ytick.minor.width': 1,

        'savefig.facecolor': '#F5F5F5'
    })
    return


def bilinear_interp(src_geo: np.array, interval: int):
    """
        Bilinear interpolation of SGLI geo-location corners to a spatial grid

        Parameters
        ----------
        src_geo: np.array
            either lon or lat
        interval: int
            resampling interval in pixels

        Return
        ------
        out_geo: np.array
            2-D array with dims == to geophysical variables
    """
    sds = np.concatenate((src_geo, src_geo[-1].reshape(1, -1)), axis=0)
    sds = np.concatenate((sds, sds[:, -1].reshape(-1, 1)), axis=1)

    ratio_0 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
        (sds.shape[0] * interval, sds.shape[1] - 1))

    ratio_1 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32).reshape(-1, 1),
        (sds.shape[0] - 1, (sds.shape[1] - 1) * interval))

    sds = np.repeat(sds, interval, axis=0)
    sds = np.repeat(sds, interval, axis=1)
    interp = (1. - ratio_0) * sds[:, :-interval] + ratio_0 * sds[:, interval:]
    return (1. - ratio_1) * interp[:-interval, :] + ratio_1 * interp[interval:, :]


def navigation_data(file: Path, key: str):
    with h5py.File(file, 'r') as hdf:
        nsl = hdf['/Image_data'].attrs['Number_of_lines'][0]
        psl = hdf['/Image_data'].attrs['Number_of_pixels'][0]
        img_size = (slice(0, nsl), slice(0, psl))

        if 'lat' in key.lower():
            # Get Latitude
            lat = get_data(h5f=hdf, key='Latitude')
            interval = hdf['/Geometry_data/Latitude'].attrs['Resampling_interval'][0]
            lat = bilinear_interp(src_geo=lat, interval=interval)[img_size]
            return lat

        # Get Longitude
        lon = get_data(h5f=hdf, key='Longitude')
        interval = hdf['/Geometry_data/Longitude'].attrs['Resampling_interval'][0]
    is_stride_180 = False
    if np.abs(np.nanmin(lon) - np.nanmax(lon)) > 180.:
        is_stride_180 = True
        lon[lon < 0] = 360. + lon[lon < 0]
    lon = bilinear_interp(src_geo=lon, interval=interval)[img_size]

    if is_stride_180:
        lon[lon > 180.] = lon[lon > 180.] - 360.
    return lon


def get_data(h5f: h5py, key: str):
    data = h5f[f'Geometry_data/{key}'][:]
    attrs = dict(h5f[f'Geometry_data/{key}'].attrs)

    if 'Error_DN' in attrs.keys():
        data[data == attrs.pop('Error_DN')[0]] = np.NaN

    if ('Minimum_valid_DN' in attrs.keys()) and \
            ('Maximum_valid_DN' in attrs.keys()):
        valid_min = attrs.pop('Minimum_valid_DN')[0]
        valid_max = attrs.pop('Maximum_valid_DN')[0]
        data[(data < valid_min) | (data > valid_max)] = np.NaN

    # Convert DN to PV
    if ('Slope' in attrs.keys()) and ('Offset' in attrs.keys()):
        data = data * attrs.pop('Slope')[0] + attrs.pop('Offset')[0]

    if ('Minimum_valid_value' in attrs.keys()) and \
            ('Maximum_valid_value' in attrs.keys()):
        valid_min = attrs.pop('Minimum_valid_value')[0]
        valid_max = attrs.pop('Maximum_valid_value')[0]
        data[(data < valid_min) | (data > valid_max)] = np.NaN

    return data


def attr_fmt(h5: h5py, address: str):
    result = {}
    for key, val in h5[address].attrs.items():
        if key in ('Dim0', 'Dim1'):
            continue
        try:
            val = val[0]
        except IndexError:
            pass

        if type(val) in (bytes, np.bytes_):
            val = val.decode()
        result.update({key: val})

    desc = result['Data_description'] \
        if 'Data_description' in result.keys() else None
    if desc and ('Remote Sensing Reflectance(Rrs)' in desc):
        result['units'] = result['Rrs_unit']
    return result


def h5_read(file: Path, key: str):
    with h5py.File(file, 'r') as h5:
        if key == 'QA_flag':
            sds = np.ma.squeeze(h5[f'Image_data/{key}'][:])
            np.ma.set_fill_value(sds, 0)
            return sds

        fill_value = np.float32(-32767)
        attrs = attr_fmt(h5=h5, address=f'Image_data/{key}')
        sdn = h5[f'Image_data/{key}'][:]

    mask = False
    if 'Error_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Error_DN')), True, False)
    if 'Land_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Land_DN')), True, False)
    if 'Cloud_error_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Cloud_error_DN')), True, False)
    if 'Retrieval_error_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Retrieval_error_DN')), True, False)
    if ('Minimum_valid_DN' in attrs.keys()) and ('Maximum_valid_DN' in attrs.keys()):
        # https://shikisai.jaxa.jp/faq/docs/GCOM-C_Products_Users_Guide_entrylevel__attach4_jp_191007.pdf#page=46
        mask = mask | np.where((sdn <= attrs.pop('Minimum_valid_DN')) |
                               (sdn >= attrs.pop('Maximum_valid_DN')), True, False)

    # Convert DN to PV
    slope, offset = 1, 0
    if 'NWLR' in key:
        if ('Rrs_slope' in attrs.keys()) and \
                ('Rrs_slope' in attrs.keys()):
            slope = attrs.pop('Rrs_slope')
            offset = attrs.pop('Rrs_offset')
    else:
        if ('Slope' in attrs.keys()) and \
                ('Offset' in attrs.keys()):
            slope = attrs.pop('Slope')
            offset = attrs.pop('Offset')

    sds = np.ma.squeeze(sdn * slope + offset)
    sds[mask] = fill_value
    sds = np.ma.masked_where(mask, sds).astype(np.float32)
    np.ma.set_fill_value(sds, fill_value)
    return sds


SEN_DICT = {'SeaWiFS': 'S', 'MODIS-Aqua': 'A', 'MERIS': 'M', 'VIIRS-SNPP': 'V', 'YOC': 'Y', 'SGLI': 'GS'}
URL = 'https://ocean.nowpap3.go.jp/image_search/{filetype}/{subarea}/{year}/{filename}'


# Day month fetching file generator
def dmo_filegen(sensor: str, year: int, start_month: int, end_month: int, filetype: tuple,
                composite_period: str, start_day: int, end_day: int, variable: str, subarea: str):
    init = SEN_DICT[sensor]
    # Define the netCDF (PNG) file name
    end_mo = end_month + 1 if end_month < 13 else end_month
    for month in range(start_month, end_mo):
        if composite_period == 'day':
            end_dy = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
            end_dy = min(datetime.fromordinal(end_dy.toordinal() - 1).day, end_day)

            for day in range(start_day, end_dy + 1):
                files = [f'{init}{year}{month:02}{day:02}_{variable}_{subarea}_{composite_period}.{ext}'
                         for ext in filetype]

                yield from [URL.format(filetype='netcdf', subarea=subarea, year=year, filename=f)
                            if f.endswith('.nc') else
                            URL.format(filetype='images', subarea=subarea, year=year, filename=f)
                            for f in files]

        if composite_period == 'month':
            files = [f'{init}{year}{month:02}_{variable}_{subarea}_{composite_period}.{ext}'
                     for ext in filetype]

            yield from [URL.format(filetype='netcdf', subarea=subarea, year=year, filename=f)
                        if f.endswith('.nc') else
                        URL.format(filetype='images', subarea=subarea, year=year, filename=f)
                        for f in files]


# Function to download the data
def get_file(query_url: str, opath: Path, bar: tqdm):
    # path = output_dir.format(sen=sen)
    # path = output_dir
    basename = os.path.basename(query_url)

    # if not os.path.isdir(path):
    #     os.makedirs(path)

    savefile = opath.joinpath(basename)
    size = 0
    if savefile.is_file():
        size = savefile.stat()

    time.sleep(0.1)
    with requests.get(query_url) as r:
        if r.status_code != 200:
            bar.set_description(f'FileNotFound: {basename}')
            bar.update()
            return
        total = int(r.headers.get('content-length'))
        if total == size:
            bar.set_description(f'FileExists, Skip: {basename}')
            bar.update()
            return

            # print('File: {} '.format(savefile), end='')
        bar.set_description(f'Downloading: {basename}')
        bar.update()
        with open(savefile, "wb") as handle:
            for chunk in r.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                # download progress check tqdm
                if chunk:
                    handle.write(chunk)
    return


def download(variable: str, subarea: str, sensor: str, start_year: int,
             end_year: int, composite_period: str, start_month: int, file_type: tuple,
             end_month: int, start_day: int, end_day: int, output_dir: Path):

    if composite_period == 'year':
        total = len(range(start_year, end_year + 1))
    else:
        if composite_period == 'month':
            total = len([datetime(y, m, 1)
                         for y in range(start_year, end_year + 1)
                         for m in range(1, 13)
                         if (y <= end_year) and (m <= end_month)])
        else:
            start = datetime(start_year, start_month, start_day).toordinal()
            try:
                end = datetime(end_year, end_month, end_day).toordinal() + 1
            except ValueError:
                end = datetime(end_year + 1, 1, 1).toordinal() \
                    if (end_month == 12) else datetime(end_year, end_month + 1, 1).toordinal()
            total = len(range(start, end))

    with tqdm(total=total) as bar:
        for year in range(start_year, end_year + 1):
            if composite_period in ('day', 'month'):
                for query in dmo_filegen(
                        sensor=sensor
                        , year=year
                        , filetype=file_type
                        , composite_period=composite_period
                        , variable=variable
                        , start_month=start_month
                        , end_month=end_month
                        , start_day=start_day
                        , end_day=end_day
                        , subarea=subarea):
                    # --------------------------------------------------
                    get_file(query_url=query, opath=output_dir, bar=bar)
                    # --------------------------------------------------

            if composite_period == 'year':
                init = SEN_DICT[sensor]
                for ext in file_type:
                    filetype = 'netcdf' if 'nc' in ext else 'images'
                    # -----------------------------------------------------------------
                    file = f'{init}{year}_{variable}_{subarea}_{composite_period}.{ext}'
                    query = URL.format(filetype=filetype,
                                       subarea=subarea, year=year, filename=file)
                    # -----------------------------------------------------------
                    get_file(query_url=query, opath=output_dir, bar=bar)
                    # --------------------------------------------------
    return


# def download_range(composite_period: str, start_year: int, end_year: int,
#                    start_month: int, end_month: int, start_day: int, end_day: int):
#     if composite_period == 'year':
#         total = len(range(start_year, end_year + 1))
#     else:
#         if composite_period == 'month':
#             total = len([datetime(y, m, 1)
#                          for y in range(start_year, end_year + 1)
#                          for m in range(1, 13)
#                          if (y <= end_year) and (m <= end_month)])
#         else:
#             start = datetime(start_year, start_month, start_day).toordinal()
#             end = datetime(end_year, end_month, end_day).toordinal() + 1
#             total = len(range(start, end))
#     return total

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
