# %%
import matplotlib.pyplot as plt
import numpy as np
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import xarray as xr
import numpy as np
from scipy.ndimage import gaussian_filter1d

def rabi_chevron(ds, xnew, ynew,
                iSWAP_amp_idx, exp_type):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    xdata = ds.state_target.isel(amp=iSWAP_amp_idx).coords["time"].values
    ydata = ds.state_target.isel(amp=iSWAP_amp_idx)
    ysmooth = gaussian_filter1d(ydata, sigma=1.5)
    ax[0].plot(xdata, ydata)
    ax[0].set_title(f'raw data idx={iSWAP_amp_idx}')
    ax[1].plot(xdata, ysmooth)
    ax[1].set_title(f'Gaussian idx={iSWAP_amp_idx}')
    ax[2].plot(xnew, ynew)
    ax[2].set_title(f'cos conv idx={iSWAP_amp_idx}')
    fig.suptitle(f'iSWAP {exp_type} state', fontsize=16, y=1.02)

    def draw_rabi_chevron(ax, xdata, ydata):
        st = xr.DataArray(
            ydata,
            dims=['time'],
            coords={'time': xdata}
        )
        fit_res = fit_oscillation_decay_exp(st, dim="time")

        # 從結果中取出參數
        a     = fit_res.sel(fit_vals="a").item()
        f     = fit_res.sel(fit_vals="f").item()
        phi   = fit_res.sel(fit_vals="phi").item()
        offset= fit_res.sel(fit_vals="offset").item()
        decay = fit_res.sel(fit_vals="decay").item() # 1/t2 (1/ns)

        xfit = np.linspace(xdata.min(), xdata.max(), 500)
        yfit = oscillation_decay_exp(xfit, a, f, phi, offset, decay)
        ax.plot(xfit, yfit, label="Fit", color="red", linestyle='--', linewidth=1)

        text = f'iSWAP {exp_type} = {1/decay:.2f} ns'
        ax.text(0.95, 0.95, text, 
            transform=ax.transAxes,
            fontsize=12, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    draw_rabi_chevron(ax[0], xdata, ydata)
    draw_rabi_chevron(ax[1], xdata, ysmooth)
    draw_rabi_chevron(ax[2], xnew, ynew)

    plt.tight_layout()
    plt.show()

    return fig

def exponetial(ds, xnew, ynew,
               iSWAP_amp_idx, exp_type, peak=True):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    xdata = ds.state_target.isel(amp=iSWAP_amp_idx).coords["time"].values
    ydata = ds.state_target.isel(amp=iSWAP_amp_idx)
    ysmooth = savgol_filter(ydata, window_length=11, polyorder=2)
    ysmooth = gaussian_filter1d(ydata, sigma=1.5)

    ax[0].plot(xdata, ydata)
    ax[0].set_title(f'raw data idx={iSWAP_amp_idx}')
    ax[1].plot(xdata, ysmooth)
    ax[1].set_title(f'Guassian idx={iSWAP_amp_idx}')
    ax[2].plot(xnew, ynew)
    ax[2].set_title(f'cos conv idx={iSWAP_amp_idx}')
    fig.suptitle(f'iSWAP {exp_type} state', fontsize=16, y=1.02)

    def draw_exponential(ax, xdata, ydata, peak=True):
        if peak:
            peaks, _ = find_peaks(ydata, distance=10)
        else:
            peaks, _ = find_peaks(-ydata, distance=10)
        xpeaks = xdata[peaks]
        ypeaks = ydata[peaks]
        ax.scatter(xpeaks, ypeaks)

        def exp_decay(x, A, tau, C):
            return A * np.exp(-x / tau) + C 

        # 初始猜測值：A = 峰值最大值, tau = x 軸範圍的一半, C = 尾端的平均值
        p0 = [max(ypeaks), (xpeaks[-1]-xpeaks[2])/2, ypeaks[-1]]

        # 進行擬合
        popt, pcov = curve_fit(exp_decay, xpeaks, ypeaks, p0=p0)

        # popt 裡面就是 [A, tau, C]
        A_fit, tau_fit, C_fit = popt

        xfit = np.linspace(xpeaks[0], xpeaks[-1], 100)
        ax.plot(xfit, exp_decay(xfit, *popt), 'k--', lw=2, label='Fit (Decay)')
        t2_text = f'iSWAP {exp_type} = {tau_fit:.2f} ns'

        ax.text(0.95, 0.95, t2_text, 
            transform=ax.transAxes,
            fontsize=12, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
    draw_exponential(ax[0], xdata, ydata, peak)
    draw_exponential(ax[1], xdata, ysmooth, peak)
    draw_exponential(ax[2], xnew, ynew, peak)

    plt.tight_layout()
    plt.show()

    return fig

def cos_conv_filter(ds, iSWAP_amp_idx, window_length):
    xdata = ds.state_target.isel(amp=iSWAP_amp_idx).coords["time"].values
    ydata = ds.state_target.isel(amp=iSWAP_amp_idx)

    fit_res = fit_oscillation_decay_exp(ydata, dim="time")
    f     = fit_res.sel(fit_vals='f').item()
    phi   = fit_res.sel(fit_vals='phi').item()

    def cos(t, f, phi):
        return np.cos(2 * np.pi * f * t + phi) ** 2
    
    cos_weight = cos(xdata, f, phi)

    
    xnew = xdata[window_length // 2:-window_length // 2 + 1]
    ynew = np.zeros(len(xdata) - window_length + 1)

    ydata -= ydata.mean()

    for i in range(len(xnew)):
        ynew[i] = sum(ydata[i + j] * cos_weight[i + j] for j in range(window_length)) / sum(cos_weight[i: i + window_length])
    
    return xnew, ynew
