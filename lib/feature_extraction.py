import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.signal import welch
import pywt


def compute_time_domain_features(x):

    """
    Compute all the time domain features
    :param x: Input time series
    :return: A list of time domain features
    """

    methods_list = [
        compute_rms,
        compute_variance,
        compute_peak_value,
        compute_crest_factor,
        compute_kurtosis_fisher,
        compute_clearance_factor,
        compute_impulse_factor,
        compute_shape_factor,
        compute_line_integral,
        compute_peak_to_peak,
        compute_skewness
    ]

    # Apply features one by one
    time_domain_features = []
    for method in methods_list:
        time_domain_features.append(method(x))

    return time_domain_features


def compute_frequency_domain_features(x, args):

    """
    Compute all the frequency domain features
    :param x: A time series dataset
    :param args: A list of dictionary of arguments for each of the methods
    :return: A list of frequency domain features
    """

    assert isinstance(args, list), "args should be a list"
    assert isinstance(args[0], dict), "args should be a list of dictionary"

    methods_list = [
        compute_peak_fft,
        compute_energy_fft,
        compute_PSD_FFT
    ]

    # Apply the features one by one
    frequency_domain_features = []
    for index, method in enumerate(methods_list):
        frequency_domain_features.append(method(x, **args[index]))

    return frequency_domain_features


def compute_time_frequency_features(x, args):

    """
    Compute time frequency features
    :param x: A time series dataset
    :param args: A list of dictionary of arguments for each of the methods
    :return: A list of time frequency domain features
    """

    assert isinstance(args, list), "args should be a list"
    assert isinstance(args[0], dict), "args should be a list of dictionary"

    methods_list = [
        compute_energy_WPD_1,
        compute_energy_WPD_2,
        compute_energy_WPD_3
    ]

    # Apply the features one by one
    time_frequency_features = []
    for index, method in enumerate(methods_list):
        time_frequency_features.append(method(x, **args[index]))

    return time_frequency_features


def compute_all_features(x, freq_args, freq_time_args):

    """

    :param x: A time series segmented data
    :param freq_args: A list of dictionary of arguments for each of the frequency features methods
    :param freq_time_args: A list of dictionary of arguments for each of the time-frequency methods
    :return: A list combined in the order of time, frequency, and time-frequency
    """

    time_features = compute_time_domain_features(x)
    freq_features = compute_frequency_domain_features(x, freq_args)
    freq_time_features = compute_time_frequency_features(x, freq_time_args)

    return time_features + freq_features + freq_time_features


########################################################################################################################
# Time domain features
########################################################################################################################
def compute_rms(x):

    """
    Computing the RMS
    :param x: array of segmented time series inputs
    :return: rms value of the time-series
    """

    # Computing RMS
    return np.sqrt(np.mean(x**2))


def compute_variance(x):

    """
    Data Variance
    :param x: A time series data
    :return: Variance of the segmented data
    """
    return np.var(x)


def compute_peak_value(x):

    """
    Peak value of data
    :param x: Time series data
    :return: Peak value of the time series data
    """
    return np.max(np.abs(x))


def compute_crest_factor(x):

    """
    Crest factor
    :param x: Time series data
    :return: Crest factor of the time series data
    """

    pvt_val = compute_peak_value(x)
    rms_val = compute_rms(x)

    return pvt_val/rms_val


def compute_kurtosis_fisher(x):

    """
    Fisher kurtosis
    :param x: A time series data
    :return: fisher kurtosis of the input
    """

    return kurtosis(x, fisher=True)


def compute_clearance_factor(x):

    """
    Clearance factor
    :param x: A time series input
    :return: Clearance factor computation
    """

    pvt_val = compute_peak_value(x)
    denom = np.mean(np.sqrt(np.abs(x))) ** 2

    return pvt_val/denom


def compute_impulse_factor(x):

    """

    :param x:
    :return:
    """

    peak_val = compute_peak_value(x)
    denom = np.mean(np.abs(x))

    return peak_val/denom


def compute_shape_factor(x):

    """

    :param x:
    :return:
    """

    rms_val = compute_rms(x)
    denom = np.mean(np.abs(x))

    return rms_val/denom


def compute_line_integral(x):

    """

    :param x:
    :return:
    """

    int_sum = 0
    for index in range(1, x.shape[0]):
        int_sum += np.abs(x[index] - x[index-1])

    return int_sum


def compute_peak_to_peak(x):

    """

    :param x:
    :return:
    """

    return np.max(x) - np.min(x)


def compute_skewness(x):

    """

    :param x:
    :return:
    """

    return skew(x)


########################################################################################################################
# Frequency domain features
########################################################################################################################
def compute_peak_fft(x, **kwargs):

    """

    :param x:
    :return:
    """

    return np.abs(np.max(fft(x, **kwargs)))


def compute_energy_fft(x, **kwargs):

    """

    :param x:
    :return:
    """

    return np.abs(np.sum(fft(x, **kwargs)))


def compute_PSD_FFT(x, **kwargs):

    """

    :param x:
    :param sampling_frequency:
    :return:
    """

    # Welch's method
    f, Pxx = welch(x, **kwargs)

    return np.sum(Pxx) * (f[1] - f[0])


########################################################################################################################
# Time-Frequency domain features
########################################################################################################################
def compute_energy_WPD_1(x, **kwargs):

    """

    :param x:
    :param kwargs:
    :return:
    """

    wp = pywt.WaveletPacket(data=np.squeeze(x), **kwargs)
    return np.sqrt(np.sum(np.array(wp['a'].data) ** 2))


def compute_energy_WPD_2(x, **kwargs):

    """

    :param x:
    :param kwargs:
    :return:
    """

    wp = pywt.WaveletPacket(data=np.squeeze(x), **kwargs)
    return np.sqrt(np.sum(np.array(wp['aa'].data) ** 2))


def compute_energy_WPD_3(x, **kwargs):

    """

    :param x:
    :param kwargs:
    :return:
    """

    wp = pywt.WaveletPacket(data=np.squeeze(x), **kwargs)
    return np.sqrt(np.sum(np.array(wp['aaa'].data) ** 2))

