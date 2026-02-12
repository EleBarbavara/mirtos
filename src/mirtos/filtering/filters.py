from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum

from astropy.stats import sigma_clip

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from scipy.signal import butter, sosfiltfilt

from mirtos.core.type_defs.filters import Step, MaskWithoutRadius, MaskWithoutRadiusMode

# callable indica che filter_fn e' di tipo funzione
# np.ndarray: tod da filtrare
# dict[str, Any]: parametri presi dallo yaml
# dict[str, Any]: parametri che gli passo a run_time (ovvero gli passo il "contesto")
filter_fn = Callable[[np.ndarray, np.ndarray, dict[str, Any]], np.ndarray]
# associa ad ogni funzione il suo nome letto dallo yaml
FILTERS: dict[str, filter_fn] = {}


# il decoratore modifica il comportamento di una funzione senza modificarne il contenuto
# il decoratore prende in input la chiave che vogliamo salvare come step di filtraggio
def register(name: str):
    # funzione interna che opera su name e associa il nome alla funzione fn nel dizionario FILTERS
    def register_fn(fn: filter_fn):
        if name not in FILTERS:
            # alla chiave FILTERS[name] assegniamo la funzione fn
            FILTERS[name] = fn
        return fn

    return register_fn


def get_without_radius_mask(tods: np.ndarray, subscan_mask: np.ndarray, params: MaskWithoutRadius):
    # per gestire il caso “None” (deterend non applicato) e un caso non programmato di default (no cutted o sigma)
    # allo stesso modo nel costrutto del match case, si usa questa sintassi

    # tods.shape[1] considera la lunghezza di una tod e non il numero di kid (.shape[0])
    tod_len = tods.shape[1]
    tod_mask = np.zeros(tod_len, dtype=bool)

    valid_tods = tods[:, subscan_mask]
    valid_mask = np.zeros(valid_tods.shape[1], dtype=bool)

    if params.mode == MaskWithoutRadiusMode.CUTTED:

        offset = int(tod_len * params.offset)
        # tod_mask[time_mask][offset:-offset] = True
        valid_mask[offset:-offset] = True

    elif params.mode == MaskWithoutRadiusMode.SIGMA:

        # false: valori al di sopra della sigma, true: valori al di sotto della sigma
        # Qui stiamo passando una matrice (KIDs, len(subscan_mask))
        # sigma_clip ritorna una mask booleana della stessa shape
        # Ora, applichiamo la stessa logica utilizzata per creare la maschera di subscan:
        # scartiamo una colonna se una delle righe corrispondenti ha un valore settato a True (out;ier)
        clipped = sigma_clip(tods[:, subscan_mask], sigma=params.sigma, maxiters=params.maxiters).mask
        valid_mask = ~np.any(clipped, axis=0)

    tod_mask[subscan_mask] = valid_mask

    return tod_mask


def polynomial_trend(time_: np.ndarray, tods: np.ndarray, deg: int):
    """
    Fits a polynomial trend to time-ordered data and returns the computed trend.

    This function takes time-ordered data points, constructs a polynomial model of
    a specified degree, and computes the trend for given observational data.

    :param time_: 1D array of time values.
    :param tods: 2D array representing observational data, where each row corresponds
                 to a different set of observations over the time data.
    :param deg: Degree of the polynomial to fit.
    :return: 2D array containing the computed polynomial trend for each observation
             set in `tods`. The dimensions match the input array `tods`.
    """

    # accetta dati ordinati per dimensione temporale
    # p = Polynomial.fit(time_, tods, deg)
    V = np.vander(time_, N=deg + 1, increasing=True)

    # Multi-RHS least squares: V @ C ≈ Y.T
    # *_ assegna il primo valore a C e scarta tutto il resto
    # equivalente a
    # res = np.linalg.lstsq(V, tods.T, rcond=None)
    # C = res[0]
    # C e' la matrice dei coefficiente polinomiali
    C, *_ = np.linalg.lstsq(V, tods.T, rcond=None)  # C: (deg + 1, N_KIDS)

    # a polynomial passo la mtrice tods traposta, ma tods e' non trasposta
    # quindi traspondo p(time_)
    trend = (V @ C).T  # (kids, times)

    return trend


@register('linear_detrend')
def linear_detrend(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):
    """
    a linear_detrend gli passiamo gia' le TOD  mascherate(e' il caso in cui non
    viene dato il raggio dal a1995_conf.yaml) e ritorniamo la TOD mascherata e la maschera
    """

    # time_ ha dim N (samples)
    # tods ha dim num_feed x N

    # a polynomial passo la mtrice tods traposta, ma tods e' non trasposta
    # quindi traspondo p(time_)
    return tods - polynomial_trend(time_, tods, 1)


@register('remove_baseline')
def remove_baseline(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):
    """
    remove_baseline ha in input la TOD gia' mascherata (in quanto e' il caso in cui
    il raggio viene definito nel a1995_conf.yaml) e rimuove la baseline
    """

    return tods - polynomial_trend(time_, tods, filter_params["baseline_poly_deg"])


@register('remove_common_mode')
def remove_common_mode(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):
    """
    passiamo gia' TOD mascherata con linear_detrend
    """

    common_mode = tods.mean(axis=1, keepdims=True)
    filtered_tods = tods - common_mode

    if filter_params["use_correlation_matrix"]:
        # gli passo una matrice di TODS (num_feed x N)
        # ottengo la matrice num_feed x num_feed
        corr = np.corrcoef(tods.T)

        # num_feed x N
        num = corr @ tods
        # num_feed -> num_feed x 1 per rendere la divisione consistente per la tassonomia numpy
        denom = corr.sum(axis=1)[:, np.newaxis]

        # num_feed x N
        common_mode = num / denom

        # coeff from linear regression: b = y * x / sum(x**2) con x = common_mode
        # e y = tods (sulla base di quanto fate con OLS)
        x = common_mode
        y = tods
        # voglio num_feed fattori di scala
        b = (x * y).sum(axis=1) / (tods ** 2).sum(axis=1)

        filtered_tods = tods - b[:, np.newaxis] * common_mode

    return (filtered_tods, common_mode) if filter_params["return_common_mode"] else filtered_tods


def apply_butterworth(time_: np.ndarray, filter_params: dict[str, Any]):
    sampling_frequency = 1 / (time_[1] - time_[0])
    nyquist_frequency = 0.5 * sampling_frequency
    norm_cutoff_freq = filter_params["cutoff_freq"] / nyquist_frequency

    wn = [norm_cutoff_freq]

    if filter_params["btype"] == 'bandpass':
        norm_cuton_freq = filter_params["cuton_freq"] / nyquist_frequency
        # inseriramo nella posizione 0
        wn.insert(0, norm_cuton_freq)

    return butter(filter_params["butter_order"],
                  wn,
                  btype=filter_params["btype"],
                  fs=sampling_frequency,
                  output='sos')  # second-order sections


@register('band_pass_filter')
def band_pass_filter(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):
    """
    tods in generale e' num_feed x N
    """

    sos = apply_butterworth(time_, filter_params)

    # sosfiltfilt e' un filtro piu' stabile di filtfilt: tagli sulle frequenze fatti meglio, genera meno artifici
    return sosfiltfilt(sos, tods, axis=1)


@register('low_pass_filter')
def low_pass_filter(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):
    sos = apply_butterworth(time_, filter_params)

    return sosfiltfilt(sos, tods, axis=1)


def run_filter_steps(time_: np.ndarray, tods: np.ndarray, steps: list[Step]):
    for step in steps:
        # funzione filtro
        fn = FILTERS[step.op]
        # filtro la tod
        tods = fn(time_, tods, step.params)

    return tods
