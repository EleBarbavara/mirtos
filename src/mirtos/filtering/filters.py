from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum

from astropy.stats import sigma_clip

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from scipy.signal import butter, sosfiltfilt

from mirtos.core.types.config import Step, LinearDetrendMode

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


def remove_polynomial_fit(time_: np.ndarray, tods: np.ndarray, deg: int):
    """
    Removes a polynomial fit from time-ordered data.

    This function calculates the best-fit polynomial of a specified degree for a given
    time-ordered dataset and time indices. It then subtracts the polynomial fit from
    the original dataset, effectively removing the fitted trend.

    Args:
        time_ (np.ndarray): A 1D array representing time indices. The data must be
            sorted in ascending order by time.
        tods (np.ndarray): A 2D array of time-ordered data. Each row represents
            observations at a specific time.
        deg (int): The degree of the polynomial to fit to the data.

    Returns:
        np.ndarray: A 2D array representing the time-ordered data with the polynomial
        trend removed, matching the shape of the input `tods`.
    """

    # accetta dati ordinati per dimensione temporale
    p = Polynomial.fit(time_, tods, deg)

    # a polynomial passo la mtrice tods traposta, ma tods e' non trasposta
    # quindi traspondo p(time_)
    return tods - p(time_).T

@register('linear_detrend')
def linear_detrend(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):

    """
    a linear_detrend facciamo calcolare la maschera da applicare alla TOD (e' il caso in cui non
    viene dato il raggio dal config.yaml) e ritorniamo la TOD mascherata e la maschera
    """

    # time_ ha dim N (samples)
    # tods ha dim num_feed x N


    if filter_params["mode"].value == 'cutted':

        offset = int(len(tods) * filter_params["offset"])
        mask = np.zeros(len(tods), dtype=bool)
        mask[offset:-offset] = True

    # per gestire il caso “None” e il caso generico allo stesso modo nel costrutto del match case, si usa questa sintassi:
    elif filter_params["mode"].value == 'sigma':

        # false: valori al di sopra della sigma, true: valori al di sotto della sigma
        mask = ~sigma_clip(tods, sigma=filter_params["sigma"], maxiters=filter_params["maxiters"]).mask


    # per gestire il caso “None” (deterend non applicato) e un caso non programmato di default (no cutted o sigma)
    # allo stesso modo nel costrutto del match case, si usa questa sintassi
    else:

        return tods, np.zeros(len(tods), dtype=bool)

    time_fit = time_[mask]
    tods_fit = tods[:, mask].T


    # a polynomial passo la mtrice tods traposta, ma tods e' non trasposta
    # quindi traspondo p(time_)
    return remove_polynomial_fit(time_fit, tods_fit, 1), mask


@register('remove_baseline')
def remove_baseline(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):

    """
    remove_baseline ha in input la TOD gia' mascherata (in quanto e' il caso in cui
    il raggio viene definito nel config.yaml) e rimuove la baseline
    """

    return tods - remove_polynomial_fit(time_, tods, filter_params["baseline_poly_deg"])


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
        b = (x * y).sum(axis=1) / (tods**2).sum(axis=1)

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
                  output='sos') # second-order sections


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