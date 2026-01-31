from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum

from astropy.stats import sigma_clip

import numpy as np
from numpy.polynomial.polynomial import Polynomial

# callable indica che filter_fn e' di tipo funzione
# np.ndarray: tod da filtrare
# dict[str, Any]: parametri presi dallo yaml
# dict[str, Any]: parametri che gli passo a run_time (ovvero gli passo il "contesto")
filter_fn = Callable[[np.ndarray, dict[str, Any], dict[str, Any]], np.ndarray]
# associa ad ogni funzione il suo nome letto dallo yaml
FILTERS: dict[str, filter_fn]


class LinearDetrendMode(Enum):
    CUTTED = 'cutted'
    SIGMA = 'sigma'
    NONE = 'none'

# TODO: da mettere nel file config.py
@dataclass(frozen=True)
class FilterParams:

    mode: LinearDetrendMode
    radius: float = 0
    offset: float = 0.1

    sigma: float = 6
    maxiters: int = 10

    baseline_poly_deg: int = 4

    use_correlation_matrix: bool = False

def remove_polynomial_fit(time_: np.ndarray, tods: np.ndarray, deg: int):

    # accetta dati ordinati per dimensione temporale
    p = Polynomial.fit(time_, tods, deg)

    # a polynomial passo la mtrice tods traposta, ma tods e' non trasposta
    # quindi traspondo p(time_)
    return tods - p(time_).T

def linear_detrend(time_: np.ndarray, tods: np.ndarray, filter_params: FilterParams):

    """
    a linear_detrend facciamo calcolare la maschera da applicare alla TOD e ritorniamo
    la TOD mascherata e la maschera
    """

    # time_ ha dim N (samples)
    # tods ha dim num_feed x N


    if filter_params.mode.value == 'cutted':

        offset = int(len(tods) * filter_params.offset)
        mask = np.zeros(len(tods), dtype=bool)
        mask[offset:-offset] = True

    # per gestire il caso “None” e il caso generico allo stesso modo nel costrutto del match case, si usa questa sintassi:
    elif filter_params.mode.value == 'sigma':

        # false: valori al di sopra della sigma, true: valori al di sotto della sigma
        mask = ~sigma_clip(tods, sigma=filter_params.sigma, maxiters=filter_params.maxiters).mask


    # per gestire il caso “None” (deterend non applicato) e un caso non programmato di default (no cutted o sigma)
    # allo stesso modo nel costrutto del match case, si usa questa sintassi
    else:

        return tods, np.zeros(len(tods), dtype=bool)

    time_fit = time_[mask]
    tods_fit = tods[:, mask].T


    # a polynomial passo la mtrice tods traposta, ma tods e' non trasposta
    # quindi traspondo p(time_)
    return remove_polynomial_fit(time_fit, tods_fit, 1), mask


def remove_baseline(time_: np.ndarray, tods: np.ndarray, filter_params: FilterParams):

    """
    remove_baseline ha in input la TOD gia' mascherata e rimuove la baseline
    """

    return tods - remove_polynomial_fit(time_, tods, filter_params.baseline_poly_deg)

def remove_common_mode(time_: np.ndarray, tods: np.ndarray, filter_params: FilterParams):

    """
    passiamo gia' TOD mascherata con linear_detrend
    """

    if filter_params.use_correlation_matrix:

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

        return tods - b[:, np.newaxis] * common_mode