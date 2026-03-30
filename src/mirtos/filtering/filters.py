from typing import Any, Callable, Optional

import numpy as np
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
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


def get_without_radius_mask(tods: np.ndarray,
                            params: MaskWithoutRadius) -> np.ndarray:

    # tods: (n_kids, tod_len)
    tod_len = tods.shape[1]
    masks2d = np.zeros_like(tods, dtype=bool)

    if params.mode == MaskWithoutRadiusMode.CUTTED:

        offset = int(tod_len * params.offset)
        off = min(offset, tod_len)

        if off > 0:
            idx = np.arange(tod_len)
            # broadcast automatico su tutte le righe
            masks2d[:] = (idx < off) | (idx >= tod_len - off)

    elif params.mode == MaskWithoutRadiusMode.SIGMA:

        # Viene calcolata una maschera per kid (axis=1, cioe' per riga)
        masks2d = ~sigma_clip(tods,
                              sigma=params.sigma,
                              maxiters=params.maxiters,
                              axis=1).mask
    return masks2d


def polynomial_trend_masked(time_: Optional[np.ndarray], tods: np.ndarray, masks2d: np.ndarray, deg: int):
    """
    Fit per-detector polynomial using only samples where masks2d[k] is True,
    but return the trend evaluated on the full time_ grid (shape K×N).
    """

    K, N = tods.shape
    trend = np.zeros_like(tods)

    time_ = time_ if time_ is not None else np.arange(N, dtype=float)

    for k in range(K):
        mk = masks2d[k]
        if mk.sum() < (deg + 1):
            continue

        tk = time_[mk]

        t0 = tk.mean()
        dt = tk - t0

        # scaliamo per portare dt in [0, 1]
        std = dt.std() or 1.

        xk = dt / std
        yk = tods[k, mk]
        Vk = np.vander(xk, N=deg + 1, increasing=True)

        ck, *_ = np.linalg.lstsq(Vk, yk, rcond=None)

        V = np.vander((time_ - t0) / std, N=deg + 1, increasing=True)
        trend[k] = V @ ck

    return trend


def polynomial_trend(time_: Optional[np.ndarray], tods: np.ndarray, deg: int):
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

    n_samples = tods.shape[1 if tods.ndim == 2 else 0]
    time_ = time_ if time_ is not None else np.arange(n_samples, dtype=float)

    t0 = time_.mean()
    dt = time_ - t0

    # scaliamo per portare dt in [0, 1]
    std = dt.std() or 1.
    x = dt / std

    # accetta dati ordinati per dimensione temporale
    # p = Polynomial.fit(time_, tods, deg)
    V = np.vander(x, N=deg + 1, increasing=True)

    # Multi-RHS least squares: V @ C ≈ Y.T
    # *_ assegna il primo valore a C e scarta tutto il resto
    # equivalente a
    # res = np.linalg.lstsq(V, tods.T, rcond=None)
    # C = res[0]
    # C e' la matrice dei coefficiente polinomiali
    C, *_ = np.linalg.lstsq(V, tods.T, rcond=None)  # C: (deg + 1, N_KIDS)

    # a polynomial passo la matrice tods traposta, ma tods e' non trasposta
    # quindi traspondo p(time_)
    trend = (V @ C).T  # (kids, times)

    return trend


@register('linear_detrend')
def linear_detrend(time_: Optional[np.ndarray], tods: np.ndarray, filter_params: dict[str, Any]):
    """
    a linear_detrend gli passiamo gia' le TOD  mascherate e ritorniamo la TOD filtrata
    """

    # time_ ha dim N (samples)
    # tods ha dim num_feed x N

    time_ = time_ if time_ is not None else np.arange(tods.shape[1], dtype=float)

    masks2d: np.ndarray = filter_params.get("masks2d")
    if masks2d is None:
        return tods - polynomial_trend(time_, tods, 1)
    return tods - polynomial_trend_masked(time_, tods, masks2d, 1)


@register('remove_baseline')
def remove_baseline(time_: Optional[np.ndarray], tods: np.ndarray, filter_params: dict[str, Any]):
    """
    remove_baseline ha in input la TOD gia' mascherata e rimuove la baseline
    """

    deg = filter_params["deg"]
    masks2d = filter_params.get("masks2d")
    # TODO: gestire il caso in cui la maschera e' cutted, ossia ha tutte le righe uguali.
    #  In particolare, possiamo utilizzare polynomial_trend e non la versione masked
    if masks2d is None:
        return tods - polynomial_trend(time_, tods, deg)
    return tods - polynomial_trend_masked(time_, tods, masks2d, deg)


@register('remove_common_mode')
def remove_common_mode(time_: np.ndarray, tods: np.ndarray, filter_params: dict[str, Any]):
    """
    """

    masks2d: np.ndarray = filter_params.get("masks2d", None)

    if masks2d is not None:
        tods_cm = tods.copy()
        tods_cm[~masks2d] = np.nan
        common_mode = np.nanmean(tods_cm, axis=0, keepdims=True)
        filtered_tods = tods_cm - common_mode

        return (filtered_tods, common_mode) if filter_params["return_common_mode"] else filtered_tods

    common_mode = tods.mean(axis=0, keepdims=True)
    filtered_tods = tods - common_mode

    # FIXME: va implementata tenendo conto di eventuali nan quando mask2d viene passata
    if filter_params["use_correlation_matrix"]:
        # gli passo una matrice di TODS (num_feed x N)
        # ottengo la matrice num_feed x num_feed
        # di base, np.corrcoef ha rowvar = True, quindi considera
        # come variabili le righe e non le colonne.
        # Vogliamo cercare le correalazioni tra le TOD, quindi dobbiamo
        # passare la matrice `tods` (N_feed x N) e non la matrice `tods.T` (N x N_feed
        corr = np.corrcoef(tods)
        # azzero diagonale di corr, cioe' l'auto-correlazione tra KIDs
        np.fill_diagonal(corr, 0)

        # num_feed x N
        num = corr @ tods
        # num_feed -> num_feed x 1 per rendere la divisione consistente per la tassonomia numpy
        denom = corr.sum(axis=1)[:, np.newaxis]

        # num_feed x N
        common_mode = num / denom

        # coeff from linear regression: b = y * x / sum(x**2) con x = common_mode
        # e y = tods
        x = common_mode
        y = tods
        # voglio num_feed fattori di scala
        b = (x * y).sum(axis=1) / (x ** 2).sum(axis=1)

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


def run_filter_steps(time_: np.ndarray,
                     tods: np.ndarray,
                     steps: list[Step],
                     masks2d: np.ndarray | None = None):
    for step in steps:
        # funzione filtro
        fn = FILTERS[step.op]
        params = dict(step.params)
        if masks2d is not None and "filter" not in step.op.lower():
            # eseguo una copia di steps.params e inserisco masks2d
            # Se la inserissi direttamente in step.params, poi rimarrebbe li fino a fine esecuzione
            params['masks2d'] = masks2d

        # filtro la tod
        tods = fn(time_, tods, params)

    return tods


def clean_noise(time_, tods_: np.ndarray, masks2d: np.ndarray, n_modes: int = 1):

    # lin_func = lambda x, m, q: m*x + q
    # X = tods_.copy()
    # for t, tod in enumerate(X):
    #
    #     tm_ = np.arange(len(tod))
    #     xdata = tm_[masks2d[t]]
    #     ydata = tod[masks2d[t]]
    #
    #     popt, pcov = curve_fit(f=lin_func, xdata=xdata, ydata=ydata)
    #     X[t] -= lin_func(tm_, *popt)
    #
    # return X

    # X = linear_detrend(time_, tods_, {"masks2d": masks2d})
    X = linear_detrend(None, tods_, {"masks2d": masks2d})

    # centro le tod per riga
    X -= np.nanmean(X, axis=1, keepdims=True)

    # sostituisco nan con 0
    X[np.isnan(X)] = 0
    U, s, V = np.linalg.svd(X, full_matrices=False)

    for m in range(n_modes):
        mode_m = s[m] * np.einsum("i, j -> ij", U[:, m], V[m, :])  # (n_kids, n_samples)
        X -= mode_m

    return X


if __name__ == "__main__":
    ...
