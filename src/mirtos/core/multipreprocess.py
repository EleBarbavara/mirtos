"""
Infrastruttura generica per preprocessing in parallelo (ProcessPool)
con pattern Job/Result + worker generico

- Job contiene solo id + payload (parametri specifici)
- Result standardizza successo/errore
- run_job esegue un singolo job (try/except + traceback)
- process_all esegue una lista di job in parallelo e ritorna i Result
"""

import traceback
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Generic, Optional, TypeVar, Iterable, Sequence


PayloadType = TypeVar("PayloadType")   # payload (parametri specifici del dominio)
OutputType = TypeVar("OutputType")     # output prodotto dal job


@dataclass(frozen=True)
class Job(Generic[PayloadType]):
    """Singola unità di lavoro: id + payload specifico"""
    id: str
    payload: PayloadType


@dataclass
class Result(Generic[PayloadType, OutputType]):
    """Esito dell'esecuzione di un Job"""
    job: Job[PayloadType]
    output: Optional[OutputType] = None
    error: Optional[str] = None  # None => ok, stringa => errore


# Single-job execution
def run_job(
    job: Job[PayloadType],
    fn: Callable[[PayloadType], OutputType],
    *, # l'asterisco vuol dire che i parametri che seguono devono essere passati obbligatoriamente come keyword = argument
    tb_limit: int = 3) -> Result[PayloadType, OutputType]:
    """
    Esegue un singolo job: chiama fn(payload) e cattura eccezioni.
    tb_limit controlla quante righe di traceback includere
    """
    try:
        out = fn(job.payload)
        return Result(job=job, output=out, error=None)
    except Exception:
        return Result(
            job=job,
            output=None,
            error=traceback.format_exc(limit=tb_limit))


# Parallel execution
def process_all(
    jobs: Sequence[Job[PayloadType]] | Iterable[Job[PayloadType]],
    fn: Callable[[PayloadType], OutputType],
    *,
    max_workers: Optional[int] = None,
    return_in_completion_order: bool = True,
    tb_limit: int = 3) -> list[Result[PayloadType, OutputType]]:
    """
    Processa tutti i job in parallelo usando ProcessPoolExecutor

    - jobs: lista/iterabile di Job
    - fn: funzione top-level picklable che processa un payload e ritorna output
    - max_workers: numero processi (None => default)
    - return_in_completion_order:
        True  -> risultati nell'ordine di completamento (più veloce)
        False -> risultati nell'ordine di expected (piu' stabile)
    - tb_limit: profondità traceback negli errori
    """
    jobs_list = list(jobs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, job, fn, tb_limit=tb_limit) for job in jobs_list]

        if return_in_completion_order:
            return [f.result() for f in as_completed(futures)]

        # ordine expected: mappa future -> index
        future_to_idx = {f: i for i, f in enumerate(futures)}
        results: list[Optional[Result[PayloadType, OutputType]]] = [None] * len(futures)

        for f in as_completed(futures):
            idx = future_to_idx[f]
            results[idx] = f.result()

        return [r for r in results if r is not None]


def outputs_valid(results: Iterable[Result[PayloadType, OutputType]]) -> list[OutputType]:
    """Estrae solo gli output dei job andati a buon fine"""
    return [r.output for r in results if not r.error and r.output]


def errors_only(results: Iterable[Result[PayloadType, OutputType]]) -> list[Result[PayloadType, OutputType]]:
    """Filtra solo i result con errore"""
    return [r for r in results if r.error is not None]


def raise_if_any_error(results: Iterable[Result[PayloadType, OutputType]]):
    """
    Se esiste almeno un errore, solleva RuntimeError con un riassunto
    """
    if not (errs := [r for r in results if r.error]):
        return

    msg_lines = [f"{len(errs)} job falliti. Errori:"]

    for r in errs[:5]:
        msg_lines.append(f"- Job {r.job.id}:")
        msg_lines.append(r.error or "")

    raise RuntimeError("\n".join(msg_lines))

