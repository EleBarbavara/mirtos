import traceback
import numpy as np
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from mirtos.io.fits import load_subscan_fits
from mirtos.core.types.beam_map import BeamMap
from mirtos.core.config_types import load_config


# maggiordomo che da' a process_subscan_file le info che gli servono per processare il subscan
@dataclass
class Job:

    id: str # job id
    tod: np.ndarray
    gain: int = 1
    multi: int = 4


@dataclass
class Result:

    job: Job
    tod: np.ndarray = None
    error: str = None

def process_tod(job: Job) -> Result:

    try:

        tod = (job.tod + job.gain) * job.multi

        return Result(job, tod=tod, error='')

    except Exception as e:
        return Result(job, tod=None, error=traceback.format_exc(3))


def process_all_tods(jobs: list[Job]):

    with ProcessPoolExecutor() as executor:

        # schedula esecuzione funzione in parallelo
        # lista di oggetti Result
        futures = {executor.submit(process_subscan_file, job): job for job in jobs}

        # ritorna i task completati, ovvero i partial subscans processati
        return [r.result() for r in as_completed(futures)]


if __name__ == "__main__":


    subscans = ... # ottenuta altrove
    gain = 1
    multi = 4
    jobs = []

    for s_i, sub in enumerate(subscans):
        for k_i, kid in enumerate(sub.kids):
            jobs.append(Job(id=f'{s_i}_{k_i}', tod=kid.tod, gain=gain, multi=multi))

    results = process_all_tods(jobs)

    for res in results:
        if not res.error:
            subscan, kid = list(map(int, res.job.id.split('_')))
            subscans[subscan].kids[kid].tod = res.tod
