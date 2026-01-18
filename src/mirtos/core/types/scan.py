import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from mirtos.core.types.beam_map import BeamMap
from pprint import pprint

# classe di enumerazione dove i campi sono interi
from enum import IntEnum

class SubscanType(IntEnum):

    RA = 1
    DEC = 2
    TEMP = 3


@dataclass
class PartialSubscan:
    id_subscan: str
    type: SubscanType
    data: np.ndarray


@dataclass
class Subscan:
    id_subscan: str
    ra: np.ndarray
    dec: np.ndarray
    temp: np.ndarray

    @classmethod
    def from_partials(cls, partials: list[PartialSubscan]):

        if len(partials) % 3:
            raise ValueError("Number of partials must be a multiple of 3: ra, dec, temp")

        if not any(partials):
            raise ValueError("Some partials are None. Check that the process_subscan_file function is working correctly.")

        # ordiniamo per id + suscan_type crescente i partial subscan: ex 078
        partials = sorted(partials, key=lambda p: int(p.id_subscan) + p.type.value)

        n = len(partials)
        subscans = []
        # itero su partials scalando ogni volta di 3
        for p1, p2, p3 in zip(*(iter(partials),) * 3):

            subscan = cls(p1.id_subscan,
                          ra = p1.data,
                          dec = p2.data,
                          temp = p3.data)

            subscans.append(subscan)

        return subscans


@dataclass
class Scan:
    id_scan: str
    # se non passiamo Subscan, fiels crea una lista vuota
    subscans: list[Subscan] = field(default_factory=list)


# singolo compito eseguito dal multiprocessing: aprire il file di un subscan e processarlo
@dataclass
class Job:

    id: str # job id
    filename: Path
    beam_map: BeamMap = None

# contiene esito esecuzione job, l'errore ritornato e in caso di esito positivo i dati processati
@dataclass
class Result:

    job: Job
    partial_subscan: PartialSubscan
    error: str = None

# funzione che processa un subscan e quindi esegue un singolo job
# deve estrarre dall'hdul tutte le informazioni necessarie in quanto portarsi dietro tutto l'hdul e' pesante.
# se ci sono informazioni che devono essere salvate si crea l'attributo relativo e ci si salvano i dati dentro
def process_subscan_file(job: Job) -> Result:

    try:
        # il job ha id 001_108
        subscan_type, id_subscan = job.id.split('_')

        #lettura subscan ed elaborazione (richiamare quello che ora sta in fits.py)
        # SubScanType di un intero ritorna l'attributo associato all'intero
        partial_subscan = PartialSubscan(id_subscan=id_subscan,
                                         type=SubscanType(int(subscan_type)),
                                         data=np.ones(1))

        return Result(job, partial_subscan=partial_subscan, error='')

    except Exception as e:
        return Result(job, partial_subscan=None, error=traceback.format_exc(3)) # ultime tre path dei file che hanno generato l'errore'

# gli passo una lista di Job, ovvero un Job per ogni file subscan da processare
def process_all_subscans(jobs: list[Job]):


    with ProcessPoolExecutor() as executor:

        # schelue esecuzione funzione in parallelo
        # lista di oggetti Result
        futures = {executor.submit(process_subscan_file, job): job for job in jobs}

        # ritorna i task completati, ovvero i partial subscans processati
        partials = [r.result().partial_subscan for r in as_completed(futures)]

        return Subscan.from_partials(partials)


if __name__ == "__main__":

    scan_dir = Path('/Volumes/Data/PycharmProjects/mirtos/data/input/examples')
    #beam_map_path = Path('')

    #beam_map = BeamMap.from_dat(beam_map_path)

    job_ids = []
    # scorrere sugli id che per noi sono le due triplette finali
    for file in scan_dir.iterdir():

        # 'dirfile_20240611_124429__MERGED_WITH__20240611-124423-MISTRAL-JUPITER_FOCUS_SCAN_001_108' il suo id e' 001_108
        job_ids.append('_'.join(file.stem.split("_")[-2:]))

    jobs = [Job(job_id, file, None) for job_id, file in zip(job_ids, scan_dir.iterdir())]


    subscans = process_all_subscans(jobs)

    pprint(subscans)
    # for subscan in subscans:
    #
    #     if subscan.id_subscan == '108':
    #         print(subscan.ra)

