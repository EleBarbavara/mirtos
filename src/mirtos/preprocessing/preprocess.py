import numpy as np
from pprint import pprint
from dataclasses import dataclass, field

from mirtos.core.multipreprocess import Job, process_all


# maggiordomo che da' a process_subscan_file le info che gli servono per processare il subscan
@dataclass
class PreprocessPayload:

    tod: np.ndarray
    gain: int = 1
    multi: int = 4


def process_tod(job: PreprocessPayload) -> np.ndarray:

    return (job.tod + job.gain) * job.multi


if __name__ == "__main__":


    subscans = ... # ottenuta altrove
    gain = 1
    multi = 4
    jobs = []

    for s_i, sub in enumerate(subscans):
        for k_i, kid in enumerate(sub.kids):
            jobs.append(
                Job(f'{s_i}_{k_i}',
                PreprocessPayload(kid.tod, gain, multi)))

    res = process_all(jobs, process_tod)

    for r in res:
        if not r.error:
            subscan, kid = list(map(int, r.job.id.split('_')))
            subscans[subscan].kids[kid].tod = r.output
