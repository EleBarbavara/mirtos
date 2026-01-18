from dataclasses import dataclass
from mirtos.io.dat import read_dat
from pathlib import Path
import pandas as pd

# ritorna le informazioni che ci servono dal beam map
@dataclass
class BeamMap:

    beam_map: pd.DataFrame
    
    @classmethod
    def from_dat(cls, filename: Path,
                 comments: str = '#',
                 dataframe_columns=('id', 'lon-offset', 'lat-offset', 'Tcal pol1', 'Tcal pol2', 'flag')):

        data = read_dat(filename, comments)

        return cls(pd.DataFrame(data, columns=dataframe_columns))


if __name__ == "__main__":

    beam_map_file= Path('/Volumes/Data/PycharmProjects/mirtos/metadata/chp_offset_rel8_14DEC24_matteo.dat')
    bm = BeamMap.from_dat(beam_map_file)