import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from mirtos.core.projections import rot
from mirtos.io.dat import read_dat
from mirtos.core.types.config import MapMakingFrame


@dataclass
class BeamMap:
    # una dataclass (o anche in generale una classe) si definisce/istanzia, passandogli gli argomenti
    # che la costituiscono

    beam_map: pd.DataFrame

    # il metodo di classe puo' agire in due modi:
    # 1) mi permette di costruire l'oggetto in modo alternativo
    # 2) mi permette di modificare lo stato interno della classe e modificare gli attributi di classe (non di istanza)
    # from_dat, essendo un metodo di classe del primo tipo e quindi mi permmette di istanziare la classe in questo modo:
    # bm = BeamMap.from_dat(beam_map_file) (passando quindi solo un file dat)
    # anizhe' in questo modo:
    # bm = BeamMap(pd.DataFrame()) (ovvero passargli un dataframe)
    @classmethod
    def from_dat(cls, filename: Path,
                 comments: str = '#',
                 valid_kids: bool = False,
                 frame: MapMakingFrame = MapMakingFrame.AZEL,
                 par_angle: np.ndarray = np.array([]),
                 dataframe_columns=('id', 'lon_offset', 'lat_offset', 'Tcal pol1', 'Tcal pol2', 'flag')):

        # se non c'e' il file della beammap, lancia un'eccezione
        data = read_dat(filename, comments)

        df = pd.DataFrame(data, columns=dataframe_columns)
        df['id'] = df['id'].astype(int)
        # ~ in modo che 0 True e 1 False
        df['flag'] = ~df['flag'].astype(bool)
        # setto l'indice sulla colonna id del file di beammap
        df.set_index('id', inplace=True)

        df['lon_offset'] = np.deg2rad(df['lon_offset'])
        df['lat_offset'] = np.deg2rad(df['lat_offset'])

        if frame == MapMakingFrame.RADEC:
            df['lon_offset'], df['lat_offset'] = rot(df['lon_offset'], df['lat_offset'], par_angle)

        # se valid_kids e' True ritorno il dataframe con i kid funzionanti (tolgo le righe dei kid non funzionanti,
        # ma l'id e' immutabile)
        return cls(df[df['flag']] if valid_kids else df)

    # usata nel caso non avessimo la beammap
    @classmethod
    def from_fits_cols(cls,
                       lon_offset: np.ndarray,
                       lat_offset: np.ndarray):

        df = pd.DataFrame({"lon_offset": lon_offset, "lat_offset": lat_offset})

        return cls(df)


if __name__ == "__main__":
    beam_map_file = Path('/Volumes/Data/PycharmProjects/mirtos/metadata/chp_offset_rel8_14DEC24_matteo.dat')
    bm = BeamMap.from_dat(beam_map_file)
