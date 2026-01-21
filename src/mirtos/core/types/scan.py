import traceback
import numpy as np
from pathlib import Path
from pprint import pprint
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

from mirtos.io.fits import load_subscan_fits
from mirtos.core.types.beam_map import BeamMap
from mirtos.core.config_types import load_config
from mirtos.core.types.focal_plane import KID, Position


@dataclass
class Subscan:
    id_subscan: int
    num_feed: int
    obs_datetime: datetime
    ra: np.ndarray
    dec: np.ndarray
    az: np.ndarray
    el: np.ndarray
    ra_center: float # TODO: controllare se float
    dec_center: float
    par_angle: np.ndarray
    accel_decel_ramps_mask: np.ndarray
    kids: list[KID]

    @classmethod
    def from_discos_fits(cls, subscan_filename: Path, beammap_filename: Path, flag_track: bool = False):


        if not beammap_filename.exists():
            raise FileNotFoundError(f"Beammap file {beammap_filename} not found.")

        id_subscan = int(subscan_filename.stem.split('_')[-1])

        # fileame = '20250402-212049-MISTRAL-A1995_RA_001_002.fits'
        # '20250402_212049'
        # datetime.datetime(2025, 4, 2, 21, 20, 49)
        # obs_datetime.date() mi da' anno mese e giorno
        # obs_datetime.time().hour mi da' l'ora
        obs_datetime = datetime.strptime('_'.join(subscan_filename.stem.split('-')[:2]), '%Y%m%d_%H%M%S')

        hdul = load_subscan_fits(subscan_filename)

        ft = hdul['DATA TABLE'].data['flag_track']

        # maschera del flag track che, applicata ai dati, nasconde rampe di accelerazione e decelerazione
        # altrimenti prendo tutto il subscan
        accel_decel_ramps_mask = ft.astype(bool) if flag_track else np.ones(len(ft), dtype=bool)

        postfix, data_postfix = '', ''
        for hdulist in hdul:
            if 'INTERP' in hdulist.name:
                postfix = 'INTERP'
                data_postfix = '_interpolated'
                break

        data_table = 'DATA TABLE' + postfix
        iq_table = 'IQ TABLE' + postfix
        ph_table = 'PH TABLE' + postfix

        par_angle = 'par_angle' + data_postfix
        ra = 'raj2000' + data_postfix
        dec = 'decj2000' + data_postfix


        angle_offset = 0  # np.pi/2

        hdul_data = hdul[data_table]

        # TODO: ma se ad esempio un ra e' Nan, non dovremmo eliminare anche il dec, az ed el relativo all'istante di
        # tempo di quel ra?
        # lista di array su cui viene richiamato isnan
        arrays = [hdul_data.data[ra],
                  hdul_data.data[dec],
                  hdul_data.data['az'],
                  hdul_data.data['el'],
                  hdul_data.data[par_angle] + angle_offset,
                  hdul_data.data['time']]

        # maschera di bool della stessa lunghezza di ra, dec, ...
        mask = np.ones_like(arrays[0], dtype=bool)
        for array in arrays:
            # maschera congiunta: matrice che ha lungo le righe ra, dec, ...
            # se un valore su una colonna e' nan, tutti gli altri valori su quella colonna vengono messi a nan
            mask &= ~np.isnan(array)

        ra = arrays[0][mask]
        dec =  arrays[1][mask]
        az =  arrays[2][mask]
        el =  arrays[3][mask]
        par_angle =  arrays[4][mask]
        time_ =  arrays[5][mask]

        ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
        dec_center = hdul['PRIMARY'].header['HIERARCH Declination']

        # quanti detector ho (non ch) - ammazzati da exluce_channels
        num_feed = len(hdul["PH TABLE"].data[0])

        # # non sono nel fits, ma usano cose del file fits - POSSIAMO ACCEDERVI QUANDO CREIAMO L'OGGETTO
        # cls.range_timestep = range(0, len(cls.par_angle),
        #                            1)  # (1222, 147704) daisy #(1209, 9647) 30s #where the fits file is not 'NULL' or not zero
        # cls.all_range_timestep.append(cls.range_timestep)
        #
        # cls.num_timestep = int(len(cls.range_timestep))
        # # TODO: freq di acquisizione dei dati - nello yaml del receiver
        # cls.sample_freq = 244.140625  # 256e6/2**20 HZ
        # # dt
        # cls.tstep = 1 / cls.sample_freq
        # cls.nyqfreq = cls.sample_freq / 2
        #
        #
        # # appesi per checkare che sono gli stessi per tutti i subscan
        # cls.num_feed_all.append(cls.num_feed)

        # creo le stringhe channel basandomi sui kid
        hdr = ["chp_" + str(i).zfill(3) for i in range(num_feed)]

        beammap = BeamMap.from_dat(beammap_filename, valid_kids=True)
        kids = []

        # named tuples: per cui accedo a cio' che contengono come se fossero degli oggetti
        for row in beammap.beammap_filename.itertuples():


            kids.append(
                KID(id=row.id, # indice riga beammap
                    pos=Position(row.lon_offset, row.lat_offset),
                    quality_factor=-1,
                    electrical_responsivity=-1,
                    optical_responsivity=-1,
                    gain=-1,
                    saturation_up=-1,
                    saturation_down=-1,
                    ch=row.id, # avendo tolto gia' i KID non validi con beammap, qui channel e' uguale a id kid
                    resonance_freq_hz=row.resonance_freq_hz,
                    sweep_amplitude=np.array([]),
                    sweep_phase=np.array([]),
                    tod=hdul['PH TABLE'].data[hdr[row.id]][mask]) # prendo TOD senza nan corrispondenti a ra, dec, ...
            )

        # # cls.timestream_raw = []
        # # cls.timestream_raw = hdul['PH TABLE'].metadata[hdr[0]][flag_track]
        # for i in range(0, num_feed):
        #     # try:
        #     # cls.timestream_raw.append(hdul['PH TABLE'].metadata[hdr[i]][flag_track])
        #     # estrazione TOD basata su ch
        #     ts_nan = hdul['PH TABLE'].data[hdr[i]]
        #     # attributo TOD_raw, lista di TOD di tutti i canali
        #     timestream_raw.append(ts_nan[np.logical_not(np.isnan(ra_nan))])
        #     # except:
        #     #    cls.timestream_raw.append([np.nan]*cls.num_timestep)
        #
        #     # mettere extract_data e usare dati beam_map
        #     # TODO: Controllo con exclude channel: poppa fuori i tod dei canali per cui non c'e' un offset associato nella beam_map
        #     # alla fine devo avere il le TOD con i channel non esclusi
        #     # exlude channel opera sugli attributi di Subscan e li deve riassegnare
        return cls(id_subscan,
                   num_feed,
                   obs_datetime,
                   ra,
                   dec,
                   az,
                   el,
                   ra_center,
                   dec_center,
                   par_angle,
                   accel_decel_ramps_mask,
                   kids)



@dataclass
class Scan:
    id_scan: str
    # se non passiamo Subscan, fiels crea una lista vuota
    subscans: list[Subscan] = field(default_factory=list)


# maggiordomo che da' a process_subscan_file le info che gli servono per processare il subscan
# deve contenere tutto quello che serve per runnare from_discos_fits
@dataclass
class Job:

    id: str # job id
    flag_track: bool
    subscan_filename: Path
    beammap_filename: Path


    # aprire file subscan con mirtos.io.fits.load_subscan_fits

    # a noi dell'hdul serve tutto quello che sta in read_discos_fits +
    # self.xOffset, self.yOffset, self.tod_raw, self.num_feed, self.excl_feed = self.exclude_channels(self.tod_raw, self.num_feed, self.dati_beammap)
    # con check se la beamap esiste, altrtimenti prendi offsets dal fits file (ma non li trova se non c'e' il dat)
    # flaggare errore

# contiene esito esecuzione job, l'errore ritornato e in caso di esito positivo i dati processati
@dataclass
class Result:

    job: Job
    subscan: Subscan = None
    error: str = None

# funzione che processa un subscan e quindi esegue un singolo job
# deve operare solo con job in quanto rappresenta il singolo compito che deve svolgere
# deve estrarre dall'hdul tutte le informazioni necessarie in quanto portarsi dietro tutto l'hdul e' pesante.
# se ci sono informazioni che devono essere salvate si crea l'attributo relativo e ci si salvano i dati dentro
def process_subscan_file(job: Job) -> Result:

    try:

        subscan = Subscan.from_discos_fits(job.subscan_filename, job.beammap_filename, flag_track=job.flag_track)

        # in questo modo accedo alle tod dei KIDS VALIDI in quanto valid_kids e' True
        # for i in range(len(subscan.kids)):
        #     subscan.kids[i].tod

        return Result(job, subscan=subscan, error='')

    except Exception as e:
        return Result(job, subscan=None, error=traceback.format_exc(3)) # ultime tre path dei file che hanno generato l'errore'

# gli passo una lista di Job, ovvero un Job per ogni file subscan da processare
# TODO: nel modulo preprocessing io dovro' importare process_all_subscan e passargli la lista di Job come
# faccio nell'if __name__
# la struttura dovra' sempre essere con Job e Result un process_subscan
def process_all_subscans(jobs: list[Job]):


    with ProcessPoolExecutor() as executor:

        # schedula esecuzione funzione in parallelo
        # lista di oggetti Result
        futures = {executor.submit(process_subscan_file, job): job for job in jobs}

        # ritorna i task completati, ovvero i partial subscans processati
        subscans = [r.result().subscan for r in as_completed(futures)]

        return subscans


if __name__ == "__main__":

    scan_dir = Path('/Volumes/Data/PycharmProjects/mirtos/data/input/20250402-212049-MISTRAL-A1995_RA_001_002.fits')
    config_path = Path('/Volumes/Data/PycharmProjects/mirtos/configs/config.yaml')
    beam_map = Path('/Volumes/Data/PycharmProjects/mirtos/metadata/chp_offset_rel8_14DEC24_matteo.dat')

    subscan = Subscan.from_discos_fits(scan_dir, beam_map)
    pprint(subscan)


    config = load_config(config_path)
    flag_track = config.flag_track
    job_ids = []
    # scorrere sugli id che per noi sono le due triplette finali
    for file in scan_dir.iterdir():

        # 'dirfile_20240611_124429__MERGED_WITH__20240611-124423-MISTRAL-JUPITER_FOCUS_SCAN_001_108' il suo id e' 001_108
        job_ids.append('_'.join(file.stem.split("_")[-1]))

    jobs = [Job(job_id, flag_track, file, None) for job_id, file in zip(job_ids, scan_dir.iterdir())]

    subscans = process_all_subscans(jobs)

    pprint(subscans[:10])


