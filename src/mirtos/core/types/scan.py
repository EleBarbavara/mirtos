import numpy as np
from pathlib import Path
from pprint import pprint
import astropy.units as u
from astropy.io import fits
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from mirtos.calibration.calibration import Calibration, SkyDipCalibration
from mirtos.core.projections import proj_radec_to_xy, conv_xy_to_latlon
from mirtos.core.types.beam_map import BeamMap
from mirtos.core.types.config import load_config, DetectorValidityConfig, BinnerFrame, BinnerProjection, FilteringSteps
from mirtos.core.types.filters import MaskWithoutRadius
from mirtos.core.types.focal_plane import KID, Position
from mirtos.core.multipreprocess import process_all, Job

from mirtos.filtering.filters import remove_baseline, linear_detrend, run_filter_steps, get_without_radius_mask


@dataclass
class Subscan:
    id_subscan: int
    num_feed: int
    obs_datetime: datetime
    ra: np.ndarray
    dec: np.ndarray
    az: np.ndarray
    el: np.ndarray
    ra_center: float
    dec_center: float
    par_angle: np.ndarray
    flag_track: np.ndarray
    time: np.ndarray
    kids: list[KID]

    # property mi consente di creare un attributo al volo usando gli attributi di Subscan
    @property
    def sampling_frequency(self):
        # 244.140625 = 256e6/2**20 HZ
        return np.median(np.diff(1 / self.time)) * u.hertz

    @property
    def nyquist_frequency(self):
        return self.sampling_frequency / 2

    @property
    def sampling_time(self):
        return 1 / self.sampling_frequency * u.second

    @classmethod
    def from_discos_fits(cls,
                         subscan_filename: Path,
                         detector_validity: DetectorValidityConfig,
                         beammap_filename: Path | None, # puo' esserci o meno la beamap
                         binner_frame: BinnerFrame = BinnerFrame.AZEL, # lo deve passare a from_dat
                         flag_track_: bool = False,
                         angle_offset: float = 0):

        # TODO: per ora gli passiamo angle_offset, da capire dove va immesso

        id_subscan = int(subscan_filename.stem.split('_')[-1])

        # fileame = '20250402-212049-MISTRAL-A1995_RA_001_002.fits'
        # '20250402_212049'
        # datetime.datetime(2025, 4, 2, 21, 20, 49)
        # obs_datetime.date() mi da' anno mese e giorno
        # obs_datetime.time().hour mi da' l'ora
        obs_datetime = datetime.strptime('_'.join(subscan_filename.stem.split('-')[:2]), '%Y%m%d_%H%M%S')

        with fits.open(subscan_filename) as hdul:

            postfix, data_postfix = '', ''
            for hdulist in hdul:
                if 'INTERP' in hdulist.name:
                    postfix = 'INTERP'
                    data_postfix = '_interpolated'
                    break

            data_table = 'DATA TABLE' + postfix
            ph_table = 'PH TABLE' + postfix
            # iq_table = 'IQ TABLE' + postfix

            par_angle = 'par_angle' + data_postfix
            ra = 'raj2000' + data_postfix
            dec = 'decj2000' + data_postfix

            ft = hdul[data_table].data['flag_track']

            # maschera del flag track che, applicata ai dati, nasconde rampe di accelerazione e decelerazione
            # altrimenti prendo tutto il subscan
            flag_track_mask = ft.astype(bool) if flag_track_ else np.ones(len(ft), dtype=bool)

            ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
            dec_center = hdul['PRIMARY'].header['HIERARCH Declination']

            # quanti detector ho (non ch) - l'esclusione dei detector non buoni avviene in beam_map.py
            num_feed = len(hdul[ph_table].data[0])

            hdul_data = hdul[data_table]

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
            dec = arrays[1][mask]
            az = arrays[2][mask]
            el = arrays[3][mask]
            par_angle = arrays[4][mask]
            time_ = arrays[5][mask]

            # creo le stringhe channel basandomi sui kid
            hdr = ["chp_" + str(i).zfill(3) for i in range(num_feed)]

            # costruisco beammap a partire da file fits se non viene passato un file beammap.dat
            if isinstance(beammap_filename, Path):
                if not beammap_filename.exists():

                    print(f"Beammap file {beammap_filename} not found. Using xOffset and yOffset from fits file.")

                    xOffset = np.deg2rad(hdul['FEED TABLE'].data['xOffset'])
                    yOffset = np.deg2rad(hdul['FEED TABLE'].data['yOffset'])
                    beammap = BeamMap.from_fits_cols(lon_offset=xOffset, lat_offset=yOffset)

                else:
                    beammap = BeamMap.from_dat(beammap_filename,
                                               frame=binner_frame,
                                               par_angle=par_angle,
                                               valid_kids=True)

            kids = []

            # named tuples sono tuple che emulano oggetti per cui accedo ai loro elementi con l'operatore '.'
            for row in beammap.beam_map.itertuples():
                kids.append(
                    KID(id=row.Index,  # indice riga beammap
                        pos=Position(row.lon_offset, row.lat_offset),
                        quality_factor=-1,
                        electrical_responsivity=-1,
                        optical_responsivity=-1,
                        gain=-1,
                        saturation_up=-1,
                        saturation_down=-1,
                        ch=row.Index,  # avendo tolto gia' i KID non validi con beammap, qui channel e' uguale a id kid
                        resonance_freq_hz=-1,
                        sweep_amplitude=np.array([]),
                        sweep_phase=np.array([]),
                        tod=hdul[ph_table].data[hdr[row.Index]][mask],
                        validity=detector_validity)  # prendo TOD senza nan corrispondenti a ra, dec, ...
                )

            # timestream_raw = []
            # timestream_raw = hdul['PH TABLE'].metadata[hdr[0]][flag_track]
            # for i in range(0, num_feed):
            #     try:
            #         timestream_raw.append(hdul['PH TABLE'].metadata[hdr[i]][flag_track])
            #         # estrazione TOD basata su ch
            #         ts_nan = hdul['PH TABLE'].data[hdr[i]]
            #         # attributo TOD_raw, lista di TOD di tutti i canali
            #         timestream_raw.append(ts_nan[np.logical_not(np.isnan(ra_nan))])
            #     except:
            #         timestream_raw.append([np.nan]*num_timestep)

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
                       flag_track_mask,
                       time_,
                       kids)

    # calibration sara' l'oggetto istanziato dalla classe Calibration (modificare skydipcalibration.py)
    def process(self,
                projection: BinnerProjection,
                frame: BinnerFrame,
                calibration: Calibration,
                radius: Optional[u.arcsec],
                steps: FilteringSteps,
                mask_without_radius: Optional[MaskWithoutRadius]):

        x, y = proj_radec_to_xy(self.ra, self.dec, self.ra_center, self.dec_center, projection)
        lon, lat = conv_xy_to_latlon(x, y,
                                     self.par_angle,
                                     self.az,
                                     self.el,
                                     self.ra_center,
                                     self.dec_center,
                                     frame)

        tods_calibrated = calibration.calibrate(self.kids)

        for kid, tod in zip(self.kids, tods_calibrated):
            # associo ai kid le tod calibrate
            kid.tod = tod

        tods = np.vstack([k.tod for k in self.kids])
        mask_type = "mask_without_radius"

        if radius:
            mask_type = "mask_with_radius"
            # da arcsec a rad
            radius = radius.to(u.rad).value
            dist_from_center = np.sqrt((lon - self.ra_center) ** 2 + (lat - self.dec_center) ** 2)
            mask = dist_from_center <= radius

        else:
            # calcolo la maschera senza avere il raggio definito nello yaml
            mask = get_without_radius_mask(tods, mask_without_radius)

        # definisco la lista dei filtri specifici e di quelli common
        # il cui ordine di esecuzione e' quello dello yaml.
        # se mask_type = mask_without_radius, come primo step di filtraggio faccio il linear_detrend
        # se mask_type = mask_with_radius, come primo step di filtraggio faccio il remove_baseline
        filters = getattr(steps, mask_type) + steps.common
        # filtro le tod mascherate, a prescindere che siano mascherate
        # con o senza raggio
        tods = run_filter_steps(self.time, tods[:, mask], filters)

        # aggiorno la maschera e la tod per ogni kid
        for kid, tod in zip(self.kids, tods):
            kid.mask, kid.tod = mask, tod




@dataclass
class Scan:
    id_scan: str
    # se non passiamo Subscan, fiels crea una lista vuota
    subscans: list[Subscan] = field(default_factory=list)

    def __post_init__(self):

        for attr in ['num_feed', 'par_angle', 'ra', 'dec']:

            # getattr chiede a subscan di accedere all'attributo attr e di ritornare il suo valore.
            # con le {} creo un set (insieme) comprehension in quanto negli insieme non posso esserci valori duplicati,
            # se tutti i subscan hanno lo stesso num_feed, l'insieme avra' lunghezza 1
            if len(all_equal := {getattr(subscan, attr) for subscan in self.subscans}) != 1:
                raise ValueError(f"Subscans in scan {self.id_scan} have different number of {attr}: {all_equal}")

        # queste tre istruzioni sono automaticamente controllate nel ciclo
        # (1222, 147704) daisy #(1209, 9647) 30s #where the fits file is not 'NULL' or not zero
        # cls.range_timestep = range(0, len(cls.par_angle), 1)
        # cls.all_range_timestep.append(cls.range_timestep)
        # cls.num_timestep = int(len(cls.range_timestep))


# maggiordomo che da' a process_subscan_file le info che gli servono per processare il subscan
# deve contenere tutto quello che serve per runnare from_discos_fits
@dataclass
class SubscanPayload:
    # contiene le informazioni necessarie per processare un singolo subscan
    flag_track: bool
    subscan_filename: Path
    beammap_filename: Path
    detector_validity: DetectorValidityConfig
    binner_frame: BinnerFrame = BinnerFrame.AZEL
    angle_offset: float = 0

    # aprire file subscan con mirtos.io.fits.load_subscan_fits

    # a noi dell'hdul serve tutto quello che sta in read_discos_fits +
    # self.xOffset, self.yOffset, self.tod_raw, self.num_feed, self.excl_feed = self.exclude_channels(self.tod_raw, self.num_feed, self.dati_beammap)
    # con check se la beamap esiste, altrtimenti prendi offsets dal fits file (ma non li trova se non c'e' il dat)
    # flaggare errore


# funzione che processa un subscan e quindi esegue un singolo job
# deve operare solo con job in quanto rappresenta il singolo compito che deve svolgere
# deve estrarre dall'hdul tutte le informazioni necessarie in quanto portarsi dietro tutto l'hdul e' pesante.
# se ci sono informazioni che devono essere salvate si crea l'attributo relativo e ci si salvano i dati dentro
def process_subscan_file(job: SubscanPayload) -> Subscan:
    # in questo modo accedo alle tod dei KIDS VALIDI in quanto valid_kids e' True
    # for i in range(len(subscan.kids)):
    #     subscan.kids[i].tod

    return Subscan.from_discos_fits(job.subscan_filename, job.detector_validity, job.beammap_filename,
                                    flag_track_=job.flag_track)


if __name__ == "__main__":

    scan_dir = Path('/Volumes/Data/PycharmProjects/mirtos/data/input')
    subscan_fit = scan_dir / '20250402-212049-MISTRAL-A1995_RA_001_002.fits'
    config_path = Path('/Volumes/Data/PycharmProjects/mirtos/configs/config.yaml')
    beam_map = Path('/Volumes/Data/PycharmProjects/mirtos/metadata/chp_offset_rel8_14DEC24_matteo.dat')
    skydip_path = Path('/Volumes/Data/PycharmProjects/mirtos/skydip/skydip_calibration.dat')

    subscan = Subscan.from_discos_fits(subscan_fit,
                                       detector_validity=DetectorValidityConfig(),
                                       beammap_filename=beam_map,
                                       flag_track_=True)

    # subscan.process(, steps=)

    # config = load_config(config_path)
    # flag_track = config.flag_track
    flag_track = True
    job_ids = []
    # scorrere sugli id che per noi sono le due triplette finali
    for file in scan_dir.iterdir():
        # 'dirfile_20240611_124429__MERGED_WITH__20240611-124423-MISTRAL-JUPITER_FOCUS_SCAN_001_108' il suo id e' 001_108
        job_ids.append(file.stem.split("_")[-1])

    jobs = [Job(job_id, SubscanPayload(flag_track, file, beam_map, DetectorValidityConfig()))
            for job_id, file in zip(job_ids, scan_dir.iterdir()) if file.suffix == '.fits']

    # lista di oggetti Result che conterranno job, output, error
    res = process_all(jobs, process_subscan_file)
    pprint(res[0].output.kids[1])

    subscan = res[0].output
    config = load_config(config_path)
    tau = config.paths.tau
    T_atm = config.paths.T_atm
    skydipcalibration = SkyDipCalibration.from_fits_file(skydip_path, T_atm, tau)
    subscan.process('ITRF', 'ITRF', skydipcalibration, radius=None)

    # subscans = process_all_subscans(jobs)
    #
    # pprint(subscans)
