import time

import numpy as np
from pathlib import Path
from pprint import pprint
import astropy.units as u
from astropy.io import fits
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field

from memory_profiler import profile

from mirtos.calibration.calibration import SkyDipCalibration
from mirtos.core.projections import conv_radec_to_latlon
from mirtos.core.type_defs.beam_map import BeamMap
from mirtos.core.type_defs.config import (load_config,
                                          MapMakingFrame,
                                          MapMakingProjection,
                                          CalibrationType,
                                          CalibrationConfig,
                                          ScanContext)

from mirtos.core.type_defs.filters import FilteringConfig
from mirtos.core.type_defs.focal_plane import KID, Position
from mirtos.core.multipreprocess import process_all, Job, outputs_valid
from mirtos.filtering.filters import run_filter_steps, get_without_radius_mask


def _get_beammap(beammap_filename: Path,
                 frame: MapMakingFrame,
                 xOffset: np.ndarray,
                 yOffset: np.ndarray,
                 par_angle: np.ndarray):

    if not beammap_filename.exists():
        print(f"Beammap file {beammap_filename} not found. Using xOffset and yOffset from fits file")

        xOffset = np.deg2rad(xOffset)
        yOffset = np.deg2rad(yOffset)
        beammap = BeamMap.from_fits_cols(lon_offset=xOffset, lat_offset=yOffset)

    else:
        beammap = BeamMap.from_dat(beammap_filename,
                                   frame=frame,
                                   par_angle=par_angle if frame == MapMakingFrame.RADEC else None,
                                   valid_kids=True)

    return beammap

@dataclass
class Subscan:
    id: int
    num_feed: int
    obs_datetime: datetime
    ra: np.ndarray
    dec: np.ndarray
    az: np.ndarray
    el: np.ndarray
    par_angle: np.ndarray
    mask: np.ndarray
    time: np.ndarray
    kids: list[KID]
    beammap: BeamMap

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
                         ctx: ScanContext):

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

            # flag_track[i] = 1 indica che non ci troviamo su una rampa e quindi e' valido
            # flag_track[i] = 0 indica che ci troviamo su una rampa e quindi e' da scartare
            # maschera del flag track che, applicata ai dati, nasconde rampe di accelerazione e decelerazione
            # altrimenti prendo tutto il subscan
            flag_track_mask = ft.astype(bool) if ctx.flag_track else np.ones(len(ft), dtype=bool)

            # quanti detector ho (non ch) - l'esclusione dei detector non buoni avviene in beam_map.py
            num_feed = len(hdul[ph_table].data[0])

            hdul_data = hdul[data_table]

            # lista di array su cui viene richiamato isnan
            arrays = [hdul_data.data[ra],
                      hdul_data.data[dec],
                      hdul_data.data['az'],
                      hdul_data.data['el'],
                      hdul_data.data[par_angle] + ctx.angle_offset,
                      hdul_data.data['time']]

            # maschera di bool della stessa lunghezza di ra, dec, ...
            mask = np.ones_like(arrays[0], dtype=bool)
            for array in arrays:
                # maschera congiunta: matrice che ha lungo le righe ra, dec, ...
                # se un valore su una colonna e' nan, tutti gli altri valori su quella colonna vengono messi a nan
                mask &= ~np.isnan(array)

            # combino la maschera che toglie i nan da quella che toglie i flag track
            mask &= flag_track_mask

            ra, dec, az, el, par_angle, time_ = arrays

            # creo le stringhe channel basandomi sui kid
            hdr = ["chp_" + str(i).zfill(3) for i in range(num_feed)]

            kids = []

            beammap = _get_beammap(ctx.beammap_filename,
                                   ctx.frame,
                                   hdul[ph_table].data['lon-offset'],
                                   hdul[ph_table]._data['lat-offset'],
                                   par_angle[mask])

            # named tuples sono tuple che emulano oggetti per cui accedo ai loro elementi con l'operatore '.'
            for row in beammap.beam_map.itertuples():
                kids.append(
                    KID(id=row.Index,  # indice riga beammap
                        pos=Position(row.lon_offset, row.lat_offset),
                        quality_factor=-1,
                        electrical_responsivity=-1,
                        optical_responsivity=-1,
                        gain=np.array([]),
                        saturation_up=-1,
                        saturation_down=-1,
                        ch=row.Index,  # avendo tolto gia' i KID non validi con beammap, qui channel e' uguale a id kid
                        resonance_freq_hz=-1,
                        tod=hdul[ph_table].data[hdr[row.Index]],  # prendo TOD senza nan corrispondenti a ra, dec
                        validity=ctx.detector_validity))

                # di base, la tod del kid va filtrata solo con la maschera del subscan
                kids[-1].mask = mask[:]

            if not np.all(not arr.size for arr in [kids[0].tod, ra, dec, par_angle]):
                return None

            return cls(id_subscan,
                       num_feed,
                       obs_datetime,
                       ra,
                       dec,
                       az,
                       el,
                       par_angle,
                       mask,
                       time_,
                       kids,
                       beammap)

    # @profile
    # calibration sara' l'oggetto istanziato dalla classe Calibration (modificare skydipcalibration.py)
    def process(self,
                projection: MapMakingProjection,
                frame: MapMakingFrame,
                ra_center: float,
                dec_center: float,
                # beam_map_: BeamMap,
                cal_conf: CalibrationConfig,
                filter_conf: FilteringConfig):

        if cal_conf.type == CalibrationType.SKYDIP:
            calibration = SkyDipCalibration.from_fits_file(
                cal_conf.path,
                cal_conf.T_atm,
                cal_conf.tau)

        else:
            raise ValueError(f"Calibration type {cal_conf.type} not supported.")

        # se inplace = True, modifica direttamente le tod dei KID passati
        _, cal_tods = calibration.calibrate(self.kids, self.el, self.mask)

        mask_type = "mask_without_radius"

        if filter_conf.radius.value:

            lon, lat = conv_radec_to_latlon(
                self.ra, self.dec,
                center_ra=ra_center, center_dec=dec_center,
                par_angle=self.par_angle,
                xOffset=self.beam_map.beam_map['lon_offset'],
                yOffset=self.beam_map.beam_map['lat_offset'],
                projection=projection, frame=frame)

            mask_type = "mask_with_radius"
            # da arcsec a rad
            radius = filter_conf.radius.to(u.rad).value
            dist_from_center = np.sqrt((lon - ra_center) ** 2 + (lat - dec_center) ** 2)
            mask = dist_from_center <= radius  # TODO: ritorna matrice (time_samples, time_samples)
            # considero i soli istanti temporali per cui sia lon che lat sono all'interno del cerchio.
            # Infine combino la maschera per i KIDs con la maschera del subscan
            mask = self.mask & mask.all(axis=0)

        else:
            # calcolo la maschera senza avere il raggio definito nello yaml
            mask = get_without_radius_mask(cal_tods, self.mask, filter_conf.mask_without_radius)

        # definisco la lista dei filtri specifici e di quelli common
        # il cui ordine di esecuzione e' quello dello yaml.
        # se mask_type = mask_without_radius, come primo step di filtraggio faccio il linear_detrend
        # se mask_type = mask_with_radius, come primo step di filtraggio faccio il remove_baseline
        filters = getattr(filter_conf.steps, mask_type) + filter_conf.steps.common
        # filtro le tod mascherate, a prescindere che siano mascherate
        # con o senza raggio
        filtered_tods = run_filter_steps(self.time[mask], cal_tods[:, mask], filters)

        # la maschera ottenuta con/senza raggio serve solamente a rimuovere la baseline e non
        # va salvata all'interno del KID proprio perche', uscurando la sorgente, non produrrebbe poi
        # in fase di map-making la mappa che vogliamo
        for kid, cal_tod, filt_tod in zip(self.kids, cal_tods, filtered_tods):
            kid.apply_calibration_inplace(cal_tod)
            kid.tod[mask] = filt_tod


def _longest_common_prefix(s1, s2):
    # Initialize the common prefix as an empty string
    common_prefix = ""

    # Find the minimum length of the two strings
    min_length = min(len(s1), len(s2))

    # Iterate through the characters up to the minimum length
    for i in range(min_length):
        # Compare characters at the same position
        if s1[i] == s2[i]:
            # Append the matching character to the common prefix
            common_prefix += s1[i]
        else:
            # Break the loop if characters do not match
            break

    return common_prefix


@dataclass
class Scan:
    id_scan: str
    ctx: ScanContext
    # se non passiamo Subscan, fiels crea una lista vuota
    subscans: list[Subscan] = field(default_factory=list)

    def __post_init__(self):

        for s in self.subscans:
            lens = {
                "ra": s.ra.shape[0],
                "dec": s.dec.shape[0],
                "par_angle": s.par_angle.shape[0],
                "time": s.time.shape[0],
            }

            if len(set(lens.values())) != 1:
                items = " ".join(f"{k}={v:>5d}" for k, v in lens.items())
                print(f"[WARNING scan={self.id_scan}] Subscan shapes differ: [ {items}]")

    def __add__(self, other):

        # serve a rimuovere l'info "RA" e "DEC" e lasciare solamente la parte in comune
        self.id_scan = _longest_common_prefix(self.id_scan, other.id_scan)

        # estendo i subscan di self con quelli di other
        if isinstance(other, Scan):
            self.subscans.extend(other.subscans)
            return self

        return NotImplemented

    def __len__(self):
        return len(self.subscans)

    # permette di iterare su un oggetto di tipo Scan, in particolare sui suoi subscans.
    # Esempio:
    # scan = Scan(...)
    # for subscan in scan:
    #     ...
    # anziche' fare
    # for subscan in scan.subscans:
    #     ...
    def __getitem__(self, index):
        return self.subscans[index]

    def _concat_data(self, attr: str):
        return np.hstack([getattr(subscan_, attr) for subscan_ in self.subscans])

    @property
    def time(self):
        return self._concat_data('time')

    @property
    def az(self):
        return self._concat_data('az')

    @property
    def el(self):
        return self._concat_data('el')

    @property
    def ra(self):
        return self._concat_data('ra')

    @property
    def dec(self):
        return self._concat_data('dec')

    @property
    def mask(self):
        return np.hstack([sub.mask for sub in self.subscans])

    @property
    def par_angle(self):
        return self._concat_data('par_angle')

    @property
    def kids(self):

        if not self.subscans:
            return []

        kids: list[KID] = []

        for i in range(len(self.subscans[0].kids)):
            k0: KID = self.subscans[0].kids[i]

            tod = np.hstack([sub.kids[i].tod for sub in self.subscans])
            gain = np.hstack([sub.kids[i].gain for sub in self.subscans])
            mask = np.hstack([sub.kids[i].mask for sub in self.subscans])

            k = KID(
                id=k0.id,
                pos=k0.pos,
                quality_factor=k0.quality_factor,
                electrical_responsivity=k0.electrical_responsivity,
                optical_responsivity=k0.optical_responsivity,
                gain=gain,
                saturation_down=k0.saturation_down,
                saturation_up=k0.saturation_up,
                ch=k0.ch,
                resonance_freq_hz=k0.resonance_freq_hz,
                sweep_amplitude=k0.sweep_amplitude,
                sweep_phase=k0.sweep_phase,
                tod=tod,
                validity=k0.validity)

            k.mask = mask

            kids.append(k)

        return kids

    @property
    def tods(self):
        return np.vstack([k.tod for k in self.kids])

    @property
    def calibrated_tods(self):
        return np.vstack([k.calibrated_tod for k in self.kids])

    def process(self, cal_conf: CalibrationConfig, filter_conf: FilteringConfig):

        for subscan in self.subscans:
            subscan.process(
                projection=self.ctx.projection,
                frame=self.ctx.frame,
                ra_center=self.ctx.ra_center,
                dec_center=self.ctx.dec_center,
                beam_map_=self.ctx.beammap,
                cal_conf=cal_conf,
                filter_conf=filter_conf)

    @classmethod
    def from_dir(cls,
                 scan_dir_: Path,
                 ctx: ScanContext):

        fits_files = sorted([p for p in scan_dir_.rglob("*.fits") if "gain" not in p.stem.lower()])
        job_ids = [p.stem.split("_")[-1] for p in fits_files]

        with fits.open(fits_files[0]) as hdul:
            ctx.ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
            ctx.dec_center = hdul['PRIMARY'].header['HIERARCH Declination']

        # ctx.beammap = beammap
        jobs = [Job(job_id, SubscanPayload(p, ctx)) for job_id, p in zip(job_ids, fits_files)]

        subscans = process_all(jobs, process_subscan_file, tb_limit=5)

        return cls(
            id_scan=scan_dir_.name,
            ctx=ctx,
            subscans=outputs_valid(subscans))


# maggiordomo che da' a process_subscan_file le info che gli servono per processare il subscan
# deve contenere tutto quello che serve per runnare from_discos_fits
@dataclass
class SubscanPayload:
    # contiene le informazioni necessarie per processare un singolo subscan
    subscan_filename: Path
    ctx: ScanContext


# funzione che processa un subscan e quindi esegue un singolo job
# deve operare solo con job in quanto rappresenta il singolo compito che deve svolgere
# deve estrarre dall'hdul tutte le informazioni necessarie in quanto portarsi dietro tutto l'hdul e' pesante.
# se ci sono informazioni che devono essere salvate si crea l'attributo relativo e ci si salvano i dati dentro
def process_subscan_file(job: SubscanPayload) -> Subscan:
    # in questo modo accedo alle tod dei KIDS VALIDI in quanto valid_kids e' True
    # for i in range(len(subscan.kids)):
    #     subscan.kids[i].tod

    return Subscan.from_discos_fits(job.subscan_filename, job.ctx)


if __name__ == "__main__":
    base_path = Path(__file__).parents[4]

    config_path = base_path / "configs/a1995_conf.yaml"

    tic = time.perf_counter()
    config = load_config(config_path)
    scan_ra = Scan.from_dir(config.paths.ra_dir, config.scan)
    scan_dec = Scan.from_dir(config.paths.dec_dir, config.scan)
    config.calibration.path = next(config.paths.gain_dir.iterdir())
    scan_ra.process(config.calibration, config.filtering)
    scan_dec.process(config.calibration, config.filtering)

    scan = scan_ra + scan_dec
    print(f"{len(scan_ra) =}, {len(scan_dec) =}, {len(scan) =}")
    #  assert len(scan) == len(scan_ra) + len(scan_dec)

    toc = time.perf_counter()

    print(f"Tempo totale: {toc - tic:.2f}s")


