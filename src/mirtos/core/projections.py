import numpy as np
from astropy.wcs import WCS


@np.vectorize
def rot(x, y, theta):
    '''
    xy = (x,y)

    mat_rot = ([np.cos(theta), -np.sin(theta)],
            [np.sin(theta),np.cos(theta)])
    
    ra_f, dec_f = np.matmul(mat_rot,xy)
    '''

    c = np.cos(theta)
    s = np.sin(theta)

    # dim num_feed x N
    ra_f = x * c - y * s
    dec_f = x * s + y * c

    return ra_f, dec_f

def proj_radec_to_xy(ra,dec,ra0,dec0, projection):
        
    if projection == 'SIN':
        x = (ra - ra0) * np.cos(dec) + ra0
        y = dec

        return x, y


    if projection == 'GNOM':
        #https://mathworld.wolfram.com/GnomonicProjection.html

        c = np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra-ra0)
        x = (np.cos(dec) * np.sin(ra-ra0))/c
        y = (np.cos(dec0)*np.sin(dec) - np.sin(dec0)*np.cos(dec)*np.cos(ra-ra0))/c
                        
        return x, y

    else:
        raise ValueError(projection, ': this projection not available.')

def conv_xy_to_latlon(x, y, par_angle, xOffset, yOffset,  center_ra, center_dec, frame):
    lon = []
    lat = []
    
    if frame == 'RADEC':

        '''  
        xoff_rot = []
        yoff_rot = []
        for i in range(num_feed):
                x_rot, y_rot = lib.rot([offset_x[i]]*len(par_angle), [offset_y[i]]*len(par_angle), par_angle)
                xoff_rot.append(x_rot)
                yoff_rot.append(y_rot)
        print(np.shape(xoff_rot)) 
        '''     

        offset_x = []
        offset_y = []

        # il ciclo for mi riduce di una dimensione, ma e' possibile lavorare in modo matriciale
        # vettori colonna num_feed x 1
        x0 = xOffset[:, np.newaxis]
        y0 = yOffset[:, np.newaxis]
        # 1 riga per N colonne (numero di angoli parallattici)
        par_angle = par_angle[np.newaxis, :]

        # per ogni KID e angolo parallattico ho il rotore dell'xoff e dell'yoff
        # ciascuno ha dim num_feed x N
        xoff_rot, yoff_rot = rot(x0, y0, par_angle)

        # ra e dec hanno dimensione N (cosi come lat e lon)
        # ra e dec (x, y) vengono resi di dimensione num_feed x N in modo da usare il broadcasting di python
        # e sommargli direttamente yoff_rot e xoff_rot che hanno dimensione num_feed x N
        # ottengo cosi delle matrici lat e lon: per ogni KID (righe) e valore dell'angolo parallattico ho la
        # rispettiva lat e lon
        lat = y[np.newaxis, :] + yoff_rot
        lon = x[np.newaxis, :] - xoff_rot

        return lat, lon

    elif frame == 'AZEL':

        # x_rot e y_rot hanno dimensione 1 x N
        x_rot, y_rot = rot(x - center_ra, y - center_dec, par_angle)

        # num_feed x 1
        x0 = xOffset[:, np.newaxis]
        y0 = yOffset[:, np.newaxis]

        # y_rot[np.newaxis, :] ha dim num_feed x N e y0 ha dim num_feed x 1
        lat = y_rot[np.newaxis, :] + y0
        lon = x_rot[np.newaxis, :] - x0 / np.cos(lat)

        return lat, lon

    else:      
        raise ValueError(frame + '-> this set of coordinates is not available.')
    


if __name__ == "__main__":

    # leggo sia i dati che ho estrapolato dal fits (N, ra, dec, ...)
    # che i risultati ottenuti con i vecchi metodi di projections
    expected_projection = np.load('../../../test/projections/expected_projection_test_data.npz')

    # dati veri del fits
    N = expected_projection['N']
    # len(subscan.kids): numero di kids validi
    num_feed = expected_projection['num_feed'] # subscan.num_feed: numero totale di kids
    ra = expected_projection['ra']
    dec = expected_projection['dec']
    center_ra = expected_projection['ra0']
    center_dec = expected_projection['dec0']
    xOffset = expected_projection['xOffset']
    yOffset = expected_projection['yOffset']
    par_angle = expected_projection['par_angle']

    # calcolo i risultati delle funzioni della nuova versione di projections.py
    rot_x, rot_y = rot(ra, dec, par_angle)

    x_sin, y_sin = proj_radec_to_xy(ra, dec, center_ra, center_dec, 'SIN')
    x_gnom, y_gnom = proj_radec_to_xy(ra, dec, center_ra, center_dec, 'GNOM')

    lat_radec_sin, lon_radec_sin = conv_xy_to_latlon(x_sin, y_sin, par_angle, xOffset, yOffset, center_ra,
                                                     center_dec, 'RADEC')
    lat_azel_sin, lon_azel_sin = conv_xy_to_latlon(x_sin, y_sin, par_angle, xOffset, yOffset, center_ra,
                                                   center_dec, 'AZEL')

    lat_radec_gnom, lon_radec_gnom = conv_xy_to_latlon(x_gnom, y_gnom, par_angle, xOffset, yOffset, center_ra,
                                                       center_dec, 'RADEC')
    lat_azel_gnom, lon_azel_gnom = conv_xy_to_latlon(x_gnom, y_gnom, par_angle, xOffset, yOffset, center_ra,
                                                     center_dec, 'AZEL')


    np.savez('../../../test/projections/output_projection_test_data.npz',
             N=N,
             num_feed=num_feed,
             ra=ra,
             dec=dec,
             ra0=center_ra,
             dec0=center_dec,
             xOffset=xOffset,
             yOffset=yOffset,
             par_angle=par_angle,
             center_ra=center_ra,
             center_dec=center_dec,
             rot_x=rot_x,
             rot_y=rot_y,
             x_sin=x_sin,
             y_sin=y_sin,
             x_gnom=x_gnom,
             y_gnom=y_gnom,
             lat_radec_sin=lat_radec_sin,
             lon_radec_sin=lon_radec_sin,
             lat_azel_sin=lat_azel_sin,
             lon_azel_sin=lon_azel_sin,
             lat_radec_gnom=lat_radec_gnom,
             lon_radec_gnom=lon_radec_gnom,
             lat_azel_gnom=lat_azel_gnom,
             lon_azel_gnom=lon_azel_gnom,
             )