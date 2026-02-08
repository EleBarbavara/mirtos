import numpy as np
from mirtos.core.types.config import MapMakingProjection, MapMakingFrame


def rot(x, y, theta):
    '''
    xy = (x,y)

    mat_rot = ([np.cos(theta), -np.sin(theta)],
            [np.sin(theta),np.cos(theta)])
    
    ra_f, dec_f = np.matmul(mat_rot,xy)
    '''

    c = np.cos(theta)
    s = np.sin(theta)

    # return ra / dec
    return x * c - y * s, x * s + y * c


def proj_radec_to_xy(ra, dec, ra0, dec0, projection):
    if projection == MapMakingProjection.SIN:
        x = (ra - ra0) * np.cos(dec) + ra0
        y = dec
        return x, y

    if projection == MapMakingProjection.GNOM:
        #https://mathworld.wolfram.com/GnomonicProjection.html

        c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
        x = (np.cos(dec) * np.sin(ra - ra0)) / c
        y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / c
        return x, y

    else:
        raise ValueError(projection, ': this projection not available.')


def conv_xy_to_latlon(x, y, par_angle, xOffset, yOffset, center_ra, center_dec, frame):

    if frame == MapMakingFrame.RADEC:
        # broadcast offsets (num_feed,1) contro angle (1,N)
        xO = xOffset[:, None]
        yO = yOffset[:, None]
        theta = par_angle[None, :]

        xoff_rot, yoff_rot = rot(xO, yO, theta)  # (num_feed, N)

        lat = y[None, :] + yoff_rot  # (num_feed, N)
        lon = x[None, :] - xoff_rot  # (num_feed, N)
        return lat, lon

    elif frame == MapMakingFrame.AZEL:
        x_rot, y_rot = rot(x - center_ra, y - center_dec, par_angle)  # (N,)

        xO = xOffset[:, None]  # (num_feed,1)
        yO = yOffset[:, None]  # (num_feed,1)

        lat = y_rot[None, :] + yO  # (num_feed, N)
        lon = x_rot[None, :] - xO / np.cos(lat)  # (num_feed, N)
        return lat, lon

    else:
        raise NotImplementedError(f"`{frame}` frame not available")


def conv_radec_to_latlon(ra, dec,  center_ra, center_dec, projection, par_angle, xOffset, yOffset, frame):

    x, y = proj_radec_to_xy(ra, dec, center_ra, center_dec, projection=projection)
    return conv_xy_to_latlon(x, y, par_angle, xOffset, yOffset, center_ra, center_dec, frame)