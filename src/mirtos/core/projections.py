import numpy as np
from astropy.wcs import WCS


@np.vectorize
def rot(x,y,theta):
    '''
    xy = (x,y)

    mat_rot = ([np.cos(theta), -np.sin(theta)],
            [np.sin(theta),np.cos(theta)])
    
    ra_f, dec_f = np.matmul(mat_rot,xy)
    '''
    ra_f = x*np.cos(theta) - y*np.sin(theta)
    dec_f = x*np.sin(theta) + y*np.cos(theta)

    return ra_f, dec_f

def proj_radec_to_xy(ra,dec,ra0,dec0, projection):
        
    if projection=='SIN':
        lam = ra
        phi = dec
        #phi1 = self.scan_center_ra
        lam0 = ra0

        x = (lam-lam0)*np.cos(phi) + lam0
        y = phi
        
        return x, y

    if projection=='GNOM':
            #https://mathworld.wolfram.com/GnomonicProjection.html
        phi = dec
        lam = ra
        phi1 = dec0
        lam0 = ra0

        c = np.sin(phi1)*np.sin(phi) + np.cos(phi1)*np.cos(phi)*np.cos(lam-lam0)
        x = (np.cos(phi) * np.sin(lam-lam0))/c
        y = (np.cos(phi1)*np.sin(phi) - np.sin(phi1)*np.cos(phi)*np.cos(lam-lam0))/c
                        
        return x, y

    else:
        raise ValueError(projection, ': this projection not available.')

def conv_xy_to_latlon(x, y, par_angle, num_feed, xOffset, yOffset,  center_ra, center_dec, frame):
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
        
        for i in range(num_feed):
                        xoff_rot, yoff_rot = rot([xOffset[i]] * len(par_angle), [yOffset[i]] * len(par_angle), par_angle)
                        offset_x.append(xoff_rot)
                        offset_y.append(yoff_rot) 
                                                
        for i in range(num_feed):        
            lat.append(y + offset_y[i])
            lon.append(x - offset_x[i])

    elif frame == 'AZEL':
        
        x_rot, y_rot = rot(x - center_ra, y - center_dec, par_angle)
        
        for i in range(num_feed):
            #try:
            lat.append(y_rot + yOffset[i])
            lon.append(x_rot - xOffset[i]/np.cos(y_rot + yOffset[i]))
            #except:
            #    lat.append(y_rot)
            #    lon.append(x_rot/np.cos(y_rot))

    else:      
        raise ValueError(frame + '-> this set of coordinates is not available.')
    
    return lat, lon
