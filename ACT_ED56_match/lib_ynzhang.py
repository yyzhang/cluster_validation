import healpy as hp
import healsparse as hsp
import numpy as np
from scipy.spatial import cKDTree
import astropy.io.fits as pyfits
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatwCDM
cosmo = FlatwCDM(H0=70, Om0=0.3)

def footprint_check(mask_file, ra, dec):
    masks=pyfits.open(mask_file)[1].data

    nside = 4096
    theta = (90.0 - dec)*np.pi/180.
    phi = ra*np.pi/180.
    pix_acts=hp.ang2pix(nside, theta, phi)
    pixels_acts= np.array([pix_acts, np.ones(len(pix_acts))]).transpose()
    
    masks=masks[ np.where( (masks['HPIX'] >= np.min(pix_acts)) & (masks['HPIX'] <= np.max(pix_acts)) )]
    pixels= np.array([masks['HPIX'], masks['FRACGOOD']]).transpose()
    tree=cKDTree(pixels_acts)


    dis, inds = tree.query(pixels , k=1, p=1)
    ind_keep=inds[np.where(dis < 0.05)]

    return ind_keep

def footprint_check_python(mask_file, ra, dec):
    masks=hsp.HealSparseMap.read(mask_file)
    values = masks.get_values_pos(ra, dec, lonlat=True)
    ind_keep= np.where(values['fracgood'] > 0.95)
    return ind_keep

def footprint_check_python_zmax(mask_file, ra, dec, zs):
    masks=hsp.HealSparseMap.read(mask_file)
    values = masks.get_values_pos(ra, dec, lonlat=True)
    ind_keep= np.where( (values['fracgood'] > 0.95) & (values['zmax'] > zs))
    return ind_keep

def match_ACT_to_redmapper(act_ra, act_dec, clusters_ra, clusters_dec, dAs, z1, z2):

    acts_coords=SkyCoord(act_ra, act_dec, frame='icrs', unit='deg')
    rdmp_coords=SkyCoord(clusters_ra, clusters_dec, frame='icrs', unit='deg')
    ind_acts = np.zeros(len(act_ra), dtype=np.int64)
    ind_rdmp = np.zeros(len(act_ra), dtype=np.int64)
    dis_acts_rp = np.zeros(len(act_ra))
    for ii in range(len(act_ra)):
        sep=rdmp_coords.separation(acts_coords[ii]).arcminute
        comp_dis = sep/60.0/180.0*np.pi #* dAs
        #comp_dis=cosmo.kpc_proper_per_arcmin(dAs).value * sep/1000.0
        ind_temp, =np.where(comp_dis*dAs < 1.)
        if (len(ind_temp)) > 1:
            print('More than one redmapper matches for RA %f DEC %f'%(act_ra[ii], act_dec[ii]))
        dis_acts_rp_ind = np.argmin(comp_dis)
        ind_rdmp[ii]=dis_acts_rp_ind
        ind_acts[ii]=ii
        dis_acts_rp[ii] = comp_dis[dis_acts_rp_ind] * dAs[dis_acts_rp_ind]

    ind_keep, =np.where(dis_acts_rp < 1.5)
    return ind_keep, ind_rdmp[ind_keep], dis_acts_rp[ind_keep]

def est_centerfrac(dist):
    ind, =np.where(dist <0.05)
    return np.float(len(ind))/np.float(len(dist))
