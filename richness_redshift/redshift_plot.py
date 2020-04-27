import numpy as np
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
import matplotlib.pyplot as plt 
import astropy.io.fits as pyfits
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import astropy.coordinates as coord
import astropy.units as u
import healpy as hp
import healsparse as hsp

def setup(options):
    section = option_section

    # read in redmapper catalog
    redmapper_file = options[section,"redmapper_file"]
    redmapper_ra_col = options[section,"redmapper_racol"] 
    redmapper_dec_col = options[section,"redmapper_deccol"] 
    redmapper_lambda_col = options[section,"redmapper_lambdacol"]
    redmapper_z_col = options[section,"redmapper_zcol"]
    max_lambda = options[section,"redmapper_maxlambda"]
    min_lambda = options[section,"redmapper_minlambda"]
    max_z = options[section,"redmapper_maxz"]
    min_z = options[section,"redmapper_minz"]
    
    maskfile = options[section,"redmapper_maskfile"]

    redmapper = pyfits.open(redmapper_file)[1].data
    redmapper_ra = redmapper[redmapper_ra_col]
    redmapper_dec = redmapper[redmapper_dec_col]
    redmapper_lambda = redmapper[redmapper_lambda_col]
    redmapper_z = redmapper[redmapper_z_col]

    return redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, max_lambda, min_lambda, max_z, min_z, maskfile

def execute(block, config):
    redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, max_lambda, min_lambda, max_z, min_z, maskfile = config

    richness_bins = np.exp(np.linspace(np.log(min_lambda), np.log(max_lambda), 4))
    z_bins = np.linspace(min_z, max_z, 4)
    masks=hsp.HealSparseMap.read(maskfile)
    make_plots(redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, richness_bins, z_bins, masks)

   
    return 0

def cleanup(config):
    #nothing to clean up
    return 0

def make_plots(redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, richness_bins, z_bins, masks):
    #fig, axs = plt.subplots(3, 3)
    fig = plt.figure(figsize=(18, 10))
    for ii in range(3):
            zlo= z_bins[ii]
            zup= z_bins[ii+1]
            ind, = np.where((redmapper_z >= zlo) & (redmapper_z < zup))
            
            ra = coord.Angle(redmapper_ra[ind]*u.degree)
            ra = ra.wrap_at(180*u.degree)
            dec = coord.Angle(redmapper_dec[ind]*u.degree)
            ax = fig.add_subplot(2, 3, (ii+1), projection="mollweide")
            ax.grid(True)
            ax.title.set_text(r'%.2f$ \leq z < %.2f$'%(zlo, zup))
            ax.hexbin(ra.radian, dec.radian, extent = (-np.pi, np.pi, -np.pi*0.5, np.pi*0.5), cmap = 'Blues', bins = 30)
    ax = fig.add_subplot(2, 3, 4)
    ax.hexbin(np.log10(redmapper_lambda), redmapper_z, extent = (1, 3.0, 0.1, 1.0), cmap='Blues', bins =50)
    ax.set_xlabel('log(richness)') 
    ax.set_ylabel('Redshift')

    ax = fig.add_subplot(2, 3, 5)
    plt.axes(ax)
    rec_hp = masks.generate_healpix_map(nside=128, key='zmax')
    hp.mollview(rec_hp, nest = True, title="Zmax", hold=True, flip = 'geo')
    hp.graticule()

    ax = fig.add_subplot(2,3, 6)
    plt.axes(ax)
    rec_hp = masks.generate_healpix_map(nside=128, key='fracgood')
    hp.mollview(rec_hp, nest = True, title="FracGood", hold=True, flip = 'geo')
    hp.graticule()
    plt.savefig('redshift_coverage.png')
    #plt.show()
    return 1
