import numpy as np
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
import matplotlib.pyplot as plt 
import lib_ynzhang
import astropy.io.fits as pyfits
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

def setup(options):
    section = option_section

    # read in redmapper catalog
    redmapper_file = options[section,"redmapper_file"]
    redmapper_ra_col = options[section,"redmapper_racol"] 
    redmapper_dec_col = options[section,"redmapper_deccol"] 
    redmapper_lambda_col = options[section,"redmapper_lambdacol"]
    redmapper_z_col = options[section,"redmapper_zcol"]

    redmapper = pyfits.open(redmapper_file)[1].data
    redmapper_ra = redmapper[redmapper_ra_col]
    redmapper_dec = redmapper[redmapper_dec_col]
    redmapper_lambda = redmapper[redmapper_lambda_col]
    redmapper_z = redmapper[redmapper_z_col]

    # read in comparison catalogs
    ACT_file = options[section,"compare_file"]
    ACT_ra_col = options[section,"compare_racol"]
    ACT_dec_col = options[section,"compare_deccol"]
    ACT_z_col = options[section,"compare_zcol"]
    ACT_mass_col = options[section,"compare_masscol"]

    ACT = pyfits.open(ACT_file)[1].data
    ACT_ra = ACT[ACT_ra_col]
    ACT_dec = ACT[ACT_dec_col]
    ACT_z = ACT[ACT_z_col]
    ACT_mass = ACT[ACT_mass_col]

    # Deciding which comparison object is in the redmapper footprint
    redmapper_maskfile = options[section,"redmapper_maskfile"]
    if options[section,"python_version"]:
        ind_ACT, = lib_ynzhang.footprint_check_python_zmax(redmapper_maskfile, ACT_ra, ACT_dec, ACT_z)
    else:
        ind_ACT, = lib_ynzhang.footprint_check(redmapper_maskfile, ACT_ra, ACT_dec) 
    # need to code up footprint_check, this file depends on the maskfile format

    return redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, ACT_ra, ACT_dec, ACT_z, ACT_mass, ind_ACT

def execute(block, config):
    redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, ACT_ra, ACT_dec, ACT_z, ACT_mass, ind_ACT = config
    
    dA_func= interp1d(block['distances', 'z'], block['distances', 'd_a'])
    dAs=dA_func(redmapper_z)
    ind_ACT_match, ind_redmapper_match, dists = lib_ynzhang.match_ACT_to_redmapper(ACT_ra, ACT_dec, redmapper_ra, redmapper_dec, dAs, ACT_z, redmapper_z)

    print('%i comparison clusters, %i in footprint, %i matched'%(len(ACT_ra), len(ind_ACT), len(ind_ACT_match)))
    make_plots(redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, ACT_ra, ACT_dec, ACT_z, ACT_mass, ind_ACT, ind_ACT_match, ind_redmapper_match, dists)

    # print out nonmatched ACT, but still in the footrpint information
    non_matches_ind = remove_match(ind_ACT, ind_ACT_match)
    for jj in non_matches_ind:
        print("ACT clusters in the footprint but not matched, ra, dec, redshift, mass: ", ACT_ra[jj], ACT_dec[jj], ACT_z[jj], ACT_mass[jj])
    return 0

def cleanup(config):
    #nothing to clean up
    return 0

def remove_match(indices, matched):
    nonmatched = indices
    for ii in matched:
        ind_temp,  = np.where(nonmatched== ii)
        nonmatched = np.delete(nonmatched, ind_temp)
    return nonmatched

def make_plots(redmapper_ra, redmapper_dec, redmapper_lambda, redmapper_z, ACT_ra, ACT_dec, ACT_z, ACT_mass, ind_ACT, ind_ACT_match, ind_redmapper_match, dists):
    gs = gridspec.GridSpec(3, 3)
    axarr0 =  plt.subplot(gs[0, :])
    axarr1 =  plt.subplot(gs[1, :])
    axarr2 =  plt.subplot(gs[2, 0])
    axarr3 =  plt.subplot(gs[2, 1])
    axarr4 =  plt.subplot(gs[2, 2])

    # plot all redmapper clusters
    axarr0.scatter(redmapper_ra, redmapper_dec,  s=1, edgecolors='r', facecolors='r',label='redmapper')
    # plot all ACT clusters
    axarr0.scatter(ACT_ra, ACT_dec, edgecolors='k', s=30, facecolors='none', label='comparison clusters')
    axarr1.scatter(ACT_z, ACT_mass, edgecolors='k', s=30, facecolors='none')

    # plot ACT clusters in the footprint
    axarr0.scatter(ACT_ra[ind_ACT], ACT_dec[ind_ACT], s=30, edgecolors='k', facecolors='k', label='In the footprint')
    axarr1.scatter(ACT_z[ind_ACT], ACT_mass[ind_ACT], edgecolors='k', s=30, facecolors='k')

    # plot ACT_Redmapper matches
    axarr0.scatter(ACT_ra[ind_ACT_match], ACT_dec[ind_ACT_match], s=40, edgecolors='r', facecolors='none', label='redmapper matched')
    axarr1.scatter(ACT_z[ind_ACT_match], ACT_mass[ind_ACT_match], edgecolors='r', s=40, facecolors='none')

    axarr2.scatter(ACT_z[ind_ACT_match], redmapper_z[ind_redmapper_match]- ACT_z[ind_ACT_match], s=30, edgecolors='k', facecolors='k')
    axarr2.scatter(ACT_z[ind_ACT_match],redmapper_z[ind_redmapper_match] - ACT_z[ind_ACT_match], s=40, edgecolors='r', facecolors='none')
    axarr3.scatter(ACT_mass[ind_ACT_match], redmapper_lambda[ind_redmapper_match], s=30, edgecolors='k', facecolors='k')
    axarr3.scatter(ACT_mass[ind_ACT_match], redmapper_lambda[ind_redmapper_match], s=40, edgecolors='r', facecolors='none')
    axarr4.hist(dists, range=[0, 1.0], bins=20, alpha=0.3, color='r', label='ACT BCG to redMaPPer BCG \n'+r'$P(\delta < 0.05)$=%f'%(lib_ynzhang.est_centerfrac(dists)))

    axarr0.legend(loc=10)
    axarr1.set_yscale('log')
    axarr1.legend(loc=1)
    axarr4.legend(loc=1, fontsize=7)
    axarr4.set_xlabel(r'$\delta = R_{sep} [Mpc]$');axarr4.set_ylabel('# of clusters')
    axarr3.set_xlabel(r'ACT Mass');axarr3.set_ylabel(r'$\lambda$')
    axarr3.tick_params(axis='both', which='minor', labelsize=6)
    axarr3.set_xscale('log');axarr3.set_yscale('log')
    axarr2.set_xlabel(r'ACT photo$z$');axarr2.set_ylabel(r'$z_\lambda$ - ACT photo$z$')
    axarr1.set_xlabel(r'Redshift');axarr1.set_ylabel(r'ACT Mass')
    #plt.tight_layout()
    plt.savefig('ED56_comparison.png')

    return 1
