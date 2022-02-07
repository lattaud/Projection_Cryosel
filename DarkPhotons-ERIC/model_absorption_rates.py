#!/usr/bin/python
###########################
# EA Dec 2019
# Theoretical DM ALP/HP absorption rates for masses lower than EDW3
###########################


import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


###########################
### Subroutines

def data_tables(basedir) :
    # From 1608.01994
    sigma1_hochberg = np.genfromtxt(basedir+"/sigma1_hochberg.txt", names=['omega','sigma'])
    sigma2_hochberg = np.genfromtxt(basedir+"/sigma2_hochberg.txt", names=['omega','sigma'])
    w, = np.where( sigma2_hochberg['sigma'] < 4.1)
    sigma2_hochberg['sigma'][w] *= -1
    omega_min = max( np.min(sigma1_hochberg['omega']), np.min(sigma2_hochberg['omega']))
    omega_max = min( np.max(sigma1_hochberg['omega']), np.max(sigma2_hochberg['omega']))
    omega_eV = np.logspace(np.log10(omega_min),np.log10(omega_max),2000)
    sigma1 = np.interp(omega_eV,sigma1_hochberg['omega'],sigma1_hochberg['sigma'])
    sigma2 = np.interp(omega_eV,sigma2_hochberg['omega'],sigma2_hochberg['sigma'])
    polar_tensor = -1j*omega_eV*(sigma1+1j*sigma2)
    medium_corr = np.array(omega_eV,dtype=[('omega','float64'),('correction','float64')])
    medium_corr['correction'] = omega_eV**4 / ( ( omega_eV**2 - np.real(polar_tensor))**2 + (np.imag(polar_tensor))**2 )
    data_tables = dict(
        sigma1=sigma1_hochberg,
        sigma2=sigma2_hochberg,
        medium_corr=medium_corr)
    return data_tables

def alp_rate(ma_eV, gae, conduct_tab, log_interpol=True) :
    # From (10) in 1608.01994
    # Returns event rate/kg/day
    alpha = 1./137.
    me_eV = 511.e3
    hbar = 1.054572e-34 # J.s
    eV_to_invsec = 1.602e-19/hbar
    day = 3600.*24.
    rho_dm = 0.3e9  # eV/cm3
    rho_Ge = 5.323e-3   # kg/cm3, wikipedia, @ 25 deg C
    e2 = 4*np.pi*alpha
    if log_interpol :
        sigma = 10.**(np.interp(np.log10(ma_eV), np.log10(conduct_tab['omega']), np.log10(conduct_tab['sigma']))) # eV
    else :
        sigma = np.interp(ma_eV, conduct_tab['omega'], conduct_tab['sigma']) # eV
    rate_sec = (1./rho_Ge)*(rho_dm/ma_eV)*3*ma_eV**2 /(e2*4.*me_eV**2)
    rate_sec *= (gae**2 *sigma*eV_to_invsec )
    return rate_sec*day

def hp_rate(mv_eV, kappa, conduct_tab, medium_corr_tab, log_interpol=True) :
    # From (7) in 1608.01994
    # Returns event rate/kg/d
    hbar = 1.054572e-34 # J.s
    eV_to_invsec = 1.602e-19/hbar
    day = 3600.*24.
    rho_dm = 0.3e9  # eV/cm3
    rho_Ge = 5.323e-3   # kg/cm3, wikipedia, @ 25 deg C
    if log_interpol :
        sigma = 10.**(np.interp(np.log10(mv_eV), np.log10(conduct_tab['omega']), np.log10(conduct_tab['sigma']))) # eV
    else :
        sigma = np.interp(mv_eV, conduct_tab['omega'], conduct_tab['sigma']) # eV
    correction = 1
    if mv_eV < np.max(medium_corr_tab['omega']) :
        if log_interpol :
            correction = np.interp(np.log10(mv_eV), np.log10(medium_corr_tab['omega']), medium_corr_tab['correction'])        
        else :
            correction = np.interp(mv_eV, medium_corr_tab['omega'], medium_corr_tab['correction'])
    rate_sec = (1./rho_Ge)*(rho_dm/mv_eV)*correction*kappa**2 *sigma*eV_to_invsec
    return rate_sec*day


###########################
### Script : a) rate calculation

basedir = os.path.join(os.environ['HOME'],"Edelweiss/Recherches_axions_autres/Quentin_axions")
dt = data_tables(basedir)

# Medium correction
plt.clf()
plt.plot(dt['medium_corr']['omega'],dt['medium_corr']['correction'])
plt.xscale('log')

# Conductivity
plt.clf()
plt.plot(dt['sigma1']['omega'], dt['sigma1']['sigma'],c='r',label='Real')
plt.plot(dt['sigma2']['omega'], np.abs(dt['sigma2']['sigma']),c='b',label='|Imag|')
plt.xscale('log')
plt.yscale('log')
plt.legend(frameon=False)

# Rate vs mass
gae_ref = 1.e-12 # order of mag of EDW3 bound
kappa_ref = 1.e-14

mass_eV = np.logspace(0., 4., 200)
R_alp = np.asarray([alp_rate(x, gae_ref, dt['sigma1']) for x in mass_eV])
R_hp = np.asarray([hp_rate(x, kappa_ref, dt['sigma1'],dt['medium_corr']) for x in mass_eV])

plt.clf()
plt.plot(mass_eV, R_alp, c='b',label='gae='+str(gae_ref))
plt.plot(mass_eV, R_hp, c='r',label='kappa='+str(kappa_ref))
plt.scatter(mass_eV, R_alp, c='b')
plt.scatter(mass_eV, R_hp, c='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('m_boson (eV)')
plt.ylabel('Rate  in Ge (/kg/d)')
plt.legend(frameon=False)
np.savetxt(basedir+"/tabulated_hp_rate.txt",np.transpose([mass_eV,R_hp]),header="Rate in Ge from HP absorption with kappa=1e-14. \n mass/eV ; Rate/kg/d", fmt="%.5e")
np.savetxt(basedir+"/tabulated_alp_rate.txt",np.transpose([mass_eV,R_alp]),header="Rate in Ge from ALP absorption with gae=1.e-12. \n mass/eV ; Rate/kg/d", fmt="%.5e")

###########################
### Script : b) derive bounds

## Dirty stuff here = should get clean bound on rate, with correct resolution...
limit_lowE = { 'E_eV' : np.asarray([5,7.5,10,20,35]) , 
                'count_per_0.1eV' : np.asarray([3000, 200, 10, 2, 1])}
limit_highE = {'E_eV' : np.asarray([15, 50, 100, 500, 1000]), 
                'dru' : np.asarray([3.e5, 3.e4, 8.e3, 2.e3, 2.e3])}               
the_kgd = 0.08
limit_lowE['dru'] = limit_lowE['count_per_0.1eV']*1.e4/the_kgd # /keV/kgd
resolution_tab = {'E_eV' : [0,150,1000], 'sigma_eV' : [1.6,10,100]}
for x in [limit_lowE, limit_highE] :
    reso = np.interp(x['E_eV'], resolution_tab['E_eV'], resolution_tab['sigma_eV'])
    x['line/kgd'] = x['dru']*2*reso*1.e-3
mass_eV = np.logspace(np.log10(5), 3., 200)
x = np.concatenate((limit_lowE['E_eV'], limit_highE['E_eV']))
y = np.concatenate((limit_lowE['line/kgd'], limit_highE['line/kgd']))
pp = np.argsort(x)
x = x[pp]
y = y[pp]
linelim_kgd = 10.**(np.interp(np.log10(mass_eV), np.log10(x), np.log10(y)))

plt.clf()
plt.scatter(limit_lowE['E_eV'], limit_lowE['line/kgd'], c='r', label='low E')
plt.scatter(limit_highE['E_eV'], limit_highE['line/kgd'], c='b', label='high_E')
plt.plot(mass_eV, linelim_kgd, c='k')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('E (eV)')
plt.ylabel('Line intensity bound (/kg/d)')
plt.legend(frameon=False)
plt.xlim((4,1200))

## Limits
# Rate vs mass
gae_ref = 1.e-12
kappa_ref = 1.e-14
R_alp = np.asarray([alp_rate(x, gae_ref, dt['sigma1']) for x in mass_eV])
R_hp = np.asarray([hp_rate(x, kappa_ref, dt['sigma1'],dt['medium_corr']) for x in mass_eV])

gae_lim = gae_ref*np.sqrt(linelim_kgd/R_alp)
kappa_lim = kappa_ref*np.sqrt(linelim_kgd/R_hp)

plt.clf()
plt.subplot(1,2,1)
plt.plot(mass_eV, gae_lim, c='b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('m_ALP (eV)')
plt.ylabel('g_ae bound')
plt.subplot(1,2,2)
plt.plot(mass_eV, kappa_lim, c='b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('m_HP (eV)')
plt.ylabel('kappa bound')

