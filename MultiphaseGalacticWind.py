################################################################
# Author: Drummond Fielding
# Reference: Fielding & Bryan (2021)
# Date: 08 Aug 2021
# Brief: This code calculates the structure of multiphase galactic winds.
#
# Execution:
# >> python MultiphaseGalacticWind.py
#
# Output: a 9 panel figure showing the properties of a multiphase galactic wind relative to a single phase galactic wind 
# 
# Overview:
# - First the code calculates the structure of a single phase galactic wind in the manner of Chevalier and Clegg (1985). 
# - Then the code calculates the structure of a multiphase galactic wind. 
# - The default values are:
#   - SFR            = 20 Msun/yr   (star formation rate)
#   - eta_E          = 1            (energy loading)
#   - eta_M          = 0.1          (initial hot phase or single pahse mass loading)
#   - eta_M_cold     = 0.2          (initial cold phase mass loading)
#   - M_cloud_init   = 10^3 Msun    (initial cloud mass)
#   - v_cloud_init   = 10^1.5 km/s  (initial cloud velocity)
#   - r_sonic        = 300 pc       (sonic radius)
#   - Z_wind_init    = 2 * Z_solar  (initial wind metallicity)
#   - Z_cloud_init   = Z_solar      (initial cloud metallicity)
#   - v_circ0        = 150 km/s     (circular velocity of external isothermal gravitational potential)
#
################################################################

import numpy as np
import glob
import h5py 
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import solve_ivp
import cmasher as cmr
from matplotlib.lines import Line2D

## Plot Styling
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = "round"
matplotlib.rcParams['lines.solid_capstyle'] = "round"
matplotlib.rcParams['legend.handletextpad'] = 0.4
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['ytick.major.width'] = 0.6
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.minor.width'] = 0.45
matplotlib.rcParams['xtick.minor.width'] = 0.45
matplotlib.rcParams['ytick.major.size'] = 2.75
matplotlib.rcParams['xtick.major.size'] = 2.75
matplotlib.rcParams['ytick.minor.size'] = 1.75
matplotlib.rcParams['xtick.minor.size'] = 1.75
matplotlib.rcParams['legend.handlelength'] = 2
matplotlib.rcParams["figure.dpi"] = 200
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright}  \usepackage[T1]{fontenc}')

## Defining useful constants
gamma   = 5/3.
kb      = 1.3806488e-16
mp      = 1.67373522381e-24
km      = 1e5
s       = 1
yr      = 3.1536e7
Myr     = 3.1536e13
Gyr     = 3.1536e16
pc      = 3.086e18
kpc     = 1.0e3 * pc
Msun    = 2.e33
mu      = 0.62 
muH     = 1/0.75
Z_solar = 0.02


"""
Cooling curve as a function of density, temperature, metallicity
"""
Cooling_File = "./CoolingTables/z_0.000.hdf5" ### From Wiersma et al. (2009) appropriate for z=0 UVB
f            = h5py.File(Cooling_File, 'r')
i_X_He       = -3 
Metal_free   = f.get('Metal_free')
Total_Metals = f.get('Total_Metals')
log_Tbins    = np.array(np.log10(Metal_free['Temperature_bins']))
log_nHbins   = np.array(np.log10(Metal_free['Hydrogen_density_bins']))
Cooling_Metal_free   = np.array(Metal_free['Net_Cooling'])[i_X_He] ##### what Helium_mass_fraction to use    Total_Metals = f.get('Total_Metals')
Cooling_Total_Metals = np.array(Total_Metals['Net_cooling'])
HHeCooling           = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Metal_free)
ZCooling             = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Total_Metals)
f.close()
Zs          = np.logspace(-2,1,31)
Lambda_tab  = np.array([[[HHeCooling.ev(lT,ln)+Z*ZCooling.ev(lT,ln) for Z in Zs] for lT in log_Tbins] for ln in log_nHbins])
Lambda_z0   = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs), Lambda_tab, bounds_error=False, fill_value=-1e-30)

def tcool_P(T,P, metallicity):
    """
    cooling time function
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    """
    T = np.where(T>10**8.98, 10**8.98, T)
    T = np.where(T<10**2, 10**2, T)
    nH_actual = P/T*(mu/muH)
    nH = np.where(nH_actual>1, 1, nH_actual)
    nH = np.where(nH<10**-8, 10**-8, nH)
    return 1.5 * (muH/mu)**2 * kb * T / ( nH_actual * Lambda_z0((np.log10(nH),np.log10(T), metallicity)))

def Lambda_T_P(T,P, metallicity):
    """
    cooling curve function as a function of
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    above nH = 0.9 * cm**-3 there is no more density dependence 
    """
    nH = P/T*(mu/muH)
    if nH > 0.9:
        nH = 0.9
    return Lambda_z0((np.log10(nH),np.log10(T), metallicity))
Lambda_T_P  = np.vectorize(Lambda_T_P)

def Lambda_P_rho(P, rho, metallicity):
    """
    cooling curve function as a function of
    P in units of erg * cm**-3
    rho in units of g * cm**-3
    metallicity in units of solar metallicity
    above nH = 0.9 * cm**-3 there is no more density dependence 
    """
    nH = rho / (muH * mp)
    T  = P/kb / (rho/(mu*mp))
    if nH > 0.9:
        nH = 0.9
    return Lambda_z0((np.log10(nH),np.log10(T), metallicity))
Lambda_P_rho  = np.vectorize(Lambda_P_rho)

def Multiphase_Wind_Evo(r, state):
    """
    Calculates the derivative of v_wind, rho_wind, Pressure, rhoZ_wind, M_cloud, v_cloud, and Z_cloud. 
    Used with solve_ivp to calculate steady state structure of multiphase wind. 
    """
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4]
    v_cloud    = state[5]
    Z_cloud    = state[6]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    Z_wind       = rhoZ_wind/rho_wind
    vc           = v_circ0 * np.where(r<r_sonic, r/r_sonic, 1.0)

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) 
    Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) 

    # cloud properties
    Ndot_cloud              = Ndot_cloud_init * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud    = Ndot_cloud/(Omwind * v_cloud * r**2)
    cs_cl_sq                = gamma * kb*T_cloud/(mu*mp)

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind              # density contrast
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * M_cloud * v_turb_cold / r_cloud 
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow - Mdot_loss, 0)

    # density
    drhodt       = (number_density_cloud * Mdot_cloud)
    drhodt_plus  = (number_density_cloud * Mdot_loss)
    drhodt_minus = (number_density_cloud * Mdot_grow) 

    # momentum
    p_dot_drag   = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)
    dpdt_drag    = (number_density_cloud * p_dot_drag)
    
    # energy
    e_dot_cool   = 0.0 if (Cooling_Factor==0) else (rho_wind/(muH*mp))**2 * Lambda_P_rho(Pressure,rho_wind,Z_wind/Z_solar)

    # metallicity
    drhoZdt      = -1.0 * (number_density_cloud * (Z_wind*Mdot_grow + Z_cloud*Mdot_loss))

    # wind gradients
    # velocity
    dv_dr       = 2/Mach_sq_wind
    dv_dr      += - (vc/v_wind)**2
    dv_dr      +=  drhodt_minus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    dv_dr      += (gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    dv_dr      += -(gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    dv_dr      += -dpdt_drag/(rho_wind*v_wind**2/r)
    dv_dr      *= (v_wind/r)/(1.0-(1.0/Mach_sq_wind))
    
    # density
    drho_dr       = -2
    drho_dr      += (vc/v_wind)**2
    drho_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    drho_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    drho_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    drho_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    drho_dr      *= (rho_wind/r)/(1.0-(1.0/Mach_sq_wind))

    # pressure
    dP_dr       = -2
    dP_dr      += (vc/v_wind)**2
    dP_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.* (v_rel**2 / cs_sq_wind)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/cs_sq_wind)
    dP_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    dP_dr      *= (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind))


    drhoZ_dr   = drho_dr*(rhoZ_wind/rho_wind) + (rhoZ_wind/r) * drhodt_plus/(rho_wind*v_wind/r) * (Z_cloud/Z_wind - 1)

    # cloud gradients
    dM_cloud_dr = Mdot_cloud/v_cloud

    dv_cloud_dr = (p_dot_drag + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)

    dZ_cloud_dr = (Z_wind-Z_cloud) * Mdot_grow / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)

    return np.r_[dv_dr, drho_dr, dP_dr, drhoZ_dr, dM_cloud_dr, dv_cloud_dr, dZ_cloud_dr]

def Single_Phase_Wind_Evo(r, state):
    """
    Calculates the derivative of v_wind, rho_wind, Pressure for a single phase wind. 
    Used with solve_ivp to calculate steady state structure of a single phase wind with no cooling and no gravity. 
    """
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(r<r_sonic, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(r<r_sonic, 1.0, 0.0)

    # density
    drhodt          = Mdot_SN
    
    # momentum
    dpdt            = 0 

    # energy
    dedt            = Edot_SN

    dv_dr    = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( 2.0/Mach_sq_wind - 1/(rho_wind*v_wind/r) * (drhodt*(gamma+1)/2. + (gamma-1)*dedt/v_wind**2)) 
    drho_dr  = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind)) 
    dP_dr    = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt + drhodt * (gamma-1)/2.*Mach_sq_wind + (gamma-1)*Mach_sq_wind*dedt/v_wind**2))

    return np.r_[dv_dr, drho_dr, dP_dr]

def cloud_ksi(r, state):
    """
    function to calculate the value of ksi = t_mix / t_cool
    """
    v_wind       = state[0]
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    M_cloud      = state[4]
    v_cloud      = state[5]
    Z_cloud      = state[6]
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    return ksi

def supersonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 + epsilon)

supersonic.terminal = True

def subsonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 - epsilon)

subsonic.terminal = True

def cold_wind(r,z):
    return np.sqrt(gamma*z[2]/z[1])/np.sqrt(gamma*kb*T_cloud/(mu*mp)) - (1.0 + epsilon)

cold_wind.terminal = True


def cloud_stop(r,z):
    return z[5] - 10e5

cloud_stop.terminal = True



# Default Parameters
SFR              = 20 * Msun/yr
eta_E            = 1
eta_M            = 0.1
eta_M_cold       = 0.2
log_M_cloud_init = 3

# feedback and SF props
E_SN        = 1e51          ## energy per SN in erg
mstar       = 100*Msun      ## mass of stars formed per SN
M_cloud_min = 1e-2*Msun     ## minimum mass of clouds
Mdot        = eta_M * SFR
Edot        = eta_E * (E_SN/mstar) * SFR

## model choices
CoolingAreaChiPower         =  0.5 
ColdTurbulenceChiPower      = -0.5 
TurbulentVelocityChiPower   =  0.0 
Mdot_coefficient            = 1.0/3.0
Cooling_Factor              = 1.0
drag_coeff                  = 0.5
f_turb0                     = 10**-1.0
v_circ0                     = 150e5 # gravitational potential assuming isothermal potential

r_sonic                 = 300*pc
Z_wind_init             = 2.0 * Z_solar
half_opening_angle      = np.pi/2
Omwind                  = 4*np.pi*(1.0 - np.cos(half_opening_angle))

# properties at r_sonic if no clouds + gravity
epsilon     = 1e-5              ## define a small number to jump above and below sonic radius / mach = 1
Mach0       = 1.0 + epsilon
v0          = np.sqrt(Edot/Mdot)*(1/((gamma-1)*Mach0) + 1/2.)**(-1/2.) ## velocity at sonic radius
rho0        = Mdot/(Omwind*r_sonic**2 * v0)                            ## density at sonic radius
P0          = rho0*v0**2 / Mach0**2 / gamma                            ## pressure at sonic radius
rhoZ0       = rho0 * Z_wind_init
print( "v_wind = %.1e km/s  n_wind = %.1e cm^-3  P_wind = %.1e kb K cm^-3" %(v0/1e5, rho0/(mu*mp), P0/kb))

Edot_per_Vol = Edot / (4/3. * np.pi * r_sonic**3) # source terms from SN
Mdot_per_Vol = Mdot / (4/3. * np.pi * r_sonic**3) # source terms from SN






##########
## integrate the single phase only solution
##########

r_init = 100*pc ### inner radius for hot solution

## calculate gradients right at sonic radius 
dv_dr0, drho_dr0, dP_dr0 = Single_Phase_Wind_Evo(r_sonic, np.r_[v0, rho0, P0])

## interpolate to within the subsonic region
dlogvdlogr   = dv_dr0 * r_sonic/v0
dlogrhodlogr = drho_dr0 * r_sonic/rho0
dlogPdlogr   = dP_dr0 * r_sonic/P0
dlogr0       = 1e-8

v0_sub   = 10**(np.log10(v0) - dlogvdlogr * dlogr0)
rho0_sub = 10**(np.log10(rho0) - dlogrhodlogr * dlogr0)
P0_sub   = 10**(np.log10(P0) - dlogPdlogr * dlogr0)

### integrate (single phase only) from sonic radius to r_init in the subsonic region.
sol = solve_ivp(Single_Phase_Wind_Evo, [10**(np.log10(r_sonic)-dlogr0),r_init], np.r_[v0_sub, rho0_sub, P0_sub], 
    events=[supersonic], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_init    = sol.t[-1]
v_init    = sol.y[0][-1]
rho_init  = sol.y[1][-1]
P_init    = sol.y[2][-1]
rhoZ_init = rho_init * Z_wind_init

## interpolate to within the supersonic region
v0_sup   = 10**(np.log10(v0) + dlogvdlogr * dlogr0)
rho0_sup = 10**(np.log10(rho0) + dlogrhodlogr * dlogr0)
P0_sup   = 10**(np.log10(P0) + dlogPdlogr * dlogr0)

### integrate (single phase only) from sonic radius to 100x sonic radius
sol_sup = solve_ivp(Single_Phase_Wind_Evo, [10**(np.log10(r_sonic)+dlogr0),10**2*r_sonic], np.r_[v0_sup, rho0_sup, P0_sup], 
    events=[supersonic], 
    dense_output=True,
    rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kb])

r_hot_only         = np.append(sol.t[::-1], sol_sup.t)
v_wind_hot_only    = np.append(sol.y[0][::-1], sol_sup.y[0])
rho_wind_hot_only  = np.append(sol.y[1][::-1], sol_sup.y[1])
P_wind_hot_only    = np.append(sol.y[2][::-1], sol_sup.y[2])

Mdot_wind_hot_only   = Omwind*r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only/(Msun/yr)
cs_wind_hot_only     = np.sqrt(gamma * P_wind_hot_only / rho_wind_hot_only)
T_wind_hot_only      = P_wind_hot_only/kb / (rho_wind_hot_only/(mu*mp))
K_wind_hot_only      = (P_wind_hot_only/kb) / (rho_wind_hot_only/(mu*mp))**gamma
Pdot_wind_hot_only   = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only**2/(1e5*Msun/yr)
Edot_wind_hot_only   = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only * (0.5 * v_wind_hot_only**2 + 1.5 * cs_wind_hot_only**2)/(1e5**2*Msun/yr)







##########
## integrate the multiphase solution
##########

# cold cloud initial properties
T_cloud             = 1e4
log_eta_M_cold      = np.log10(eta_M_cold)

#########################
## cold clouds can either be introduced instantaneously at cold_cloud_injection_radial_extent 
## or distributed in space according to some power law slope (cold_cloud_injection_radial_power) between cloud_radial_offest and cold_cloud_injection_radial_extent

## Immediate injection of cold clouds
cold_cloud_injection_radial_power   = np.inf 
cold_cloud_injection_radial_extent  = 1.33*r_sonic
cloud_radial_offest                 = 2e-2 ### don't start integration exactly at r_sonic

## Distributed injection of cold clouds --- uncomment for distributed cloud injection
# cold_cloud_injection_radial_power   = 6
# cold_cloud_injection_radial_extent  = 1.33*r_sonic
# cloud_radial_offest                 = 2e-2

### where to start the integration
irstart         = np.argmin(np.abs(r_hot_only-r_sonic*(1.0+cloud_radial_offest))) 
r_init          = r_hot_only[irstart]
v_init          = v_wind_hot_only[irstart]
rho_init        = rho_wind_hot_only[irstart]
P_init          = P_wind_hot_only[irstart]
M_cloud_init    = 10**log_M_cloud_init * Msun
Z_cloud_init    = 1 * Z_solar 

# cold cloud total properties
v_cloud_init    = 10**1.5 * km/s 
Mdot_cold_init  = eta_M_cold * SFR              ## mass flux in cold clouds
Ndot_cloud_init = Mdot_cold_init / M_cloud_init ## number flux in cold clouds

#### ICs 
supersonic_initial_conditions = np.r_[v_init, rho_init, P_init, Z_wind_init*rho_init, M_cloud_init, v_cloud_init, Z_cloud_init]

### integrate!
sol = solve_ivp(Multiphase_Wind_Evo, [r_init, 1e2*r_sonic], supersonic_initial_conditions, events=[supersonic,cloud_stop,cold_wind], dense_output=True,rtol=1e-10)
print(sol.message)
print(sol.t_events)

## gather solution and manipulate into useful form
r           = sol.t
v_wind      = sol.y[0]
rho_wind    = sol.y[1]
P_wind      = sol.y[2]
rhoZ_wind   = sol.y[3]
M_cloud     = sol.y[4]
v_cloud     = sol.y[5]
Z_cloud     = sol.y[6]

cloud_Mdots = (np.outer(Ndot_cloud_init, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud / (Msun/yr))
Mdot_wind   = Omwind * r**2 * rho_wind * v_wind/(Msun/yr)
cs_wind     = np.sqrt(gamma * P_wind / rho_wind)
T_wind      = P_wind/kb/(rho_wind/(mu*mp))
K_wind      = (P_wind/kb) / (rho_wind/(mu*mp))**gamma

Pdot_wind   = Omwind * r**2 * rho_wind * v_wind**2/(1e5*Msun/yr)
cloud_Pdots = (np.outer(Ndot_cloud_init, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * v_cloud / (1e5 * Msun/yr))

Edot_wind   = Omwind * r**2 * rho_wind * v_wind * (0.5 * v_wind**2 + 1.5 * cs_wind**2)/(1e5**2*Msun/yr)
cloud_Edots = (np.outer(Ndot_cloud_init, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * (0.5 * v_cloud**2 + 2.5 * kb * T_cloud/(mu*mp)) / (1e5**2 * Msun/yr))








##########
## Plot
##########

fig,((axv,axZ,axd),(axM,axX,axP),(axMd,axEd,axK)) = plt.subplots(3,3,sharex=True,constrained_layout=True)
single_phase_color  = 'gray'
cloud_colors        = '#1f77b4'
cs_linestyle        = ':'
cloud_linestyle     = '--'
axv.loglog(r_hot_only/kpc, v_wind_hot_only/1e5,                                    color = single_phase_color )
axv.loglog(r_hot_only/kpc, cs_wind_hot_only/1e5,                                   color = single_phase_color , ls = cs_linestyle)
axd.loglog(r_hot_only/kpc, rho_wind_hot_only/(mu*mp),                              color = single_phase_color )
axP.loglog(r_hot_only/kpc, P_wind_hot_only/kb,                                     color = single_phase_color )
axZ.loglog(r_hot_only/kpc, np.ones_like(rho_wind_hot_only)*Z_wind_init/Z_solar,    color = single_phase_color )
axK.loglog(r_hot_only/kpc, K_wind_hot_only,                                        color = single_phase_color )

axMd.loglog(r_hot_only/kpc, Mdot_wind_hot_only, color = single_phase_color )
axEd.loglog(r_hot_only/kpc, Edot_wind_hot_only, color = single_phase_color )

axv.loglog(r/kpc, v_wind/1e5,                                               color = cloud_colors )
axv.loglog(r/kpc, cs_wind/1e5,                                              color = cloud_colors ,ls = cs_linestyle )
axv.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, v_cloud)/1e5,     color = cloud_colors ,ls = cloud_linestyle )
axd.loglog(r/kpc, rho_wind/(mu*mp),                                         color = cloud_colors )
axP.loglog(r/kpc, P_wind/kb,                                                color = cloud_colors )
axZ.loglog(r/kpc, rhoZ_wind / rho_wind / Z_solar,                           color = cloud_colors )
axZ.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, Z_cloud)/Z_solar, color = cloud_colors ,ls = cloud_linestyle )
axK.loglog(r/kpc, K_wind,                                                   color = cloud_colors )
axM.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)/Msun,    color = cloud_colors )
if sol.status == 1:
    axM.scatter(r[-1]/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)[-1]/Msun,  color = cloud_colors , marker='x', s=8)
axX.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_ksi(r, sol.y)), color = cloud_colors )

axMd.loglog(r/kpc, Mdot_wind,                                                color = cloud_colors )
axMd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Mdots[0]),  color = cloud_colors , ls = cloud_linestyle)


axEd.loglog(r/kpc, Edot_wind,                                                color = cloud_colors )
axEd.loglog(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Edots[0]),  color = cloud_colors , ls = cloud_linestyle)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cs_linestyle),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axv.legend(custom_lines, [r'$v_r$', r'$c_s$', r'$v_{\rm cl}$'],loc='lower center',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axZ.legend(custom_lines, [r'$Z$', r'$Z_{\rm cl}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)


custom_lines = [Line2D([0], [0], color='k', ls = '-'),
                Line2D([0], [0], color='k', ls = cloud_linestyle)]
axMd.legend(custom_lines, [r'${\rm hot}$', r'${\rm cold}$'],loc='best',fontsize=6, frameon=False, handlelength=2.2, labelspacing=0.3)

axv.set_ylabel(r'$v \; [{\rm km/s}]$')
axd.set_ylabel(r'$n \; [{\rm cm}^{-3}]$')
axP.set_ylabel(r'$P \; [{\rm K cm}^{-3}]$')
axZ.set_ylabel(r'$Z \; [Z_\odot]$')
axK.set_ylabel(r'$K \; [{\rm K cm}^{2}]$')
axM.set_ylabel(r'$M_{\rm cl} \; [M_\odot]$')
axX.set_ylabel(r'$\xi = r_{\rm cl} / v_{\rm turb} t_{\rm cool}$')
axX.axhline(1, color='grey', lw=0.5, ls=':')

axMd.set_ylabel(r'$\dot{M} \; [M_\odot/{\rm yr}]$')
axEd.set_ylabel(r'$\dot{E} \; [{\rm km}^2/{\rm s}^2 \; M_\odot/{\rm yr}]$')

axMd.set_xlabel(r'$r\; [{\rm kpc}]$')
axEd.set_xlabel(r'$r\; [{\rm kpc}]$')
axK.set_xlabel(r'$r\; [{\rm kpc}]$')
plt.savefig('Multiphase_Galactic_Wind_Example.pdf',dpi=200,bbox_inches='tight')
plt.clf()



