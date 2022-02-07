#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 08:30:48 2020

@author: arnaud
"""
from sys import exit
import numpy as np
from math import sin, cos,pi,erf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy import special
import time


class WIMP_Parameters:
    
    def __init__(self, v0=220.,vearth=232.,vesc=544.,Target="Ge",Exposure=1,WIMPmass=100,CrossSection=1):
        self.v0 = v0
        self.vearth=vearth
        self.vesc=vesc
        self.Target = Target
        self.Exposure = Exposure # always put 1 kg.day default
        self.WIMPmass = WIMPmass
        self.CrossSection = CrossSection
 
    # # v0
    # @property
    # def v0(self): 
    #     return self._v0
    
    # @v0.setter 
    # def v0(self, value): 
    #     self._v0 = value
        
    # # Exposure
    # @property
    # def Exposure(self): 
    #     return self._Exposure
    
    # @Exposure.setter 
    # def Exposure(self, value): 
    #     self._Exposure = value
        
    # # WIMPmass
    # @property
    # def WIMPmass(self): 
    #     return self._WIMPmass
    
    # @WIMPmass.setter 
    # def WIMPmass(self, value): 
    #     self._WIMPmass = value
        
    # # CrossSection
    # @property
    # def CrossSection(self): 
    #     return self._CrossSection
    
    # @CrossSection.setter 
    # def CrossSection(self, value): 
    #     self._CrossSection = value
        
    # # vearth
    # @property
    # def vearth(self): 
    #     return self._vearth
    
    # @vearth.setter 
    # def vearth(self, value): 
    #     self._vearth = value
        
    # # vesc
    # @property
    # def vesc(self): 
    #     return self._vesc
    
    # @vesc.setter 
    # def vesc(self, value): 
    #     self._vesc = value
        
    # @property
    # def Target(self): 
    #     return self._Target
    
    # @Target.setter 
    # def Target(self, value): 
    #     self._Target = value
        
    @property
    def Target_uma(self): 
        return atomicmass(self.Target)
    
    @property
    def E_max(self):
        return E_max(self.WIMPmass,self.vesc,self.vearth,self.Target_uma)
    
    def Information(self):
        print("Detailed information")
        print("v0 = ",self.v0, "km/s")
        print("vesc = ",self.vesc, "km/s")
        print("vearth = ",self.vearth, "km/s")
        print("Target = ",self.Target)
        print("Target_uma = ",self.Target_uma, "uma")
        print("Exposure = ",self.Exposure," kg.days")
        print("WIMPmass = ",self.WIMPmass," GeV/c2")
        print("CrossSection = ",self.CrossSection," pb")

        
def atomicmass(target):
    if((target=="Ne") | (target=="Neon")):
        result=20.1797
    elif((target=="Si") | (target=="Silicium")):
        result=28.0855;
    elif((target=="Ge") | (target=="Germanium")):
        result=72.64
    elif((target=="Xe") | (target=="Xenon")):
        result=131.293
    elif((target=="He") | (target=="Helium")):
        result=4.002602
    elif((target=="Ar") | (target=="Argon")):
        result=39.948
    elif((target=="H") | (target=="Hydrogen")):
        result=1.00794
    elif((target=="O") | (target=="Oxygen")):
        result=15.9994
    elif((target=="Ca") | (target=="Calcium")):
        result=40.078
    elif((target=="W") | (target=="Tungsten")):
        result=183.84
    else:
        print("problem uma unkwnown")
        exit()
    return result


def E_max(M_chi,vesc,vearth,M_T):
    """
    Parameters
    ----------
    M_chi : DM mass in GeV/c2
    vesc : escape velocity in km/s
    vearth : Tearth velocity in 
    M_T : Target Mass in GeV/c2
    
    Returns
    -------
    maximum recoil energy in eV
    """
    masse_nucleon= 0.9315 # GeV/c2
    c = 3e5 # speed of light km/s
    M_chi_ev = M_chi*1e9/(c*c) # DMmass in eV
    M_T_ev = M_T*1e9/(c*c)*masse_nucleon # Target atomic mass in eV
    r = (4.*M_T_ev*M_chi_ev)/((M_T_ev+ M_chi_ev)*(M_T_ev + M_chi_ev))
    Er_max_eV = (r/2.)*M_chi_ev*((vesc+vearth)*(vesc+vearth)) # 
    return Er_max_eV

def conversion_factor(M_chi,A):
    """
    Parameters
    ----------
    M_chi : DM mass in GeV/c2
    A : Target in uma

    Returns
    -------
    returns a coefficient to convert the nuclear cross-section into nucleon cross-section
    """
    masse_nucleon= 0.9315 # converts 1 uma into GeV/c2 (1 uma = 0.9315 GeV/c2)
    m_p = masse_nucleon # proton mass in GeV/c^2
    m_T = A * masse_nucleon # Target nuclear mass in GeV/c^2
    mu_p = (M_chi*m_p)/(M_chi+m_p)
    mu_T = (M_chi*m_T)/(M_chi+m_T)                                           
    C_p_Ne_over_C_p =pow(A,2)
    conv_factor= (pow(mu_T,2)/pow(mu_p,2))*(C_p_Ne_over_C_p) 
    return conv_factor

def formfactor(ER,A):
    """
    Parameters
    ----------
    ER : Recoil energy in eV
    A : Target in uma

    Returns
    -------
    Nuclear Form factor, dimensionless
    """
    
    ER = np.asarray(ER,dtype=float)
    #assert np.all(ER>0)
    #assert np.all(ER>=0) 
    masse_nucleon= 0.9315 # //GeV/c2
    ER=ER*1e-9 # conversion from eV to GeV
    with np.errstate(divide='ignore',invalid='ignore'):
        q=np.sqrt(2.*A*masse_nucleon*ER) # in GeV/c
        ss=0.9*1e-15 # in m	
        aa=0.52*1e-15 # in m 
        cc=(1.23*np.power(A,1./3.)-0.6)*1e-15 # in m
        rn=np.sqrt(np.power(cc,2)+(7/3.)*np.power(np.pi,2)*np.power(aa,2)-5.*np.power(ss,2)) # Lewin (4.11)
        hbarc=0.1973*1e-15 # in GeV.m
        qrn=q*rn/hbarc # dimensionless
        formfactor = 3*np.exp(-np.power(q*ss/hbarc,2)/2.)*(np.sin(qrn)-qrn*np.cos(qrn))/(np.power(qrn,3)) # # Lewin (4.7)
    formfactor = formfactor * formfactor
    formfactor = np.where(ER==0,1.,formfactor) # if ER=0 set formfactor to 1.0
    return formfactor

def Calculdrder(ER,*args):
    c = 3e5 # speed of light km/s
    masse_nucleon= 0.9315 # GeV/c2
    par = list(args)
    M_chi = par[0]
    sigma_0 = par[1] # crosssection in pb
    A = par[2] # Target mass in uma
    M_T=par[2] * masse_nucleon; # Target Mass in GeV/c2
    exposure = par[3] # kg.days
    realv0 = par[4]
    v0 = par[4]/c
    vearth=par[5]/c
    vesc=par[6]/c


    y = vearth/v0
    z=vesc/v0
    r=(M_T*M_chi)/(M_T + M_chi)
    r_chi= 4.*(M_T*M_chi)/((M_T + M_chi)*(M_T + M_chi)); # dimensionless
    E0=0.5*M_chi*(v0*v0)*1e9 # in ev
    quantA=np.sqrt(M_T/(2.*r*r)) 
    # quantA=sqrt(2./(M_chi*r_chi))  identical to the above line actually 
    k1=1./(erf(z) - 2.0*z*np.exp(-z*z)/np.sqrt(np.pi))
    extra=k1*np.exp(-np.power(z,2)) # due to vesc!=0
    kcst=(np.sqrt(np.pi)/4.0)*(1./y)
    rho=0.3 # local DM density GeV/cm3/c2
    v0_cmd=realv0*1e5*24*3600. # in cm/d  
    eff = 1.0 # use the full efficiency instead of the 0.36
    R0 = (2./np.sqrt(np.pi))*(6.0223e23/(A*0.001))*(rho/M_chi)*1e-36*v0_cmd*sigma_0*conversion_factor(M_chi,A)*exposure# *eff  in events (2)
    ER_gev=ER*1e-9;   
    formfact= formfactor(ER,A)
    vminratio=(quantA*np.sqrt(ER_gev))/v0
    
    corrected = True
    # (corrected = False) means Lewin & Smith initial formula
    if(corrected):
        if( (vminratio*v0) < (vesc-vearth) ):
            dRdER = (k1*kcst*(erf(vminratio+y) - erf(vminratio-y)) - extra)*R0*eff
        elif(np.logical_and( (vminratio*v0)>=(vesc-vearth), (vminratio*v0)<=(vesc+vearth) ) ):
            dRdER = (k1*kcst*(erf(z) - erf(vminratio-y)) - extra*( (vesc+vearth-vminratio*v0)  / (2*vearth)))*R0*eff 
        else:
            dRdER=0 
    else:
        dRdER = (k1*kcst*(erf(vminratio+y) - erf(vminratio-y)) - extra)*R0*eff
    
    dRdER=dRdER*(formfact/(E0*r_chi))
    
    if (dRdER<0):
        dRdER = 0 
    return dRdER 

def Calculdrder_arrayultimate(ER,W):
    # ER in eV
    # returns rate in events/eV
    ER = np.asarray(ER,dtype=float)
    finalresult = np.zeros_like(ER,dtype=float)
    maskpositive = (ER>0)
    maskpositive_IDS = np.where(maskpositive)[0]
    
    ER = ER[maskpositive] # or ER[maskpositive] (slicing or giving IDs gives same result
    #ER = ER[maskpositive]
    dRdER = np.zeros_like(ER) # initiliaze result matrix to 0 with same shape than ER arra
    
    c = 3e5 # speed of light km/s
    masse_nucleon= 0.9315 # GeV/c2
    
    M_chi = W.WIMPmass
    sigma_0 = W.CrossSection # crosssection in pb
    A =  W.Target_uma # Target mass in uma
    M_T= A * masse_nucleon; # Target Mass in GeV/c2
    exposure = W.Exposure # kg.days
    realv0 = W.v0
    v0 = realv0 / c
    vearth = W.vearth /c
    vesc = W.vesc /c

    y = vearth/v0
    z=vesc/v0
    r=(M_T*M_chi)/(M_T + M_chi)
    r_chi= 4.*(M_T*M_chi)/((M_T + M_chi)*(M_T + M_chi)); # dimensionless
    E0=0.5*M_chi*(v0*v0)*1e9 # in ev
    quantA=np.sqrt(M_T/(2.*r*r)) 
    # quantA=sqrt(2./(M_chi*r_chi))  identical to the above line actually 
    k1=1./(erf(z) - 2.0*z*np.exp(-z*z)/np.sqrt(np.pi))
    extra=k1*np.exp(-np.power(z,2)) # due to vesc!=0
    kcst=(np.sqrt(np.pi)/4.0)*(1./y)
    rho=0.3 # local DM density GeV/cm3/c2
    v0_cmd=realv0*1e5*24*3600. # in cm/d  
    eff = 1.0 # use the full efficiency instead of the 0.36
    R0 = (2./np.sqrt(np.pi))*(6.0223e23/(A*0.001))*(rho/M_chi)*1e-36*v0_cmd*sigma_0*conversion_factor(M_chi,A)*exposure# *eff  in events (2)
    ER_gev=ER*1e-9;   
    formfact= formfactor(ER,A)
    vminratio=(quantA*np.sqrt(ER_gev))/v0
    
    corrected = True
    # (corrected = False) means Lewin & Smith initial formula
    if(corrected):
        # if
        with np.errstate(divide='ignore',invalid='ignore'):
            mask1 = ((vminratio*v0) < (vesc-vearth))
            res1 = (k1*kcst*(special.erf(vminratio+y) - special.erf(vminratio-y)) - extra)*R0*eff
            dRdER = np.where(mask1, res1,dRdER)
            # elif
            mask2 = np.logical_and( (vminratio*v0)>=(vesc-vearth), (vminratio*v0)<=(vesc+vearth) )
            res2 = (k1*kcst*(special.erf(z) - special.erf(vminratio-y)) - extra*( (vesc+vearth-vminratio*v0)  / (2*vearth)))*R0*eff 
            dRdER = np.where(mask2, res2,dRdER)
            # else
            mask3 = ~(mask1|mask2)        
            dRdER[mask3] = 0

    else:
        dRdER = (k1*kcst*(special.erf(vminratio+y) - special.erf(vminratio-y)) - extra)*R0*eff
    
    dRdER=dRdER*(formfact/(E0*r_chi))
   
    
    
    #with np.errstate(divide='ignore',invalid='ignore'):
    dRdER=np.where(dRdER<0,0,dRdER)
    finalresult[maskpositive_IDS] = dRdER
    
    return finalresult

def Calculdrder_array(ER,*args):
    
    ER = np.asarray(ER,dtype=float)
    dRdER = np.zeros_like(ER) # initiliaze result matrix to 0 with same shape than ER arra

    c = 3e5 # speed of light km/s
    masse_nucleon= 0.9315 # GeV/c2
    par = list(args)
    M_chi = par[0]
    sigma_0 = par[1] # crosssection in pb
    A = par[2] # Target mass in uma
    M_T=par[2] * masse_nucleon; # Target Mass in GeV/c2
    exposure = par[3] # kg.days
    realv0 = par[4]
    v0 = par[4]/c
    vearth=par[5]/c
    vesc=par[6]/c


    y = vearth/v0
    z=vesc/v0
    r=(M_T*M_chi)/(M_T + M_chi)
    r_chi= 4.*(M_T*M_chi)/((M_T + M_chi)*(M_T + M_chi)); # dimensionless
    E0=0.5*M_chi*(v0*v0)*1e9 # in ev
    quantA=np.sqrt(M_T/(2.*r*r)) 
    # quantA=sqrt(2./(M_chi*r_chi))  identical to the above line actually 
    k1=1./(erf(z) - 2.0*z*np.exp(-z*z)/np.sqrt(np.pi))
    extra=k1*np.exp(-np.power(z,2)) # due to vesc!=0
    kcst=(np.sqrt(np.pi)/4.0)*(1./y)
    rho=0.3 # local DM density GeV/cm3/c2
    v0_cmd=realv0*1e5*24*3600. # in cm/d  
    eff = 1.0 # use the full efficiency instead of the 0.36
    R0 = (2./np.sqrt(np.pi))*(6.0223e23/(A*0.001))*(rho/M_chi)*1e-36*v0_cmd*sigma_0*conversion_factor(M_chi,A)*exposure# *eff  in events (2)
    ER_gev=ER*1e-9;   
    formfact= formfactor(ER,A)
    vminratio=(quantA*np.sqrt(ER_gev))/v0
    
    corrected = True
    # (corrected = False) means Lewin & Smith initial formula
    if(corrected):
        # if
        mask1 = ((vminratio*v0) < (vesc-vearth))
        res1 = (k1*kcst*(special.erf(vminratio+y) - special.erf(vminratio-y)) - extra)*R0*eff
        dRdER = np.where(mask1, res1,dRdER)
        # elif
        mask2 = np.logical_and( vminratio*v0>=(vesc-vearth), vminratio*v0<=(vesc+vearth) )
        res2 = (k1*kcst*(special.erf(z) - special.erf(vminratio-y)) - extra*( (vesc+vearth-vminratio*v0)  / (2*vearth)))*R0*eff 
        dRdER = np.where(mask2, res2,dRdER)
        # else
        mask3 = ~(mask1|mask2)        
        dRdER[mask3] = 0

    else:
        dRdER = (k1*kcst*(special.erf(vminratio+y) - special.erf(vminratio-y)) - extra)*R0*eff
    
    dRdER=dRdER*(formfact/(E0*r_chi))
  
    dRdER=np.where(dRdER<0,0,dRdER)
    dRdER=np.where(ER<0,0,dRdER)
    return dRdER 





