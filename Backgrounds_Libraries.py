#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def conversion_from_scale(scale):
    assert scale in ['eV','keV']
    if scale=="keV":
        conv = 1
    elif scale=="eV":
        conv = 1000
    else:
        print("ERROR ! scale does not exist")
    return conv

def EDWfuncCOMPTON(Er,scale = "keV",norm=1):
    Er = np.array(Er,dtype=float)
    conv = conversion_from_scale(scale)   
    p0 = 0.1 / conv # dru 
    result =  norm * np.ones_like(Er) * p0
    return result

def EDWfuncTRITIUM(Er,scale = "keV",norm=1):
    Er = np.array(Er,dtype=float)
    conv = conversion_from_scale(scale)   
    p0 = 1.406e-8 / np.power(conv,4) / conv # dru / keV^4
    p1 = 18.6 * conv # 18.6 keV
    p2 = 511 * conv # 511 keV
    result =  norm * (Er<=p1) * p0 * np.power(p1-Er,2)* (p2+Er) * np.sqrt( np.power(Er,2)+2*p2*Er) 
    return result

def EDWfuncBETA(Er,scale = "keV",norm=1):
    Er = np.array(Er,dtype=float)
    conv = conversion_from_scale(scale)  
    p0=  1.34  / conv # dru
    p1= -0.058 / conv # keV−1
    p2= 0.2    / conv # dru
    p3= 40     * conv # keV
    p4= 11.4   * conv # keV.
    result =  p0 * np.exp(p1*Er) + p2*np.exp(-np.power(Er-p3,2)/(2*np.power(p4,2)))
    return result

def EDWfuncNEUTRON(Er,scale = "keV",norm=1):
    Er = np.array(Er,dtype=float)
    conv = conversion_from_scale(scale)  
    p0=  4.827*1e-4 / conv # dru
    p1=  0.3906     / conv # keV−1
    p2=  2.986*1e-4 / conv # dru
    p3=  0.05549    / conv # keV-1
    result =  p0 * np.exp(-p1*Er) + p2*np.exp(-p3*Er)
    return result

def EDWfuncLEAD(Er,scale = "keV",norm=1):
    Er = np.array(Er,dtype=float)
    conv = conversion_from_scale(scale)  
    p0 =  0.037 / conv # dru
    p1 =  0.15  / conv # dru
    p2 =  95    * conv # keV
    p3 =  5.7   * conv # keV
    result = p0 * np.ones_like(Er) + p1*np.exp(-np.power(Er-p2,2)/(2*np.power(p3,2)))
    return result

def EDWfuncHEATONLY(Er,scale = "keV",norm=1):
    Er = np.array(Er,dtype=float)
    conv = conversion_from_scale(scale)  
    p0 =  38.2725 / conv # dru
    p1 =  0.293   / conv # keV-1
    p2 =  1.4775  / conv # dru
    p3 =  0.0812  / conv # keV-1
    result = p0 * np.exp(-p1*Er) + p2 * np.exp(-p3*Er)
    return result