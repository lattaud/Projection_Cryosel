#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:53:30 2020

@author: arnaud
"""

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import time
import random
import matplotlib.animation as animation
import copy
from matplotlib.widgets import Slider, RadioButtons, TextBox
import matplotlib.ticker as ticker
from scipy import interpolate
#import matplotlib.colors as mcolors
from Tools_Libraries import *
from IonModel_Libraries import *
from LimitSetting_Libraries import *
from WIMP_Libraries import *
from DP_Libraries import *
from DMES_Libraries import *
from Backgrounds_Libraries import *

from functools import partial

RecoilType_list=["monoER","sumMonoER","HO","ER","NR","BETA","LEAD","NEUTRON","COMPTON"]

def Nuclear_Ionization_Yield(Er,Quenching="EDW",energyscale="eV"):
    #assumes Target is "Ge"
    Er = np.asarray(Er,dtype=float)
    assert energyscale in ["eV","keV"]
    if(energyscale == "eV"):
        Er = Er / 1000. # converts eV into keV
 
    if(Quenching == "EDW"):
        return 0.16*np.power(Er,0.18)
    
    elif(Quenching == "Lindhard_pure"): 
        # arxiv 1608.03588
        k = 0.157
        Qmin = 0.0001 # arbitrary choice Quentin
        Z = 32  # Z = 32 for Ge 
        eps = 11.5 * np.power(Z,-7/3.) * Er
        g = 3.*np.power(eps,0.15) + 0.7*np.power(eps,0.6) + eps
        Q = (k*g)/(1.+k*g)
        #F = 1. - np.exp(-Er/0.16) # correction factor model
        result = Q
        result = np.where(result<Qmin,Qmin,result)
        return result
    elif(Quenching == "Lindhard_corrected"): 
        # arxiv 1608.03588
        k= 0.1789 #0.1789 for Ge
        Qmin=0.01 # arbitrary choice Quentin
        Z = 32  # Z = 32 for Ge 
        AESF = 0.16 # adiabatic_energy_scale_factor 0.16 for Ge
        
        eps = 11.5 * np.power(Z,-7/3.) * Er
        g = 3.*np.power(eps,0.15) + 0.7* np.power(eps,0.6) + eps
        Q = (k*g)/(1.+k*g)
        F = 1. - np.exp(-Er/0.16) # correction factor model
        result = Q * F
        result = np.where(result<Qmin,Qmin,result)
        return result
    else:
        print("Problem quenching")
        
def Ionization_Yield(Er,recoiltype="ER",energyscale="eV",Quenching="EDW"):
    Er = np.asarray(Er,dtype=float)
    assert energyscale in ["eV","keV"]

    if recoiltype == "BETA":
        return np.ones_like(Er) * 0.4
    
    elif recoiltype == "LEAD":
        return np.ones_like(Er) * 0.08
    
    elif recoiltype == "HO":
        return np.ones_like(Er) * 0
    
    elif recoiltype in ["monoER","sumMonoER","ER","COSMOGENIC","TRITIUM","COMPTON"]:
        return np.ones_like(Er) * 1 
    
    elif recoiltype in ["NR","NEUTRON"]:
        return Nuclear_Ionization_Yield(Er,Quenching,energyscale)
    
    else:
        print(f"We've got a problem, {recoiltype} is not implemented in Ionization_Yield function")

def comparequenching():
    x=np.logspace(-1,4,1000,dtype=float)
    energyscale = "eV"
    qEDW = Nuclear_Ionization_Yield(x,"EDW",energyscale)*100.
    qL = Nuclear_Ionization_Yield(x,"Lindhard_pure",energyscale)*100.
    qLC = Nuclear_Ionization_Yield(x,"Lindhard_corrected",energyscale)*100.
    
    plt.plot(x,qEDW,label="EDW")
    plt.plot(x,qL,label="Lindhard_pure")
    plt.plot(x,qLC,label="Lindhard_corrected")
    plt.legend()
    plt.xscale('log')
    plt.xlabel(energyscale)
    plt.ylabel('Ionization Yield')
    plt.grid() 

class Experiment:
    def __init__(self, Voltage=0,SigmaeV=10,Sigmapercent=0,Fano=1,IonModel="CDMS",Target="Ge",Exposure=1,Quenching="EDW"):
        self.Voltage = Voltage
        self.SigmaeV = SigmaeV
        self.SigmaeVpercent = 0
        self.Fano = Fano
        self.IonModel = IonModel
        self.Target = Target
        self.Exposure = Exposure
        self.Quenching = Quenching
        self.Quantization = True
        self.TriggerNsigma = -100
        self._EfficiencyCurveImport = None
        self._EfficiencyCurveInterpolator = None
        
        
    def ImportEfficiencyCurve(self,file,EnergyScale="eVee"):
        #Efficiency curve given in keVee and we will convert it into eV
        assert EnergyScale in ["eV","eVee","keV","keVee"]
        Energy,Eff = np.loadtxt(file,dtype=float,usecols=(0,1),unpack=True)
           
        if EnergyScale in ["keV","keVee"]:
            Energy = Energy*1000 # to convert keV into eV or keVee into eVee
            # need to think about how to deal with Ein in eVee
        if EnergyScale in ["eVee","keVee"]:
            Epsilon = self.Epsilon
            V = self.Voltage
            Energy = Energy*(1.+V/Epsilon) 
        #now efficiency curve necessarily in eV
        self._EfficiencyCurveInterpolator = interpolate.interp1d(Energy,Eff,kind='cubic',fill_value=(0,Eff[-1]),bounds_error=False,assume_sorted=True)
        self._EfficiencyCurveImport = True
        return self.GetEfficiencyCurve()
    
    def GetEfficiencyCurve(self):
        return self._EfficiencyCurveInterpolator

    def SetEfficiencyCurve(self,effcurve):
        self._EfficiencyCurveInterpolator = effcurve
        
    @property
    def Epsilon(self): 
        return self._Epsilon
    
    @property
    def Gap(self): 
        return self._Gap
    
    @property
    def Target(self): 
        return self._Target
    
    @Target.setter
    def Target(self, value):
        self._Target = value
        self._Epsilon,self._Gap = get_values_from_Target(self._Target)
        
    @property
    def SigmaeVee(self): 
        return self.SigmaeV/(1.+self.Voltage/self._Epsilon)
    
    @property
    def Target_uma(self): 
        return atomicmass(self.Target)
    
    #On ne peut faire une fonction de resolution energie dépendant qui tienne compte de Fano dans la classe Experiment pour la raison suivante :
    #La résolution pour une énergie donnée et un Fano fixé dépend du nombre de paires créées et donc du type de recul
    #Cette fonction de résolution doit donc être appelée dans la classe Spectrum car le type de recul y est déclaré à l'initialisation de la classe 

        
    def Information(self):
        print("Detailed information about Experiment")
        print("Voltage = ",self.Voltage, "V")
        print("Sigma = ",self.SigmaeV, "eV")
        print("Sigmapercent = ",self.SigmaeVpercent, "%   pas encore utilisé")
        print("Fano = ",self.Fano)
        print("IonModel = ",self.IonModel," options are [CDMS,Rouven]")
        print("Target = ",self._Target, "options are [Ge,Si]")
        print("Epsilon = ",self.Epsilon)
        print("Gap = ",self.Gap)
        print("Exposure = ",self.Exposure," kg.days")
        print("Quenching = ",self.Quenching," options are [EDW,Lindhard_pure,Lindhard_corrected]")
        print("Target_uma = ",self.Target_uma, "uma")
        

        
class Spectrum:
    def __init__(self,Stype="ER",Intel=Experiment):
        self.Stype = Stype
        self.RecoilType = Stype
        self.Nptx = 1000
        #self.EnergyScale = "eV"
        self.Intel = Intel
        #self.Func = lambda x : np.exp(-0.1*x)
        self.Emin = 1e-3
        self.Emax = 10000
        self.Line = False
        self.LineEnergy = None
        self.Lines = False
        self.LineRate = None
        self.LinesMass = None
        self.LineEnergies = None
        self.LineRates = None


        if("WIMP" in str(type(self.Stype))): 
            W = self.Stype
            self.RecoilType = "NR"
            # Target_of_Experiment = self.Intel.Target
            # Target_of_WIMPs = W.Target
            # if Target_of_WIMPs!=Target_of_Experiment:
            #     W.Target = Target_of_Experiment
            #     print("Target for WIMP sepctrum has changed from ",Target_of_WIMPs," to ",W.Target)
            TF1 = lambda x : Calculdrder_arrayultimate(x,W) 
            self.SetFunc(TF1,1e-3,W.E_max)
            
        if("DP" in str(type(self.Stype))): 
            D = self.Stype
            self.RecoilType = "monoER"
            self.SetLine(D.GetDPmass(),D.GetDPrate())
            
        if("ALP" in str(type(self.Stype))): 
            ALP = self.Stype
            self.RecoilType = "monoER"
            self.SetLine(ALP.GetALPmass(),ALP.GetALPrate())
        if("DMES" in str(type(self.Stype))): 
            DMES = self.Stype
            self.RecoilType = "sumMonoER"
            energies, rates = DMES.GetDMLinesAndRates()
            self.SetLines(DMES.GetDMmass(),energies,rates)
                       
        if(self.RecoilType in ["NEUTRON","NR"]):
            self.MakeTGraphs(self.Intel)
        
    def SetLine(self,mono_Er=160,rate_per_Exposure=0.1):
        assert self.RecoilType in ["monoER"]
        self.Line = True
        self.LineEnergy = mono_Er
        self.LineRate_per_Exposure = rate_per_Exposure
        self.LineRate = self.Intel.Exposure * rate_per_Exposure
        
    def SetLines(self, massDMES = 1.05, monoEr = np.array([1,2,3]) ,rate_per_Exposure = np.array([0.1,0.2,0.3])):
        assert self.RecoilType in ["sumMonoER"]
        self.Lines = True
        self.LinesMass = massDMES
        self.LineEnergies = monoEr
        self.LineRate_per_Exposure = rate_per_Exposure
        self.LineRates = self.Intel.Exposure * rate_per_Exposure
                                 
    def SetFunc(self, TF1,Emin=0,Emax=100):
        assert self.RecoilType not in ["monoER"]
        self._Func = lambda x: TF1(x) * (x>=Emin) * (x<=Emax) * self.Intel.Exposure
        self.Emin = Emin
        self.Emax = Emax
        #self.SetRange(Emin,Emax)
        
    def SetRange(self,Emin=0,Emax=100):
        self.Emin = Emin
        self.Emax = Emax
        
    def Integral(self,TF1=None,Emin=None,Emax=None,Nptx=1000):  
        if Emin is None:
            Emin = self.Emin 
        if Emax is None:
            Emax = self.Emax
        if TF1 is None:
            TF1 = self.GetFunc
        return TF1Integral(TF1,Emin,Emax,Nptx)
    

    def GetMuFromEph(self,Eph):
        """ L'idée est la suivante : 
            Eph = Er (1+QV/eps)   -> mu = Q * Er / eps 
            Eph = eps * mu/Q + mu *V = mu (eps/Q + V)
            mu = Eph /(eps/Q + V)
            pour Q = 1   mu = Eph/(eps+V)
            return Eph/(V+Epsilon)
        """
        assert np.all(Eph>=0)
        assert self.RecoilType in RecoilType_list
        Epsilon = self.Intel.Epsilon
        V = self.Intel.Voltage
        Eph = np.asarray(Eph,dtype=float)
        
        
        if (self.RecoilType in ["NR","NEUTRON"]): 
            return self._TGraphMuFromEph(Eph)
        elif (self.RecoilType == "HO"):
            return np.zeros_like(Eph)
        elif (self.RecoilType == "ER"): 
            return Eph/(V+Epsilon)
        elif (self.RecoilType in ["BETA","LEAD","COSMOGENIC","TRITIUM","COMPTON"]): 
            return Eph/(V+Epsilon/Ionization_Yield(Er=0,recoiltype=self.RecoilType ,energyscale="eV"))
        else:
            print(f"problem, {self.RecoilType} is not implemented in GetMuFromEph")
            
    def GetErFromEph(self,Eph,energyscale="eV"):
        #assert np.all(Eph>=0)
        assert self.RecoilType in RecoilType_list
        assert energyscale in ["eV","eVee"]
        
        Epsilon = self.Intel.Epsilon
        V = self.Intel.Voltage
        Eph = np.array(Eph,dtype=float)
        
        result = np.zeros_like(Eph)
        mask = (Eph>0)
        
        if (self.RecoilType in ["NR","NEUTRON"]): 
            result[mask] = self._TGraphErFromEph(Eph[mask])
        elif (self.RecoilType == "HO"):
            result[mask] = Eph[mask]
        elif (self.RecoilType == "ER"): 
            result[mask] = Eph[mask] / (1+V/Epsilon)
        elif (self.RecoilType in ["BETA","LEAD","COSMOGENIC","TRITIUM","COMPTON"]): 
            Q = Ionization_Yield(Er=0,recoiltype=self.RecoilType)
            result[mask] = Eph[mask] / (1 + Q*V/Epsilon)
        else:
            print(f"problem, {self.RecoilType} is not implemented in GetMuFromEph")
        if energyscale=="eVee":
            result = result *(1+V/Epsilon)
        return result
    
    def MakeTGraphs(self,Intel=Experiment,Ermin=0,Ermax=300000,npts=300000):
        ErType = self.RecoilType
        assert ErType in RecoilType_list
        Epsilon = Intel.Epsilon
        V = Intel.Voltage
        if (ErType in ["NR","NEUTRON"]):
            Er = np.linspace(Ermin,Ermax,npts)
            Q = Nuclear_Ionization_Yield(Er,Intel.Quenching,energyscale="eV")
            Eph = Er * (np.ones_like(Er,dtype=float)+Q*V/Epsilon)
            mu = Er * Q / Epsilon
            h_MuFromEph = interp1d(Eph, mu)
            h_ErFromEph = interp1d(Eph, Er)
        else:
            print("problem")
        self._TGraphMuFromEph = h_MuFromEph
        self._TGraphErFromEph = h_ErFromEph
        
    def GetFunc(self):
        return self._Func
 
    def GetEphononFunc(self,EnergyScale="eV"):
        TF1GetMuFromEph = lambda xx : self.GetMuFromEph(xx) # toujours mu from eV, pas eVee
        args = self._Func,self.Intel,EnergyScale,self.RecoilType,TF1GetMuFromEph
        return lambda Eph : Eph_from_Er_spectrum(Eph,args)
    
    def GetEphononSmearedLine(self,EnergyScale="eV",quantized=None):
        assert (self.Line == True)
        args = self.LineRate, self.LineEnergy, self.Intel,EnergyScale,self.RecoilType
        
        if quantized == None:
            quantized = self.Intel.Quantization
            
        if quantized == True:
            if EnergyScale=="eV":
                sigmareso = self.Intel.SigmaeV
            elif EnergyScale=="eVee":
                sigmareso = self.Intel.SigmaeVee
            else:
                print("Problem with EnergyScale {}".format(EnergyScale))
            trigger_threshold = self.Intel.TriggerNsigma*sigmareso
            trigger_error_function = lambda x :  0.5 * (special.erf((x-trigger_threshold)/(np.sqrt(2)*sigmareso)) + 1)
            lambdafuncsmeared = lambda Ephs : Ephsmeared_from_monoline_Er_spectrum(Ephs,args) * trigger_error_function(Ephs)
            return lambdafuncsmeared
        else:
            if EnergyScale=="eV":
                sigmareso = self.Intel.SigmaeV
            elif EnergyScale=="eVee":
                sigmareso = self.Intel.SigmaeVee
            else:
                print("Problem with EnergyScale {}".format(EnergyScale))
            trigger_threshold = self.Intel.TriggerNsigma*sigmareso
            trigger_error_function = lambda x :  0.5 * (special.erf((x-trigger_threshold)/(np.sqrt(2)*sigmareso)) + 1)
            lambdafuncsmeared = lambda Ephs : self.ultimate_smeared1D_from_monoline_Er_spectrum(Ephs,args) * trigger_error_function(Ephs)
            return lambdafuncsmeared 
            print("not implemented yet")

    
        
    def GetEphononSmearedLines(self,EnergyScale="eV",quantized=None): 
            # sum_func_var = np.zeros((500, self.Nptx), dtype=np.float64)
            
            def trigger_error_function(x):
                trigger_threshold=self.Intel.TriggerNsigma*self.Intel.SigmaeV
                sigmareso=self.Intel.SigmaeV
                return 0.5 * (special.erf((x-trigger_threshold)/(np.sqrt(2)*sigmareso)) + 1)
                
            def get_func_smeared(Ephs, args):
                
                res = Ephsmeared_from_monoline_Er_spectrum(Ephs,args) * trigger_error_function(Ephs)
                return res
                
            def sum_func(x, func_list):
                
                
                for i, func in enumerate(func_list) : 
                        if i == 0:
                            func_tot = func(Ephs=x)
                        else:
                            func_tot += func(Ephs=x)
        
                return func_tot
            
            assert (self.Lines == True)
            argsLines = self.LineRates, self.LineEnergies, self.Intel,EnergyScale,self.RecoilType
            
            if quantized == None:
                quantized = self.Intel.Quantization
            
            ListFunctionSmeared = list()
            if quantized == True:
                if EnergyScale=="eV":
                    sigmareso = self.Intel.SigmaeV
                elif EnergyScale=="eVee":
                    sigmareso = self.Intel.SigmaeVee
                else:
                    print("Problem with EnergyScale {}".format(EnergyScale))

                for index, Energy in enumerate(self.LineEnergies):
                #for index in range(0,200):
                    args = self.LineRates[index], Energy, self.Intel,EnergyScale,self.RecoilType
                    #print("testing "+str(index))
                    trigger_threshold = self.Intel.TriggerNsigma*sigmareso
                    # trigger_error_function = lambda x :  0.5 * (special.erf((x-trigger_threshold)/(np.sqrt(2)*sigmareso)) + 1)
                    # lambdafuncsmeared = lambda Ephs : Ephsmeared_from_monoline_Er_spectrum(Ephs,args) * trigger_error_function(Ephs)
                    lambdafuncsmeared = partial(get_func_smeared, args=args)
                    
                    if index < 200 :                        
                        ListFunctionSmeared.append(lambdafuncsmeared)
                        
                    
                return partial(sum_func, func_list=ListFunctionSmeared)

            else:
                if EnergyScale=="eV":
                    sigmareso = self.Intel.SigmaeV
                elif EnergyScale=="eVee":
                    sigmareso = self.Intel.SigmaeVee
                else:
                    print("Problem with EnergyScale {}".format(EnergyScale))
                index = 0
                for Energy in self.LineEnergies:                    
                    args = self.LineRates[index], Energy, self.Intel,EnergyScale,self.RecoilType
                    trigger_threshold = self.Intel.TriggerNsigma*sigmareso
                    trigger_error_function = lambda x :  0.5 * (special.erf((x-trigger_threshold)/(np.sqrt(2)*sigmareso)) + 1)
                    lambdafuncsmeared = lambda Ephs : self.ultimate_smeared1D_from_monoline_Er_spectrum(Ephs,args) * trigger_error_function(Ephs)
                    ListFunctionSmeared.append(lambdafuncsmeared)
                    print("testing "+str(index))
                    index += 1
                sumfunc = self.SumFunc(ListFunctionSmeared) 
                return sumfunc
                print("not implemented yet")  
    
    class SumFunc :
        def __init__(self,List):
            self.List = List
            
        def __call__(self,Ephs):
            return sum(l(Ephs) for l in self.List)
            
        
        
    def GetEphononSmearedFunc(self,EnergyScale="eV",smearing_option="gaussian_filter1d",nsig=5,precision=100,quantized = None):
        if quantized == None:
            quantized = self.Intel.Quantization
        
        #print("Quantization : ",quantized)
        if quantized == False:
            dNdEr = self.GetFunc()
            #TF1GetErFromEph = lambda xx : self.GetErFromEph(xx)
            args = dNdEr,self.Intel,EnergyScale,self.RecoilType
            lambdafuncsmeared = lambda z :self.ultimate_smeared1D_from_Er_spectrum(z,args)
            return lambdafuncsmeared
        else:
            tempTF1eph = self.GetEphononFunc(EnergyScale)
            #print(self.Intel.SigmaeV,"vs " ,self.Intel.SigmaeVee)
            if EnergyScale=="eV":
                sigmareso = self.Intel.SigmaeV
            elif EnergyScale=="eVee":
                sigmareso = self.Intel.SigmaeVee
            else:
                print("Problem with EnergyScale {}".format(EnergyScale))
            #smearing_option = "ultime_quad2"
            #smearing_option = "Rieman_Trapz"
            smearing_option="gaussian_filter1d"
            #smearing_option = "ultimate"
            if self.Intel._EfficiencyCurveImport == True:
                TF1eph = lambda t : tempTF1eph(t) * self.Intel._EfficiencyCurveInterpolator(t)
            else:
                TF1eph = lambda t : tempTF1eph(t) * (t>=(self.Intel.TriggerNsigma*sigmareso))
            lambdafuncsmeared = lambda z : Gaus_smearing(z,TF1eph,sigmareso,nsig=nsig,method=smearing_option,precision=precision,onlypositive=False)
            return lambdafuncsmeared
       
    def ultimate_smeared1D_from_Er_spectrum(self,Ephs,args):
        # check with Q = 1 instead of ionizaion yeild
        Ephs = np.array(Ephs,dtype=float)

        dNdEr, AllIntel,scale,RecoilType = args
        nsig = 7
        npts = 1000
        pct = AllIntel.SigmaeVpercent
        V = AllIntel.Voltage
        epsilon = AllIntel.Epsilon
        Quenching = AllIntel.Quenching
        sigma_phonon = AllIntel.SigmaeV
        F = AllIntel.Fano

        assert scale in ['eV','eVee']
        
        def resolutionVSer(Er):
            Er = np.array(Er,dtype=float)
            Q = Ionization_Yield(Er,recoiltype=RecoilType,energyscale="eV",Quenching=Quenching)
            sigma_baseline2 = np.power(np.ones_like(Er)*sigma_phonon,2)
            sigma_Fano2 = F*Er*Q/epsilon
            sigma_pct2 = np.power(pct * Er,2)
            sigma_reso = np.sqrt(sigma_baseline2+sigma_Fano2+sigma_pct2) 
            #sigma_reso = np.sqrt(sigma_baseline2+sigma_Fano2)
            return sigma_reso
        
            
            
        
        # calculations performed with eV, not eVee, so in case there is eVee, convert it into eV
        if (scale == 'eVee'):
            Ephs = Ephs * (1+V/epsilon)
    
        Er_reso = self.GetErFromEph( np.maximum(Ephs,0) ,energyscale="eV")
        sigma_reso = resolutionVSer(Er_reso)
    
        Eph_min = np.maximum(0, Ephs-(nsig*sigma_reso) )
        Er_minintegrale = self.GetErFromEph( Eph_min ,energyscale="eV")
        Er_minintegrale = np.maximum(Er_minintegrale,0)
        #recoil energies are necessarily greater than 0

        Eph_max = np.maximum(0, Ephs+(nsig*sigma_reso) )
        Er_maxintegrale = self.GetErFromEph( Eph_max ,energyscale="eV")
        #no boundary necessary   
        
        Er_edges = np.linspace(Er_minintegrale,Er_maxintegrale,npts)    
        Er_means = 0.5 * (Er_edges[1:] + Er_edges[:-1])
        dEr = Er_edges[1:] - Er_edges[:-1]    
        
        #assert np.all(dEr>=0)
        """
        for val,ind in enumerate(dEr):
            if (val<=0):
                print('val=',val) 
        """
        Q = Ionization_Yield(Er_means,recoiltype=RecoilType,energyscale="eV",Quenching=Quenching)
        
        Ephs_from_Er = Er_means*(1.+Q * V/epsilon)
        #if (scale == 'eVee'):
        #    Ephs_from_Er / (1+V/epsilon)
        #    sigma_phonon = sigma_phonon / (1+V/epsilon)
        #print(Ephs_from_Er)
        #if (scale == 'eVee'):
        #    Ephs_from_Er = Ephs_from_Er / (1+V/epsilon)
        sigma_integrale = resolutionVSer(Er_means)
        gausarg =np.exp(-np.power(Ephs - Ephs_from_Er,2)/(2*np.power(sigma_integrale,2)))
        coef = 1. / (np.sqrt(2*pi)*sigma_integrale)
        if (scale == 'eVee'):
            coef = coef * (1+V/epsilon)
        result = np.sum(coef * dNdEr(Er_means)*gausarg*dEr,axis=0) 
        return result 
    
    ### ATTENTION EN DESSOUS C EST EN COURS !!!!! CEST TOUT FAUX
    def ultimate_smeared1D_from_monoline_Er_spectrum(self,Ephs,args):
        print('yes you are in')
        # check with Q = 1 instead of ionizaion yiEld
        Ephs = np.array(Ephs,dtype=float)

        #dNdEr, AllIntel,scale,RecoilType = args
        
        rate_Er, monoline_Er, AllIntel,scale,RecoilType = args
    
        
        assert RecoilType in ["monoER","sumMonoER"]
        nsig = 7
        npts = 1000
        pct = AllIntel.SigmaeVpercent
        V = AllIntel.Voltage
        epsilon = AllIntel.Epsilon
        Quenching = AllIntel.Quenching
        sigma_phonon = AllIntel.SigmaeV
        F = AllIntel.Fano

        assert scale in ['eV','eVee']
        
        def resolutionVSer(Er):
            Er = np.array(Er,dtype=float)
            Q = Ionization_Yield(Er,recoiltype=RecoilType,energyscale="eV",Quenching=Quenching)
            sigma_baseline2 = np.power(np.ones_like(Er)*sigma_phonon,2)
            sigma_Fano2 = F*Er*Q/epsilon
            sigma_pct2 = np.power(pct * Er,2)
            sigma_reso = np.sqrt(sigma_baseline2+sigma_Fano2+sigma_pct2) 
            #sigma_reso = np.sqrt(sigma_baseline2+sigma_Fano2)
            return sigma_reso
        
        # calculations performed with eV, not eVee, so in case there is eVee, convert it into eV
        #print("rate:",rate_Er)
        #print("mono:",monoline_Er)
        #print("Ephs before",Ephs)
        if (scale == 'eVee'):
            Ephs = Ephs * (1+V/epsilon)
    
        #print("Ephs after",Ephs)
        #Er_reso = self.GetErFromEph( np.maximum(Ephs,0) ,energyscale="eV")
        #sigma_reso = resolutionVSer(monoline_Er)
        """
        Eph_min = np.maximum(0, Ephs-(nsig*sigma_reso) )
        Er_minintegrale = self.GetErFromEph( Eph_min ,energyscale="eV")
        Er_minintegrale = np.maximum(Er_minintegrale,0)
        #recoil energies are necessarily greater than 0

        Eph_max = np.maximum(0, Ephs+(nsig*sigma_reso) )
        Er_maxintegrale = self.GetErFromEph( Eph_max ,energyscale="eV")
        #no boundary necessary   
        
        Er_edges = np.linspace(Er_minintegrale,Er_maxintegrale,npts)    
        Er_means = 0.5 * (Er_edges[1:] + Er_edges[:-1])
        dEr = Er_edges[1:] - Er_edges[:-1]    
         """ 
        #assert np.all(dEr>=0)
        """
        for val,ind in enumerate(dEr):
            if (val<=0):
                print('val=',val) 
        """
        Q = Ionization_Yield(monoline_Er,recoiltype=RecoilType,energyscale="eV",Quenching=Quenching)
        #print("Q : ",Q)
        Ephs_from_monoline_Er = monoline_Er*(1.+Q * V/epsilon)
        #print("Ephs_from_monoline_Er: ",Ephs_from_monoline_Er)
        sigma_integrale = resolutionVSer(monoline_Er)
        gausarg =np.exp(-np.power(Ephs - Ephs_from_monoline_Er,2)/(2*np.power(sigma_integrale,2)))
        coef = 1. / (np.sqrt(2*pi)*sigma_integrale)
        if (scale == 'eVee'):
            coef = coef * (1+V/epsilon)
        result = coef * rate_Er * gausarg
        #result = np.sum(,axis=0) 
        return result 
        
        
    def Information(self):
        print("Detailed information about Spectrum")
        print("Emin = ",self.Emin)
        print("Emax = ",self.Emax)
        print("Nptx = ",self.Nptx)
        print("sigmaeV = ",self.Intel.SigmaeV)
        print("Stype",self.Stype)
        
    



def Eph_from_Er_spectrum(Eph,args):
    
    Eph = np.array(Eph,dtype=float)
    func_Er, AllIntel,scale,RecoilType,TF1mufromEph = args
    
    finalresult = np.zeros_like(Eph,dtype=float)
    maskpositive = (Eph>0)
    maskpositive_IDS = np.where(maskpositive)[0]
    Eph = Eph[maskpositive]
    
    assert RecoilType in RecoilType_list
    V = AllIntel.Voltage
    epsilon = AllIntel.Epsilon
    gap = AllIntel.Gap
    F = AllIntel.Fano
    Quenching = AllIntel.Quenching
    switch = 200
    assert (scale=="eV") | (scale=="eVee")
    tolerance = 1e-9
    
    if(scale=="eVee"):
        Eph = Eph * (1+V/epsilon) # to convert eVee into eV
    #from that point Eph is necessarily in eV

    Er = Eph
    N = np.zeros_like(Eph,dtype=int)
    result = np.zeros_like(Eph,dtype=float)
    counter = 0

    
    if (V==0 or RecoilType=="HO"):
        result = func_Er(Er) # dNdEr = dNdEph for V=0
    else:
        MU = TF1mufromEph(Eph) #np.zeros_like(Eph) #
        nstart = np.floor(MU)
        N = nstart
        resultfound = np.zeros_like(Eph,dtype=bool)
        while(1):
            Er = Eph - N * V 

            mask = (N>=0) & (Eph>=0) & (Er>0) & (resultfound==False)
            mask_IDS = np.where(mask)[0]
            if(np.all(~mask)):
                break

            counter +=1

            rate = func_Er(Er[mask])
            if(RecoilType=="ER"):
                prob = Prob_Ion(N[mask],Er[mask],AllIntel)
            else:
                #Q = Nuclear_Ionization_Yield(Er[mask],Quenching,energyscale="eV")   
                Q = Ionization_Yield(Er[mask],RecoilType,energyscale="eV",Quenching=Quenching)   
                mu = Er[mask] / epsilon * Q
                #print("avant",mu)
                mu = mu * (Er[mask]>gap) # ATTENTION AJOUT QUENTIN
                #ATTENTION, pansement pour s'assurer que si energie en dessous du gap on ait 0 paire crée
                # je considère dans ce cas que mu=0 et donc que P(N=0|mu=0) = 1
                # idealement il faudrait que Prob_Ion dispose de l'option ER et NR
                prob = Prob_DWB_arraymu(N[mask],mu,F,switch)  

            result[mask_IDS] += prob * rate
            specialIDS = np.where((prob<tolerance) & (counter>10))

            lowprob_IDS = mask_IDS[specialIDS]
            resultfound[lowprob_IDS] = True

            N = N-1

            if counter>10000:
                print("wow................That should not happen, check Eph_from_Er_spectrum in Ultimate_Libraries.py")
        N = nstart +1
        resultfound = np.zeros_like(Eph,dtype=bool)
        while(1):
            Er = Eph - N * V 

            mask = (N>=0) & (Eph>=0) & (Er>0) & (resultfound==False)
            mask_IDS = np.where(mask)[0]

            if(np.all(~mask)):
                break

            counter +=1

            rate = func_Er(Er[mask])
            if(RecoilType=="ER"):
                prob = Prob_Ion(N[mask],Er[mask],AllIntel)
            else:
                Q = Ionization_Yield(Er[mask],RecoilType,energyscale="eV",Quenching=Quenching) 
                #Q = Nuclear_Ionization_Yield(Er[mask],Quenching,energyscale="eV") 
                mu = Er[mask] / epsilon * Q
                mu = mu * (Er[mask]>gap) # ATTENTION AJOUT QUENTIN
                prob = Prob_DWB_arraymu(N[mask],mu,F,switch)  
            result[mask_IDS] += prob * rate

            specialIDS = np.where((prob<tolerance) & (counter>10))#& (N[mask]>3))  # & (N[mask]>10)

            lowprob_IDS = mask_IDS[specialIDS]
            resultfound[lowprob_IDS] = True

            N = N+1

            if counter>10000:
                print("wow................That should not happen, check Eph_from_Er_spectrum in Ultimate_Libraries.py")

    if(scale=="eVee"):
        result = result * (1+V/epsilon)

    finalresult[maskpositive_IDS] = result
    return finalresult



def Ephsmeared_from_monoline_Er_spectrum(Ephs,args):
    
    Ephs = np.array(Ephs,dtype=float)
    rate_Er, monoline_Er, AllIntel,scale,RecoilType = args
    
    #finalresult = np.zeros_like(Eph,dtype=float)
    
    assert RecoilType in ["monoER","sumMonoER"]
    V = AllIntel.Voltage
    epsilon = AllIntel.Epsilon
    F = AllIntel.Fano
    Quenching = AllIntel.Quenching
    
    switch = 200
    assert (scale=="eV") | (scale=="eVee")
    tolerance = 1e-9
    
    if(scale=="eVee"):
        Ephs = Ephs * (1+V/epsilon) # to convert eVee into eV
        #resolution = AllIntel.SigmaeVee
        
    resolution = AllIntel.SigmaeV
    #from that point Eph is necessarily in eV

    #Er = Eph
    N = 0
    result = np.zeros_like(Ephs,dtype=float)
    counter = 0
    calculationmethod = "fast"
    
    if (V==0 or RecoilType=="HO"):
        result = rate_Er * ss.norm.pdf(x=Ephs,loc=monoline_Er,scale=resolution)
        
    else:
        if(RecoilType=="monoER" or RecoilType=="sumMonoER"):
            MU = monoline_Er/epsilon
        else:
            print("not implemented yet, ")
            #MU = Er*Q/Globalepsilongamma
            
        nstart = np.floor(MU)
        N = nstart
        resultfound = np.zeros_like(Ephs,dtype=bool)
        while(1):
            Eph = monoline_Er + N * V 
            
            mask = (N>=0) & (Eph>=0) & (resultfound==False)
            mask_IDS = np.where(mask)[0]
            if(np.all(~mask)):
                break

            counter +=1
            rate = rate_Er # always same rate
            if(RecoilType=="monoER"or RecoilType=="sumMonoER"):
                prob = Prob_Ion(N,monoline_Er,AllIntel)
            else:
                print("not implemented yet")
                #Q = Nuclear_Ionization_Yield(Er[mask],Quenching,energyscale="eV")
                #mu = Er[mask] / epsilon * Q
                #prob = Prob_DWB_arraymu(N[mask],mu,F,switch)  
            gausfactor = ss.norm.pdf(x=Ephs,loc=Eph,scale=resolution)
            result += prob * rate * gausfactor
            #result[mask_IDS] += prob * rate * gausfactor
            specialIDS = np.where((prob<tolerance) & (counter>10))#& (N[mask]>3))  # & (N[mask]>10)
 
            lowprob_IDS = mask_IDS[specialIDS]
            resultfound[lowprob_IDS] = True

            N = N-1
             
            if counter>10000:
                print("wow.......................")
        N = nstart +1
        resultfound = np.zeros_like(Ephs,dtype=bool)
        while(1):
            Eph = monoline_Er + N * V 
            
            mask = (N>=0) & (Eph>=0) & (resultfound==False)
            mask_IDS = np.where(mask)[0]
            if(np.all(~mask)):
                break

            counter +=1
            rate = rate_Er # always same rate
            if(RecoilType=="monoER"or RecoilType=="sumMonoER"):
                prob = Prob_Ion(N,monoline_Er,AllIntel)
            else:
                print("not implemented yet")
                #Q = Nuclear_Ionization_Yield(Er[mask],Quenching,energyscale="eV")
                #mu = Er[mask] / epsilon * Q
                #prob = Prob_DWB_arraymu(N[mask],mu,F,switch)  
            gausfactor = ss.norm.pdf(x=Ephs,loc=Eph,scale=resolution)
            result += prob * rate * gausfactor
            #result[mask_IDS] += prob * rate * gausfactor
            specialIDS = np.where((prob<tolerance) & (counter>10))#& (N[mask]>3))  # & (N[mask]>10)
            
            lowprob_IDS = mask_IDS[specialIDS]
            resultfound[lowprob_IDS] = True
            
            N = N+1
            
            if counter>10000:
                print("wow.......................")
            
    if(scale=="eVee"):
        result = result * (1+V/epsilon)

    return result



def fonctionHO(x):
    args = [1/1000.,906897,39.2,47765,202.607]
    par0,par1,par2,par3,par4 = args
    result = par0 * (par1*np.exp(-x/par2) + par3 * np.exp(-x/par4))
    return result

def fonctionHOdivided1000(x):
    args = [1/1000./1000.,906897,39.2,47765,202.607]
    par0,par1,par2,par3,par4 = args
    result = par0 * (par1*np.exp(-x/par2) + par3 * np.exp(-x/par4))
    return result

def ExamplePlotANR():
    
    # Plotoptions
    Emin, Emax, nptx, Energyscale = 200, 1000, 1000, "eV"
    ymin,ymax = 1e-5,3e3

    Exp = Experiment()
    Exp.Voltage = 200 # Volts
    Exp.SigmaeV = 30  # Phonon baseline resolution
    Exp.Exposure = 1 # kg.days
    Exp.IonModel = "CDMS" # options are ["Rouven","CDMS"]
    Exp.Fano = 0.2 # used only when IonModel=="CDMS"
    Exp.Target = "Ge"
    # Quenching options are ["EDW","Lindhard_pure","Lindhard_corrected"]
    Exp.Quenching = "EDW" 
    #Exp.Quenching = "Lindhard_pure"
    #Exp.Quenching = "Lindhard_corrected"
    
    
    # Backgrounds Definition
    H = Spectrum("HO",Exp)
    H.SetFunc(fonctionHO,0,1e9)
    TF1_HO_Erecul = H.GetFunc()
    TF1_HO_Ephonon = H.GetEphononFunc(Energyscale) # ["eV","eVee]
    TF1_HO_EphononSmeared = H.GetEphononSmearedFunc(Energyscale)  # ["eV","eVee]
    
    x = np.linspace(Emin,Emax,nptx)
    HO_Erecul = TF1_HO_Erecul(x)
    HO_Ephonon = TF1_HO_Ephonon(x)
    HO_EphononSmeared = TF1_HO_EphononSmeared(x)
    
    C = Spectrum("ER",Exp)
    C.SetFunc(lambda x : np.exp(-0.000001*x)*1e-2,0,1e9)
    TF1_COMPTON_Erecul = C.GetFunc()
    TF1_COMPTON_Ephonon = C.GetEphononFunc(Energyscale) # ["eV","eVee]
    TF1_COMPTON_EphononSmeared = C.GetEphononSmearedFunc(Energyscale)  # ["eV","eVee]
    COMPTON_Erecul = TF1_COMPTON_Erecul(x)
    COMPTON_Ephonon = TF1_COMPTON_Ephonon(x)
    COMPTON_EphononSmeared = TF1_COMPTON_EphononSmeared(x)
    W = WIMP_Parameters()
    W.WIMPmass = 1 # GeV/c2
    W.CrossSection = 1e-1 # pb 
    #W.Exposure = 1 # kg.days always put 1 kg.day for WIMPs to avoid double scaling with Spectrum
    W.Target = Exp.Target # ["Ge",...]
    
    S = Spectrum(W,Exp)   
    TF1_WIMPs_Erecul = S.GetFunc()
    TF1_WIMPs_Ephonon = S.GetEphononFunc(Energyscale) # ["eV","eVee]
    TF1_WIMPs_EphononSmeared = S.GetEphononSmearedFunc(Energyscale)  # ["eV","eVee]
    
    WIMPs_Erecul = TF1_WIMPs_Erecul(x)
    WIMPs_Ephonon = TF1_WIMPs_Ephonon(x)
    WIMPs_EphononSmeared = TF1_WIMPs_EphononSmeared(x)
    
    plt.figure(figsize=(9,6))
    fig,ax = plt.subplots() 
    plt.plot(x,HO_EphononSmeared,label="HO",color="red")
    plt.plot(x,WIMPs_EphononSmeared,label="WIMPs",color="darkorange")
    plt.plot(x,COMPTON_EphononSmeared,label="COMPTON",color="royalblue")
    plt.yscale('log')
    plt.grid()
    plt.title("{0}, F = {1}, IonModel = {2}, baseline = {3} eV , {4} V".format(Exp.Target,Exp.Fano,Exp.IonModel,Exp.SigmaeV,Exp.Voltage))
    plt.xlabel("Energy [{0}]".format(Energyscale))
    plt.ylabel("Event Rate [/{0} kg.days/{1}]".format(Exp.Exposure,Energyscale))
    plt.ylim((ymin,ymax))
    if(Energyscale=="eVee"):
        ftransform = lambda x : x/Exp.Epsilon
        finv = lambda x : x*Exp.Epsilon
        secax = ax.secondary_xaxis('top', functions=(ftransform,finv))
    else:
        ftransform = lambda x : x/(1+Exp.Voltage/Exp.Epsilon)/Exp.Epsilon
        finv = lambda x : x*(1+Exp.Voltage/Exp.Epsilon)/Exp.Epsilon
        secax = ax.secondary_xaxis('top', functions=(ftransform,finv),color="royalblue")
        
    secax.set_xlabel('Electron number of e$^-$/ h$^+$ for electron recoils')
    plt.ylim((1e-5,1000))
    plt.legend(loc="upper right")
    
    #print("n WIMP events expected between 100 and 200 eV:",quad(TF1_WIMPs_Erecul,0.01,1000)[0])


def ExamplePlotMonoEnergetique():

    E = Experiment()
    E.Voltage = 10
    E.SigmaeV = 1
    E.Exposure = 10
    S = Spectrum("monoER",E)
    S.SetLine(5.3,10)
    tf1ephs = S.GetEphononSmearedLine("eVee")
    
    eminplot,emaxplot,nptxplot= 0,25,1000
    x = np.linspace(eminplot,emaxplot,nptxplot)
    y = tf1ephs(x)
    print(S.Integral(tf1ephs,eminplot,emaxplot,nptxplot))
    plt.plot(x,y)
    plt.gca().set_ylim(bottom=1e-2,top=10E2)
    plt.yscale('log')
    

class Oracle:
    def __init__(self, AnalysisEmin=None,AnalysisEmax=None,AnalysisNptx=None,LimitSetterCO = None , SignalType = 'WIMP' ):
        self._Mode = "Projection"
        self._Signal = None
        self._BackgroundTotal = None
        self._BackgroundList = dict()
        self._AnalysisEmin = AnalysisEmin
        self._AnalysisEmax = AnalysisEmax
        self._AnalysisNptx = AnalysisNptx
        self._ROIlist = None
        self._LimitSetterCO = LimitSetterCO
        self._VariableOI = None
        self._VariableOIExcluded = None
        self._mub = None
        self._mubexcluded = None
        self._mus = None
        self._SignalType = SignalType

        if LimitSetterCO == None:
            self._LimitSetterCO = LimitSetter(CL=90,Nvalues=1000,rewrite=False)
        else:
            self._LimitSetterCO = LimitSetterCO
            

    
    def PlotROIs(self,color="red",ls="--",prop=1e-4):
        liminf,limsup = plt.gca().get_ylim()
        for minroi,maxroi in self._ROIlist:
            sup = liminf + (limsup-liminf) * prop
            x = np.linspace(liminf,sup,10)
            plt.plot(minroi*np.ones_like(x),x,color=color,ls=ls)
            plt.plot(maxroi*np.ones_like(x),x,color=color,ls=ls)
    
    def PlotThreshold(self,color="gray",ls="--"):
        liminf,limsup = plt.gca().get_ylim()
        x = np.linspace(liminf,limsup,10)
        plt.plot(self._AnalysisEmin*np.ones_like(x),x,color=color,ls=ls)
        
    
    def SetVariableOfInterest(self,value=None):
        self._VariableOI = value
    
    def GetVariableOfInterest(self):
        return self._VariableOI
        
    def GetVariableOfInterestExcluded(self):
        return self._VariableOIExcluded
    
    def CalculateROI_Bounds(self, fs=None, fb=None, emin=None, emax=None, nptx=None, Plot=False):
        if fs == None:
            fs = self._Signal
        if fb == None:
            fb = self._BackgroundTotal
        if emin == None:
            emin = self._AnalysisEmin
        if emax == None:
            emax = self._AnalysisEmax
        if nptx == None:
            nptx = self._AnalysisNptx
        LSCO = self._LimitSetterCO
        self._ROIlist = LSCO.Multi_DetermineROI(fs,fb,emin,emax,nptx,Plot=Plot)
        return self._ROIlist
            
    def CalculateROI_ExpectedBackground(self,precision = None):
        if precision == None:
            precision = self._AnalysisNptx
        self._mub = TF1IntegralMulti(self._BackgroundTotal,self._ROIlist,precision)
        return self._mub
    
    def CalculateROI_ExpectedSignal(self,precision = None):
        if precision == None:
            precision = self._AnalysisNptx
        self._mus = TF1IntegralMulti(self._Signal,self._ROIlist,precision)
        return self._mus
        
    def CalculateROI_ExcludedEvents(self,Mode = None):
        if Mode == None:
            Mode = self._Mode
        assert Mode in ["Projection","Experiment"]
        mub = self.CalculateROI_ExpectedBackground()
        LSCO = self._LimitSetterCO
        if Mode == "Projection":
            self._mubexcluded = LSCO.PoissonLimitMediane(mub)
        elif Mode == "Experiment":
            self._mubexcluded = LSCO.PoissonLimit(mub)
        return self._mubexcluded
    
    def GetExcludedMu(self):
        return self._mubexcluded
    
    def CalculateExcludedVariableOfInterest(self,VOI=None):
        if VOI==None:
            VOI = self._VariableOI
        self.CalculateROI_ExcludedEvents() # calculates both mub and mubexcluded
        self.CalculateROI_ExpectedSignal() # calculates mus for initial VOI
        self._VariableOIExcluded = self._mubexcluded * VOI/self._mus
        if self._SignalType == "DP"  :
            self._VariableOIExcluded = np.sqrt(self._mubexcluded * VOI**2/self._mus)
        if self._SignalType == "ALP"  :
            self._VariableOIExcluded = np.sqrt(self._mubexcluded * VOI**2/self._mus)
        if self._SignalType == "DMES"  :
            self._VariableOIExcluded =  self._mubexcluded * VOI/self._mus
        return self._VariableOIExcluded 
        
    def SetAnalysisNptx(self,value):
        self._AnalysisNptx = value
        
    def GetAnalysisNptx(self):
        return self._AnalysisNptx
        
    def SetAnalysisRange(self,values):
        self._AnalysisEmin, self._AnalysisEmax = values
        
    def GetAnalysisRange(self):
        return self._AnalysisEmin,self._AnalysisEmax
    
    def GetSignal(self):
        return self._Signal
    
    def GetBackgroundNames(self):
        return [name for name,_ in self._BackgroundList.items()]
    
    def GetBackground(self,name=None):
        if name.lower == "all":
            return GetTotalBackground()
        else:    
            return self._BackgroundList[name]

    def GetTotalBackground(self):
        return self._BackgroundTotal
    
    def AddBackground(self,TF1,name):
        f = self._BackgroundTotal
        if f is None:
            newf = TF1
        else:
            newf = lambda x : f(x) + TF1(x)
        self._BackgroundTotal = newf 
        self._BackgroundList[name] = TF1
        
    def AddBackgrounds(self,multipleTF1=None,names=None):
        for TF1,name in zip(multipleTF1,names):
            self.AddBackground(TF1,name)
            
    def AddSignal(self,TF1):
        f = self._Signal
        if f is None:
            newf = TF1
        else:
            newf = lambda x : f(x) + TF1(x)
        self._Signal = newf     
        
    def AddSignals(self,multipleTF1):
        for TF1 in multipleTF1:
            self.AddSignal(TF1)
            
        
