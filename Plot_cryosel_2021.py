from Tools_Libraries import *
from IonModel_Libraries import *
from LimitSetting_Libraries import *
from WIMP_Libraries import *
from DP_Libraries import *
from ALP_Libraries import *
from DMES_Libraries import *
from Ultimate_Libraries import *
from Backgrounds_Libraries import *
import sys



def CalculateLimit(Exp = Experiment(),OR = Oracle(), EnergyScale="eV",mass=200,massindex=False,bash=False):
    start = time.time()
    assert OR._SignalType in ["DP","WIMP","ALP","DMES"]
    assert EnergyScale in ["eV","eVee"]
    Emin, Emax = OR.GetAnalysisRange()
    nptx = OR.GetAnalysisNptx()
    logx = False
    x = np.linspace(Emin,Emax,nptx) if logx==False else np.logspace(np.log10(Emin),np.log10(Emax),nptx)
    
    Emin = Exp.TriggerNsigma*Exp.SigmaeV if (EnergyScale=="eV") else Exp.TriggerNsigma*Exp.SigmaeVee
    OR.SetAnalysisRange((Emin,Emax))
    
    # define background components
    Compton = Spectrum("ER",Exp)
    Compton.SetFunc(lambda x : EDWfuncCOMPTON(x,"eV"),0,1e9)
    ComptonEnergySpectrum = Compton.GetEphononSmearedFunc(EnergyScale)
    
    HeatOnly = Spectrum("HO",Exp)
    HeatOnly.SetFunc(fonctionHOdivided1000,0,1e9)
    HeatOnlyEnergySpectrum = HeatOnly.GetEphononSmearedFunc(EnergyScale)
    
    Beta = Spectrum("BETA",Exp)
    Beta.SetFunc(lambda x :  EDWfuncBETA(x,"eV"),0,1e9)
    BetaEnergySpectrum = Beta.GetEphononSmearedFunc(EnergyScale) 
    
    Neutron = Spectrum("NEUTRON",Exp)
    Neutron.SetFunc(lambda x :  EDWfuncNEUTRON(x,"eV"),0,1e9)
    NeutronEnergySpectrum = Neutron.GetEphononSmearedFunc(EnergyScale) 
    
    
    Tritium = Spectrum("ER",Exp)
    Tritium.SetFunc(lambda x :  EDWfuncTRITIUM(x,"eV"),0,1e9)
    TritiumEnergySpectrum = Tritium.GetEphononSmearedFunc(EnergyScale) 
    
    Lead = Spectrum("LEAD",Exp)
    Lead.SetFunc(lambda x :  EDWfuncLEAD(x,"eV"),0,1e9)
    LeadEnergySpectrum = Lead.GetEphononSmearedFunc(EnergyScale) 
    
    
    
    if OR._SignalType == "DP" : 
        D = DP_Parameters()
        if massindex == False :
            D.SetDPmass(mass)
        else :
            D.SetDPmassindex(mass) 
        actualmass = D.GetDPmass()
        D.SetKappa(1e-14)
        DarkPhotons = Spectrum(D,Exp)
        DarkPhotonsEnergySpectrum = DarkPhotons.GetEphononSmearedLine(EnergyScale)
    
    if OR._SignalType == "WIMP" :
        W = WIMP_Parameters()
        W.WIMPmass = mass
        actualmass = W.WIMPmass
        W.CrossSection = 1e-38
        Wimps = Spectrum(W,Exp)
        WimpsEnergySpectrum = Wimps.GetEphononSmearedFunc(EnergyScale) 
    
    elif OR._SignalType == "ALP":
        A = ALP_Parameters()
        if massindex == False :
            A.SetALPmass(mass)
        else :
            A.SetALPmassindex(mass) 
        actualmass = A.GetALPmass()
        A.SetgAe(1e-12) 
        ALP = Spectrum(A,Exp)
        ALPEnergySpectrum = ALP.GetEphononSmearedLine(EnergyScale) 

    elif OR._SignalType == "DMES":
        DE = DMES_Parameters()
        if massindex == False :
            DE.SetDMmass(mass)
        else :
            DE.SetDMmassindex(mass) 
        actualmass = DE.GetDMmass()
        #DE.SetCrossSection(1e-37) 
        DES = Spectrum(DE,Exp)
        DESEnergySpectrum = DES.GetEphononSmearedLines(EnergyScale)
          
    OR.AddBackground(ComptonEnergySpectrum,"Compton Bkg")
    OR.AddBackground(HeatOnlyEnergySpectrum,"HO Bkg")
    OR.AddBackground(BetaEnergySpectrum,"Beta Bkg") 
    OR.AddBackground(NeutronEnergySpectrum,"Neutron Bkg")    
    OR.AddBackground(TritiumEnergySpectrum,"Tritium Bkg") 
    OR.AddBackground(LeadEnergySpectrum,"Lead Bkg") 
    
    
    if OR._SignalType=="DP":
        OR.AddSignal(DarkPhotonsEnergySpectrum)
        OR.SetVariableOfInterest(D.GetKappa())
    elif OR._SignalType == "WIMP":
        OR.AddSignal(WimpsEnergySpectrum)
        OR.SetVariableOfInterest(W.CrossSection)
    elif OR._SignalType == "ALP":
        OR.AddSignal(ALPEnergySpectrum)
        OR.SetVariableOfInterest(A.GetgAe())
    elif OR._SignalType == "DMES":
        OR.AddSignal(DESEnergySpectrum)
        OR.SetVariableOfInterest(DE.GetCrossSection())
    
    fb = OR.GetTotalBackground()
    fs = OR.GetSignal()
    ROIlist = OR.CalculateROI_Bounds(fs,fb) # checked it is multi ROI methods
    
    if OR._SignalType=="DP":
        kappaexcluded = OR.CalculateExcludedVariableOfInterest()
        Dexcluded = copy.copy(D)
        Dexcluded.SetKappa(kappaexcluded)
        DarkPhotonsexcluded = Spectrum(Dexcluded,Exp)
   
    elif OR._SignalType == "WIMP":
        sigmaexcluded = OR.CalculateExcludedVariableOfInterest()
        Wexcluded = copy.copy(W)
        Wexcluded.CrossSection = sigmaexcluded
        Wimpsexcluded = Spectrum(Wexcluded,Exp)
        
    elif OR._SignalType == "ALP":
        gAeexcluded = OR.CalculateExcludedVariableOfInterest()
        Aexcluded = copy.copy(A)
        Aexcluded.SetgAe(gAeexcluded)  
        ALPsexcluded = Spectrum(Aexcluded,Exp)
        
    elif OR._SignalType == "DMES":
        sigmaDMexcluded = OR.CalculateExcludedVariableOfInterest()
        DEexcluded = copy.copy(DE)
        DEexcluded.SetCrossSection(sigmaDMexcluded)  
        DESexcluded = Spectrum(DEexcluded,Exp)           
    muexcluded = OR.GetExcludedMu()
    
    end = time.time()
    print(f"{end-start:.2f} seconds to calculate the limit" )
    
    if (bash == False):
        fig, ax = plt.subplots(figsize=(7,5))
        
        for name in OR.GetBackgroundNames():
            tf1 = OR.GetBackground(name)
            plt.plot(x,tf1(x),label=name,lw=4)
            
        yb = fb(x)
        ax.plot(x,yb,color="black",ls='--',label="Total Bkg",lw=4)       
        ax.set_ylim(bottom=1e-9,top=1e3)
        ax.set_yscale('log')
        
        if logx:
            ax.set_xscale('log')
            
        if OR._SignalType=="DP":
            title1 = "DP signal excluded, m={:.2f} eV/c$^2$".format(actualmass)
            title2 = "$\kappa={:.2e}$, $\mu={:.2f}$".format(kappaexcluded,muexcluded)
            fsexcluded = DarkPhotonsexcluded.GetEphononSmearedLine(EnergyScale)
        elif OR._SignalType == "WIMP":
            title1 = "WIMP signal excluded, m={:.2f} GeV/c$^2$".format(actualmass)
            title2 = "$\sigma$={:.2e} pb, $\mu={:.2f}$".format(sigmaexcluded,muexcluded) 
            fsexcluded = Wimpsexcluded.GetEphononSmearedFunc(EnergyScale)
        elif OR._SignalType == "ALP":
            title1 = "ALP signal excluded, m={:.2f} eV/c$^2$".format(actualmass)
            title2 = "$gAe$={:.2e} , $\mu={:.2f}$".format(gAeexcluded,muexcluded) 
            fsexcluded = ALPsexcluded.GetEphononSmearedLine(EnergyScale)
        elif OR._SignalType == "DMES":
            title1 = "DMES signal excluded, m={:.2f} MeV/c$^2$".format(actualmass/1e6)
            title2 = "$\sigma$={:.2e} cm^2, $\mu={:.2f}$".format(sigmaDMexcluded,muexcluded) 
            fsexcluded = DESexcluded.GetEphononSmearedLines(EnergyScale)   
        yexcl = fsexcluded(x)
        ax.plot(x,yexcl,label=title1,lw=4,color='red')
        ax.fill_between(x,yb,yexcl, alpha=0.5,where=(yexcl>yb),color='red')
        ax.plot([], [],' ',label=title2)

        if(EnergyScale=="eVee"):
            ftransform = lambda x : x/Exp.Epsilon
            finv = lambda x : x*Exp.Epsilon
            secax = ax.secondary_xaxis('top', functions=(ftransform,finv))
        else:
            ftransform = lambda x : x/(1+Exp.Voltage/Exp.Epsilon)/Exp.Epsilon
            finv = lambda x : x*(1+Exp.Voltage/Exp.Epsilon)/Exp.Epsilon
            secax = ax.secondary_xaxis('top', functions=(ftransform,finv),color="royalblue")
         

        secax.set_xlabel('Electron number of e$^-$/ h$^+$ for electron recoils')
        ax.grid(which='both')
        ax.set_xlabel("Energy [{}]".format(EnergyScale))
        ax.set_ylabel("Events / {} kg.days / {}".format(Exp.Exposure,EnergyScale))
        ax.legend(loc='upper right')
        
        OR.PlotROIs(color="red",ls="--",prop=1)
        OR.PlotThreshold()
        plt.show()
        end2 = time.time()
        print(f"{end2-end:.2f} additional seconds to do the plot" )   
    if OR._SignalType=="DP":
        return actualmass, kappaexcluded, muexcluded 
    elif OR._SignalType == "WIMP":
        return actualmass, sigmaexcluded, muexcluded
    elif OR._SignalType == "ALP":
        return actualmass, gAeexcluded, muexcluded
    elif OR._SignalType == "DMES":
        return actualmass/1e6, sigmaDMexcluded*1.0e+36, muexcluded
        

        
# We define here all parameter of the experiment
Exp = Experiment()
Exp.Voltage = float(sys.argv[1])
Exp.SigmaeV = float(sys.argv[2])
Exp.Exposure = int(sys.argv[3])
Exp.Fano = 0.15
Exp.TriggerNsigma = 5
Exp.Target = "Ge"
Exp.Quenching = "Lindhard_pure"
Exp.Quantization = True
Exp.Information()


#   initialization of limit setter
L = LimitSetter(CL=90,Nvalues=1000)
quantified = "Quantized"
if Exp.Quantization == False : 
    quantified = "NotQuantized"

EnergyScale = "eV"
EminAnalysis = 0 
EmaxAnalysis = 3000
nptx = 3000 # precision of the analysis, the higher the better, but slower
Mass = [0.11,0.115,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19]#[0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,2.,3.,5.,10.]
Indices = False
if(str(sys.argv[4]) == "DP") : 
    Mass = [10]#list(range(1,10,1))+list(range(10,20,5))+list(range(20,81,10))
    #Mass = range(80,81)
    Indices = True
if(str(sys.argv[4]) == "DMES") : 
    Mass = [1.5*1e6,3e6, 15*1e6,25*1e6,150*1e6,300*1e6]#[0.5*1e6,1e6, 2*1e6,50*1e6,100*1e6,1000*1e6]
    #Mass = range(80,81)
    Indices = False
if(str(sys.argv[4]) == "ALP") : 
    Mass = [1,10,50]
 
for m in Mass :
    
    EminAnalysis = 0
    #if(str(sys.argv[4]) == "WIMP") : 
        #EmaxAnalysis = m * Exp.Voltage * 10 
    #if(str(sys.argv[4]) == "DP") : 
     #   EmaxAnalysis = m * Exp.Voltage * 5 
    if(str(sys.argv[4]) == "ALP") : 
        EmaxAnalysis = m * Exp.Voltage * 5 
    if   EmaxAnalysis > 3000 :
        EmaxAnalysis= 3000
    nptx = int(( EmaxAnalysis - EminAnalysis )*3)
    OR = Oracle(EminAnalysis,EmaxAnalysis,nptx,L,str(sys.argv[4]))   
    mass, POIexl, Muexl = CalculateLimit(Exp = Exp, OR = OR, EnergyScale = "eV", mass = m, massindex = Indices, bash = True  )
    print("Projection_"+str(Exp.Voltage)+"V_"+str(Exp.SigmaeV)+"eVsigma_"+str(Exp.Exposure)+"kgday_"+str(Exp.TriggerNsigma)+"trigger_"+Exp.Quenching+"_"+str(Exp.Fano)+"Fano_"+quantified+"_"+str(sys.argv[4])+"_"+str(m))
    with open("Projection_"+str(Exp.Voltage)+"V_"+str(Exp.SigmaeV)+"eVsigma_"+str(Exp.Exposure)+"kgday_"+str(Exp.TriggerNsigma)+"trigger_"+Exp.Quenching+"_"+str(Exp.Fano)+"Fano_"+quantified+"_"+str(sys.argv[4]),'a') as f:
            f.write(str(mass)+" "+str(POIexl)+" "+str(Muexl)+" \n")
    
