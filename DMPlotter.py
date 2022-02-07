import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.legend_handler
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator,FormatStrFormatter

def loadlimit(path,filename,rescalefactor=1,usecols=(0,1)):
    mass,cross = np.loadtxt(str(path)+"/"+str(filename),unpack=True,usecols=usecols)
    cross=cross*rescalefactor
    return cross,mass
    

def plotlimit(path,name,filename,rescalefactor=1,usecols=(0,1),**goption):
    mass,cross = np.loadtxt(str(path)+"/"+str(filename),unpack=True,usecols=usecols)
    cross=cross*rescalefactor
    
    if not goption:
        goption = {"lw":3,
                   "marker":"o",
                   "ls":'--'
                   }    
    plt.plot(mass,cross,label=name,**goption)
    
def plotlimits(path,experiments,rescalefactor=1,usecols=(0,1),**goption):    
    for name in experiments:
        file = experiments[name]
        plotlimit(path,name,file,**goption,usecols=usecols)




def progWIMP():
    xmin,xmax,xunit = 0.05,10,"GeV"
    ymin,ymax,yunit = 1e-42,1e-31, "cm$^2$"
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xscale('log')
    ax.set_xlabel('Mass [{}]'.format(xunit))
    ax.set_yscale('log')
    ax.set_ylabel('$\sigma$ [{}]'.format(yunit))
    plt.grid(True,"both")
    
    goption = dict(lw=3,
                   marker="",
                   alpha=0.4)
    path = 'quentin'
    plotlimit(path,'CDEX Migdal','CDEX_Migdal.txt',ls='dashed',**goption,color='blue')
    plotlimit(path,'CRESST','CREST3_Migdal.txt',**goption,color='purple')
    plotlimit(path,'LUX Migdal','LUX_Migdal.txt',**goption,ls='dashed',color='green')
    plotlimit(path,'LUX','LUX_STD.txt',**goption,color='green')
    plotlimit(path,'Xenon1T Migdal','Xe1TMig.txt',**goption,ls='dashed',color='blueviolet')
    plotlimit(path,'CDMSlite','cdmslite.txt',**goption,color='orange')
    #plotlimit(path,'EDELWEISS Migdal Surf','RED20-Migdal.txt',**goption,ls='dashed',color='red')
    plotlimit(path,'EDELWEISS Surf','RED20-STD.txt',**goption,color='red')

    
    
    path = '.'
    goption = dict(lw=3,
                   marker="",
                   alpha=1)
    plotlimit(path,'CRYOSEL : 1 kg.d','WIMP-1kg.days.txt',rescalefactor=1e-36,**goption,color='red')
    #plotlimit(path,'365 kg.days','WIMP-365kg.days.txt',rescalefactor=1e-36,**goption,ls='dotted',color='red')
    #plotlimit(path,'last-WIMP-365kg.days.txt','last-WIMP-365kg.days.txt',rescalefactor=1e-36,**goption)

    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=25))
    plt.legend()
    
    plt.savefig("FigWIMP.pdf")
    plt.show()
    
def progDP():
    
    xmin,xmax = 1,40
    ymin,ymax = 1e-16,1e-9
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xscale('log')
    ax.set_xlabel('Mass [eV]')
    ax.set_yscale('log')
    ax.set_ylabel('$\kappa$')
    plt.grid(True,"both")
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=25))
    
   # ~/Desktop/FanoFactor/com-poisson-code/limits/LimitsPRL
   #EDELWEISS_Limit_DarkPhoton.txt
    #########################
    path = 'limitsallexperiments'
    experiments = dict(DAMIC='checked-DAMIC2019-DP.txt',
                       SENSEI='Sensei-DP.txt',
                       XENON='checked-XENON10-DP.txt',
                       SOLAR='checked-SOLAR-DP.txt',
                       CDMS='checked-CDMS-DP.txt'
                       )
    goption = dict(lw=3,marker="",alpha=0.4)
   # plotlimits(path,experiments,**goption)
    
    plotlimits(path,dict({'XENON':'checked-XENON10-DP.txt'}),**goption,ls='dotted',color='blueviolet')
    plotlimits(path,dict({'CDMS ':'checked-CDMS-DP.txt'}),**goption,ls='solid',color='orange')
    plotlimits(path,dict({'SENSEI ':'checked-SENSEI-F1Q2.txt'}),**goption,ls='dotted',color='seagreen')
    plotlimits(path,dict({'DAMIC ':'checked-DAMIC2019-DP.txt'}),**goption,ls='dashed',color='royalblue')
    plotlimits(path,dict({'SENSEI ':'Sensei-DP.txt'}),**goption,ls='dashed',color='darkgreen')
    
    
    path = 'LimitsPRL'
    experiments = {"EDW PRL":"EDELWEISS_Limit_DarkPhoton.txt"}
    plotlimits(path,experiments,**goption,color='red')
    path = 'limitsallexperiments'
    goption = dict(lw=1,marker="",alpha=1)
    plotlimits(path,dict({'SOLAR':'checked-SOLAR-DP.txt'}),**goption,ls='dashed',color='black')
    #########################
    path = '.'
    experiments = dict({'  1 kg.days':'DP-1kg.days.txt'})#,'365 kg.days':'DP-365kg.days.txt'
    goption = dict(lw=3,marker="",alpha=1)
    #plotlimits(path,experiments,**goption,color='red')
    #plotlimits(path,'DP-1kg.days.txt','DP-1kg.days.txt',rescalefactor=1,**goption,ls='3',color='red')
    #plotlimits(path,'DP-365kg.days.txt','DP-365kg.days.txt',rescalefactor=1,**goption,ls='dotted',color='red')
    plotlimit(path,'CRYOSEL : 1 kg.d ','DP-1kg.days.txt',rescalefactor=1,**goption,color='red')
    #plotlimit(path,'365 kg.days','DP-365kg.days.txt',rescalefactor=1,**goption,ls='dotted',color='red')
    #########################

    plt.legend()
    
    plt.savefig("FigDP.pdf")
    plt.show()
    

def progDMES():
    
    xmin,xmax = 0.53,1000
    ymin,ymax = 1e-39,1e-28
    #xmin,xmax = 0.4,1000
    #ymin,ymax = 1e-37,1e-28
    fig = plt.figure(figsize=(6,6))
    plt.grid(True,"both")
    ax = fig.add_subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xscale('log')
    ax.set_xlabel('$\mathrm{m_{DM}}$ [MeV/$\mathrm{c}^{2}$]', fontsize=15)
    #ax.yaxis.tick_right()
    ax.set_yscale('log')
    ax.set_ylabel('$\overline{\sigma_e}\quad [\mathrm{cm}^{2}]$', fontsize=15)
    #plt.grid(True,"both")
    ax.text(0.01, 0.95, 'DM-electron scattering ',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=10)#\n$\mathrm{F_{DM}} \propto 1/\mathrm{q}^{2}$'
    
    ax.tick_params(axis="y", which='both',direction='in')
    ax.tick_params(axis="x", which='both',direction='in')
    
    ylocator = ticker.LogLocator(base=10.0, numticks=25)
    plt.gca().yaxis.set_major_locator(ylocator)
    
    xlocator   = ticker.LogLocator(base=10,subs=(10.,0.2),numdecs=1,numticks=None)
    xformatter = ticker.LogFormatter(base=10.0, labelOnlyBase=False, minor_thresholds=None, linthresh=None)
    plt.gca().xaxis.set_major_locator(xlocator)
    plt.gca().xaxis.set_major_formatter(xformatter)
    #plt.gca().xaxis.set_minor_locator( ticker.LogLocator(base=1,subs=(1.0,),numdecs=1,numticks=None))
    #plt.gca().xaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(200,0.1)))
    #########################
    path = 'limitsallexperiments'
    goption = dict(lw=3,marker="",alpha=0.4)
    
    plotlimits(path,dict({"DarkSide50":"checked-DARKSIDE50-F1Q2.txt"}),**goption,ls='dotted',color='dimgray')
    plotlimits(path,dict(XENON10='checked-XENON10-F1Q2.txt'),**goption,ls='dotted',color='blueviolet')
    plotlimits(path,dict({'CDMS HVeV':'checked-CDMS-F1Q2.txt'}),**goption,ls='dotted',color='orange')
    plotlimits(path,dict({'protoSENSEI @ MINOS':'checked-SENSEI-F1Q2.txt'}),**goption,ls='dotted',color='seagreen')
    plotlimits(path,dict({'DAMIC @ SNOLAB':'checked-DAMIC2019-F1Q2.txt'}),**goption,ls='dashed',color='royalblue')
    plotlimits(path,dict({'SENSEI @ MINOS':'Sensei-DMe-FDM1q2.txt'}),**goption,ls='dashed',color='darkgreen')
    
    
    path = 'LimitsPRL'
    
    limitPRLF000,m = loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,1))    
    limitPRLF015,_ = loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,2))
    limitPRLF030,_ = loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,3))
    limitPRLRouv,_= loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,4))
    
    officiallimit, = plt.plot(m,limitPRLF015,label='EDELWEISS',ls='solid',color='red',alpha=0.6,lw=3)
    #uncertaintyregion = plt.fill_between(m,limitPRLF000,limitPRLF030,alpha=0.5,color='red')
    #linear, = plt.plot(m,limitPRLRouv,ls='dashed',color='red')
    goption = dict(lw=1,marker="",alpha=1)
    path = 'limitsallexperiments'
    #plotlimits(path,dict({'Freeze-in Model':'freezein.txt'}),**goption,ls='dashed',color='black')
    
    #plt.legend(loc='upper right')
    
    
    path = '.'
    experiments = dict({'CRYOSEL : 1 kg.d':'DMES-1kg.days.txt'
                        })
    goption = dict(lw=3,marker="",alpha=1)
    plotlimit(path,'CRYOSEL: 1 kg.d','DMES-1kg.days.txt',rescalefactor=1e-36,**goption,color='red')
    #plotlimits(path,experiments,**goption,color='red',rescalefactor=1e-36)
    
    #handles1, labels1 = plt.gca().get_legend_handles_labels()
    #plt.legend(handles=handles1+[linear,(officiallimit,uncertaintyregion)],labels=labels1+["linear ionization model","EDELWEISS"])
    plt.legend(loc='upper right')
    plt.savefig("ExclusionPlotElsa_Cryosel.pdf")
    plt.show()
    

def PRLprogDP(): 
    xmin,xmax = 1,40
    ymin,ymax = 5e-16,4e-11
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xscale('log')
    ax.set_xlabel('$\mathrm{m_{V}}$ [eV/$\mathrm{c}^{2}$]')
    #ax.yaxis.tick_right()
    ax.set_yscale('log')
    ax.set_ylabel('$\kappa$')
    #plt.grid(True,"both")
    
    ax.tick_params(axis="y", which='both',direction='in')
    ax.tick_params(axis="x", which='both',direction='in')
        
    ylocator = ticker.LogLocator(base=10.0, numticks=25)
    plt.gca().yaxis.set_major_locator(ylocator)
    
    xlocator   = ticker.LogLocator(base=10,subs=(10.,0.2),numdecs=1,numticks=None)
    xformatter = ticker.LogFormatter(base=10.0, labelOnlyBase=False, minor_thresholds=None, linthresh=None)
    plt.gca().xaxis.set_major_locator(xlocator)
    plt.gca().xaxis.set_major_formatter(xformatter)
    #plt.gca().xaxis.set_minor_locator( ticker.LogLocator(base=1,subs=(1.0,),numdecs=1,numticks=None))
    #plt.gca().xaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(200,0.1)))
    #########################
    path = 'limitsallexperiments'
    goption = dict(lw=1.5,marker="",alpha=0.4)
    
    plotlimits(path,dict({'Solar constraints':'checked-SOLAR-DP.txt'}),**goption,ls='dotted',color='dimgray')
    plotlimits(path,dict(XENON10='checked-XENON10-DP.txt'),**goption,ls='dotted',color='blueviolet')
    plotlimits(path,dict({'CDMS HVeV':'checked-CDMS-DP.txt'}),**goption,ls='solid',color='orange')
    plotlimits(path,dict({'protoSENSEI @ MINOS':'checked-SENSEI-DP.txt'}),**goption,ls='dotted',color='seagreen')
    plotlimits(path,dict({'DAMIC @ SNOLAB':'checked-DAMIC2019-DP.txt'}),**goption,ls='dashed',color='royalblue')
    plotlimits(path,dict({'SENSEI @ MINOS':'SENSEI@MINOS-DP.txt'}),**goption,ls='dashed',color='darkgreen')
    
    path = 'LimitsPRL'
    
    limitPRLF000,m = loadlimit(path,"EDELWEISS_Limit_DarkPhoton.txt",rescalefactor=1,usecols=(0,1))    
    limitPRLF015,_ = loadlimit(path,"EDELWEISS_Limit_DarkPhoton.txt",rescalefactor=1,usecols=(0,2))
    limitPRLF030,_ = loadlimit(path,"EDELWEISS_Limit_DarkPhoton.txt",rescalefactor=1,usecols=(0,3))
    limitPRLRouv,_= loadlimit(path,"EDELWEISS_Limit_DarkPhoton.txt",rescalefactor=1,usecols=(0,4))
    
    officiallimit, = plt.plot(m,limitPRLF015,ls='solid',color='red')
    uncertaintyregion = plt.fill_between(m,limitPRLF000,limitPRLF030,alpha=0.5,color='red')
    linear, = plt.plot(m,limitPRLRouv,ls='dashed',color='red')
    
    plt.legend(loc='lower left')
    handles1, labels1 = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles1+[linear,(officiallimit,uncertaintyregion)],labels=labels1+["linear ionization model","EDELWEISS"])
    
    plt.savefig("FigPRLprogDP.pdf")
    plt.show()
    
def PRLprogF1(): 
    xmin,xmax = 0.5,100
    ymin,ymax = 1e-35,1e-28
    #xmin,xmax = 0.4,1000
    #ymin,ymax = 1e-40,1e-27
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xscale('log')
    ax.set_xlabel('$\mathrm{m_{DM}}$ [MeV/$\mathrm{c}^{2}$]')
    #ax.yaxis.tick_right()
    ax.set_yscale('log')
    ax.set_ylabel('$\overline{\sigma_e}\quad [\mathrm{cm}^{2}]$')
    #plt.grid(True,"both")
    
    ax.text(0.15, 0.88, 'DM-electron scattering \n$\mathrm{F_{DM}} = 1$',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=10)
    
    ax.tick_params(axis="y", which='both',direction='in')
    ax.tick_params(axis="x", which='both',direction='in')
    
    ylocator = ticker.LogLocator(base=10.0, numticks=25)
    plt.gca().yaxis.set_major_locator(ylocator)
    
    xlocator   = ticker.LogLocator(base=10,subs=(10.,0.2),numdecs=1,numticks=None)
    xformatter = ticker.LogFormatter(base=10.0, labelOnlyBase=False, minor_thresholds=None, linthresh=None)
    plt.gca().xaxis.set_major_locator(xlocator)
    plt.gca().xaxis.set_major_formatter(xformatter)
    #plt.gca().xaxis.set_minor_locator( ticker.LogLocator(base=1,subs=(1.0,),numdecs=1,numticks=None))
    #plt.gca().xaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(200,0.1)))
    #########################
    path = 'limitsallexperiments'
    goption = dict(lw=1.5,marker="",alpha=1)
    
    plotlimits(path,dict({"DarkSide50":"checked-DARKSIDE50-F1.txt"}),**goption,ls='dotted',color='dimgray')
    plotlimits(path,dict(XENON10='checked-XENON10-F1.txt'),**goption,ls='dotted',color='blueviolet')
    plotlimits(path,dict({'CDMS HVeV':'checked-CDMS-F1.txt'}),**goption,ls='solid',color='orange')
    plotlimits(path,dict({'protoSENSEI @ MINOS':'checked-SENSEI-F1.txt'}),**goption,ls='dotted',color='seagreen')
    plotlimits(path,dict({'DAMIC @ SNOLAB':'checked-DAMIC2019-F1.txt'}),**goption,ls='dashed',color='royalblue')
    plotlimits(path,dict({'SENSEI @ MINOS':'SENSEI@MINOS-F1.txt'}),**goption,ls='dashed',color='darkgreen')
    
    path = 'LimitsPRL'
    
    limitPRLF000,m = loadlimit(path,"EDELWEISS_Limit_FDM_heavy.txt",rescalefactor=1,usecols=(0,1))    
    limitPRLF015,_ = loadlimit(path,"EDELWEISS_Limit_FDM_heavy.txt",rescalefactor=1,usecols=(0,2))
    limitPRLF030,_ = loadlimit(path,"EDELWEISS_Limit_FDM_heavy.txt",rescalefactor=1,usecols=(0,3))
    limitPRLRouv,_= loadlimit(path,"EDELWEISS_Limit_FDM_heavy.txt",rescalefactor=1,usecols=(0,4))
    
    officiallimit, = plt.plot(m,limitPRLF015,ls='solid',color='red')
    uncertaintyregion = plt.fill_between(m,limitPRLF000,limitPRLF030,alpha=0.5,color='red')
    linear, = plt.plot(m,limitPRLRouv,ls='dashed',color='red')
    
    plt.legend(loc='lower left')
    handles1, labels1 = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles1+[linear,(officiallimit,uncertaintyregion)],labels=labels1+["linear ionization model","EDELWEISS"])
    
    plt.savefig("FigPRLprogF1.pdf")
    plt.show()
    
def PRLprogF1Q2(): 
    xmin,xmax = 0.5,100
    ymin,ymax = 1e-34,1e-28
    #xmin,xmax = 0.4,1000
    #ymin,ymax = 1e-37,1e-28
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xscale('log')
    ax.set_xlabel('$\mathrm{m_{DM}}$ [MeV/$\mathrm{c}^{2}$]')
    #ax.yaxis.tick_right()
    ax.set_yscale('log')
    ax.set_ylabel('$\overline{\sigma_e}\quad [\mathrm{cm}^{2}]$')
    #plt.grid(True,"both")
    ax.text(0.15, 0.88, 'DM-electron scattering \n$\mathrm{F_{DM}} \propto 1/\mathrm{q}^{2}$',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=10)
    
    ax.tick_params(axis="y", which='both',direction='in')
    ax.tick_params(axis="x", which='both',direction='in')
    
    ylocator = ticker.LogLocator(base=10.0, numticks=25)
    plt.gca().yaxis.set_major_locator(ylocator)
    
    xlocator   = ticker.LogLocator(base=10,subs=(10.,0.2),numdecs=1,numticks=None)
    xformatter = ticker.LogFormatter(base=10.0, labelOnlyBase=False, minor_thresholds=None, linthresh=None)
    plt.gca().xaxis.set_major_locator(xlocator)
    plt.gca().xaxis.set_major_formatter(xformatter)
    #plt.gca().xaxis.set_minor_locator( ticker.LogLocator(base=1,subs=(1.0,),numdecs=1,numticks=None))
    #plt.gca().xaxis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(200,0.1)))
    #########################
    path = 'limitsallexperiments'
    goption = dict(lw=1.5,marker="",alpha=1)
    
    plotlimits(path,dict({"DarkSide50":"checked-DARKSIDE50-F1Q2.txt"}),**goption,ls='dotted',color='dimgray')
    plotlimits(path,dict(XENON10='checked-XENON10-F1Q2.txt'),**goption,ls='dotted',color='blueviolet')
    plotlimits(path,dict({'CDMS HVeV':'checked-CDMS-F1Q2.txt'}),**goption,ls='solid',color='orange')
    plotlimits(path,dict({'protoSENSEI @ MINOS':'checked-SENSEI-F1Q2.txt'}),**goption,ls='dotted',color='seagreen')
    plotlimits(path,dict({'DAMIC @ SNOLAB':'checked-DAMIC2019-F1Q2.txt'}),**goption,ls='dashed',color='royalblue')
    plotlimits(path,dict({'SENSEI @ MINOS':'SENSEI@MINOS-F1Q2.txt'}),**goption,ls='dashed',color='darkgreen')
    
    path = 'LimitsPRL'
    
    limitPRLF000,m = loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,1))    
    limitPRLF015,_ = loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,2))
    limitPRLF030,_ = loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,3))
    limitPRLRouv,_= loadlimit(path,"EDELWEISS_Limit_FDM_light.txt",rescalefactor=1,usecols=(0,4))
    
    officiallimit, = plt.plot(m,limitPRLF015,ls='solid',color='red')
    uncertaintyregion = plt.fill_between(m,limitPRLF000,limitPRLF030,alpha=0.5,color='red')
    linear, = plt.plot(m,limitPRLRouv,ls='dashed',color='red')
    
    plt.legend(loc='lower left')
    handles1, labels1 = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles1+[linear,(officiallimit,uncertaintyregion)],labels=labels1+["linear ionization model","EDELWEISS"])
    
    plt.savefig("FigPRLprogF1Q2.pdf")
    plt.show()
 
#progWIMP()
#progDP()
progDMES()

#PRLprogDP()
#PRLprogF1()
#PRLprogF1Q2()
