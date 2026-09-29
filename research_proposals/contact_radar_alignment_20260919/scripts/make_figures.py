"""Reproducible vector/raster research diagrams. All figures are schematic."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "svg.fonttype": "none"})
C = {"blue":"#2875B8", "teal":"#12877D", "orange":"#D78428", "purple":"#805BAA",
     "red":"#BF4A50", "ink":"#24364B", "muted":"#647589", "gray":"#E9EEF3"}

def box(ax, xy, w, h, title, body="", color="blue", fs=11):
    col=C.get(color,color)
    ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle="round,pad=0.012,rounding_size=0.015",
                              fc=col+"13",ec=col,lw=1.4))
    x,y=xy
    if body:
        ax.text(x+w/2,y+h*.72,title,ha="center",va="center",weight="bold",color=C["ink"],fontsize=fs)
        ax.text(x+w/2,y+h*.33,body,ha="center",va="center",color=C["ink"],fontsize=fs-1,linespacing=1.5)
    else:
        ax.text(x+w/2,y+h/2,title,ha="center",va="center",weight="bold",color=C["ink"],fontsize=fs)

def arrow(ax,a,b,color="muted",style="-",rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,lw=1.4,
                               color=C.get(color,color),linestyle=style,
                               connectionstyle=f"arc3,rad={rad}"))

def save(fig,name):
    for ext in ("png","svg","pdf"):
        fig.savefig(OUT/f"{name}.{ext}",dpi=190,bbox_inches="tight")
    plt.close(fig)

def framework():
    fig,ax=plt.subplots(figsize=(14,8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.02,.97,"Contact-to-radar transfer with a fixed diagnostic interface",fontsize=18,weight="bold",color=C["ink"])
    ax.text(.02,.91,"PROPOSED DESIGN  /  training uses paired sensors; deployment uses radar",fontsize=11,color=C["muted"])
    box(ax,(.02,.69),.2,.14,"Accelerometer","XYZ packets + real times\n5 kHz / 4096 samples", "blue")
    box(ax,(.29,.69),.2,.14,"Shared-band teacher","Adapt once, then freeze\nhealth reference policy", "blue")
    box(ax,(.57,.69),.18,.14,"Teacher targets","Diagnostic hidden state\nclass probabilities", "blue")
    box(ax,(.02,.34),.2,.16,"AWR1843 + DCA1000","Complex 4RX data\nchirps + frame-gap mask", "teal")
    box(ax,(.29,.34),.2,.16,"Radar encoder","IQ / cyclic-order / stats\nquality-based spatial pooling", "teal")
    box(ax,(.57,.34),.18,.16,"Student interface","128-D hidden or\n128 x 47 feature map", "teal")
    box(ax,(.8,.34),.18,.16,"Frozen downstream","Diagnosis + calibrated\nconfidence / optional LLM", "purple",10)
    box(ax,(.26,.075),.49,.13,"Physical support + uncertain correspondence","Bandwidth and SNR masks  |  constrained clock map  |  match / skip posterior", "orange",11)
    arrow(ax,(.225,.76),(.28,.76));arrow(ax,(.495,.76),(.56,.76))
    arrow(ax,(.225,.42),(.28,.42));arrow(ax,(.495,.42),(.56,.42));arrow(ax,(.756,.42),(.79,.42))
    arrow(ax,(.66,.68),(.66,.515),"blue","--")
    ax.text(.675,.595,"weighted feature KD\nlogit KD + contrast",fontsize=10,va="center",color=C["blue"])
    arrow(ax,(.385,.21),(.385,.325),"orange","--")
    ax.plot([.22,.245,.245],[.69,.65,.25],color=C["orange"],linestyle="--",lw=1.4)
    arrow(ax,(.245,.25),(.27,.215),"orange","--")
    arrow(ax,(.12,.33),(.25,.145),"orange","--",rad=.1)
    ax.text(.8,.77,"TRAINING ONLY",color=C["blue"],fontsize=11,weight="bold")
    ax.text(.8,.24,"RADAR-ONLY INFERENCE",color=C["teal"],fontsize=10,weight="bold")
    ax.text(.02,.015,"Schematic only. No measured performance is implied.",fontsize=9,color=C["muted"])
    save(fig,"01_framework")

def alignment():
    fig=plt.figure(figsize=(13,8));gs=fig.add_gridspec(3,1,height_ratios=[1,1,2.5],hspace=.5)
    ax=fig.add_subplot(gs[0]);a2=fig.add_subplot(gs[1]);a3=fig.add_subplot(gs[2])
    t=np.arange(0,4,.0005);mask=(np.arange(t.size)%200)<192
    signal=np.sin(2*np.pi*(4*t+.45*t*t))*(.45+.2*np.sin(2*np.pi*.35*t))
    ax.plot(t,np.where(mask,signal,np.nan),c=C["teal"],lw=1)
    for m in range(40):ax.axvspan(m*.1+.096,(m+1)*.1,color=C["red"],alpha=.22,lw=0)
    ax.set(title="Radar: actual 2 kHz chirp grid, 8 missing slots per 100 ms",xlim=(0,4),ylabel="illustrative signal")
    ta=np.arange(0,4,.0002);ma=np.zeros(len(ta),bool)
    for start in [.12,1.12,2.12,3.12]:ma|=(ta>=start)&(ta<start+.8192)
    a2.plot(ta,np.where(ma,np.sin(2*np.pi*(4*(ta-.12)+.45*(ta-.12)**2)),np.nan),c=C["blue"],lw=1)
    a2.set(title="Accelerometer: 4096-point packets; displayed packet gaps are illustrative",xlim=(0,4),ylabel="illustrative signal",xlabel="Time (s)")
    tr=np.linspace(0,4,130);at=np.linspace(0,4.25,138);g=.12+1.0002*tr
    q=np.exp(-.5*((at[None,:]-g[:,None])/.055)**2)
    valid=np.zeros(at.size,bool)
    for start in [.12,1.12,2.12,3.12]:valid|=(at>=start)&(at<start+.8192)
    q[:,~valid]=np.nan
    im=a3.imshow(q.T,origin="lower",aspect="auto",extent=(0,4,0,4.25),cmap="YlGnBu",vmin=0,vmax=1)
    a3.plot(tr,g,c=C["orange"],ls="--",lw=1.5,label="constrained time map (illustration)")
    a3.legend(loc="upper left",fontsize=9)
    a3.set(xlabel="Radar descriptor time (s)",ylabel="Contact descriptor time (s)",title="Conceptual match posterior: retain gaps and ambiguity")
    fig.colorbar(im,ax=a3,pad=.015,label="Relative match weight (constructed)")
    fig.suptitle("Clock alignment is separate from mechanical waveform transfer",fontsize=17,weight="bold",y=1.01)
    save(fig,"02_alignment")

def observability():
    f=np.linspace(60,800,1200);alpha=40.
    ps=lambda x: .3+np.exp(-((x-250)/100)**2)+.7*np.exp(-((x-580)/110)**2)
    h1=lambda x: .5+1.5*np.exp(-((x-220)/70)**2)
    h2=lambda x: (.65+2.1*np.exp(-((x-540)/80)**2))*(1-.999*np.exp(-((x-360)/22)**2))
    gamma=.2+.6*np.exp(-((f-430)/180)**2)
    noise=.03
    def observed(h):
        s1=h(f+alpha/2)**2*ps(f+alpha/2);s2=h(f-alpha/2)**2*ps(f-alpha/2)
        return gamma*np.sqrt(s1*s2/((s1+noise)*(s2+noise)))
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    axs[0,0].plot(f,h1(f),c=C["blue"],label="path A");axs[0,0].plot(f,h2(f),c=C["orange"],label="path B with deep notch")
    axs[0,0].set(title="A. Constructed transfer magnitudes",ylabel="Gain")
    axs[0,1].plot(f,h1(f)**2*ps(f),c=C["blue"],label="path A PSD")
    axs[0,1].plot(f,h2(f)**2*ps(f),c=C["orange"],label="path B PSD")
    axs[0,1].set(title="B. Ordinary spectra change with the path",ylabel="Relative power")
    axs[1,0].plot(f,gamma,c=C["blue"],lw=3,label="path A, ideal")
    axs[1,0].plot(f,gamma,c=C["orange"],ls="--",lw=2,label="path B, ideal")
    axs[1,0].set(title="C. Ideal cyclic coherence magnitudes coincide",ylabel=r"$|\gamma^{\alpha}(f)|$",ylim=(0,1))
    axs[1,1].plot(f,observed(h1),c=C["blue"],label="path A + noise")
    axs[1,1].plot(f,observed(h2),c=C["orange"],label="path B + noise")
    axs[1,1].axvspan(310,410,color=C["red"],alpha=.13,label="low-observability region")
    axs[1,1].set(title="D. Additive noise breaks ideal invariance",ylabel=r"$|\gamma^{\alpha}(f)|$",ylim=(0,1))
    for ax in axs.flat:ax.set_xlabel("Carrier vibration frequency (Hz)");ax.grid(alpha=.16);ax.legend(fontsize=9)
    fig.suptitle("Analytical illustration: one excitation, LTI paths, cycle frequency 40 Hz\nConstructed curves, not measured data or estimator results",fontsize=14,weight="bold")
    save(fig,"03_observability")

def protocols():
    fig,axs=plt.subplots(1,2,figsize=(13,6),gridspec_kw={"width_ratios":[1.15,1]})
    ax=axs[0]
    # Example joint holdout: no bearing or run is shared across partitions.
    grid=np.array([[0,0,3],[1,1,3],[3,3,2]])
    from matplotlib.colors import ListedColormap
    ax.imshow(grid,cmap=ListedColormap([C["blue"],C["orange"],C["teal"],"#F2F4F6"]),vmin=0,vmax=3)
    labels=[['TRAIN','TRAIN','reserved'],['VALIDATE','VALIDATE','reserved'],['reserved','reserved','TEST']]
    for i in range(3):
        for j in range(3):ax.text(j,i,labels[i][j],ha="center",va="center",color="white" if grid[i,j]<3 else C["muted"],weight="bold",fontsize=11)
    ax.set(xticks=range(3),xticklabels=["E1","E2","E3 (unseen)"],yticks=range(3),yticklabels=["Train specimens","Validation specimens","Test specimens"],xlabel="Environment",title="Strong protocol if a third environment is available")
    ax.spines[:].set_visible(False)
    ax2=axs[1];ax2.set(xlim=(0,1),ylim=(0,1));ax2.axis("off")
    box(ax2,(.05,.73),.9,.2,"Current two-environment protocol","Train: E1 / train specimens\nValidate: E1 / separate specimens\nTest: E2 / held-out specimens","blue",11)
    box(ax2,(.05,.43),.9,.2,"External motor cases","One healthy + one faulty motor\nDevice identity is confounded with health\nReport case results, not population accuracy","orange",11)
    box(ax2,(.05,.13),.9,.2,"Every partition keeps whole runs","No overlapping windows across splits\nTrain-only PCA / references / thresholds\nCalibration P0 / P1 / P2 reported separately","teal",11)
    fig.suptitle("Evaluate independent physical evidence, not the number of windows",fontsize=16,weight="bold",y=1.01)
    fig.tight_layout()
    save(fig,"04_protocols")

def testbed():
    fig,ax=plt.subplots(figsize=(14,6.5));ax.set(xlim=(0,14),ylim=(0,6.5));ax.axis("off")
    ax.text(.3,6.15,"Existing testbed: a remote measurement point for bearing faults",fontsize=17,weight="bold",color=C["ink"])
    ax.text(.3,5.7,"Conceptual geometry only; no additional synchronization or measurement device",fontsize=11,color=C["muted"])
    ax.add_patch(Rectangle((1,1.25),10,.3,fc="#A7B5C4",ec=C["muted"]))
    ax.add_patch(Rectangle((1.7,1.55),1.1,1.25,fc=C["blue"]+"22",ec=C["blue"],lw=2))
    ax.add_patch(Circle((2.25,2.8),.7,fc="white",ec=C["blue"],lw=3))
    ax.add_patch(Circle((2.25,2.8),.3,fc="#D3E0ED",ec=C["blue"],lw=2))
    for theta in np.linspace(0,2*np.pi,8,endpoint=False):
        ax.add_patch(Circle((2.25+.49*np.cos(theta),2.8+.49*np.sin(theta)),.09,fc=C["blue"]))
    ax.add_patch(Circle((2.67,3.1),.13,fc=C["red"],ec="white"))
    ax.plot([2.95,5.8],[2.8,2.8],color=C["muted"],lw=10,solid_capstyle="round")
    ax.add_patch(Rectangle((4.25,2.48),.55,.64,fc=C["orange"]+"44",ec=C["orange"],lw=2))
    ax.add_patch(FancyBboxPatch((5.8,1.9),3.4,1.8,boxstyle="round,pad=.06,rounding_size=.2",fc=C["teal"]+"16",ec=C["teal"],lw=2))
    for x in np.linspace(6.15,8.85,10):ax.plot([x,x],[2.03,3.55],c=C["teal"],alpha=.35,lw=2)
    ax.text(7.5,2.75,"Fixed motor\nhousing",ha="center",va="center",fontsize=13,color=C["ink"],weight="bold")
    ax.add_patch(Rectangle((6.5,3.68),.35,.25,fc=C["blue"],ec="white"))
    ax.annotate("XYZ accelerometer",xy=(6.68,3.87),xytext=(5.1,4.9),color=C["blue"],fontsize=12,arrowprops={"arrowstyle":"->","color":C["blue"]})
    ax.add_patch(Circle((8.45,3.45),.16,fc=C["orange"],ec="white"))
    ax.text(8.65,3.86,"Radar ROI",color=C["orange"],fontsize=11)
    ax.add_patch(Polygon([(11.5,4.5),(8.0,3.4),(8.8,3.1)],fc=C["orange"],alpha=.14,ec=None))
    ax.add_patch(FancyBboxPatch((11.25,4.15),1.85,.85,boxstyle="round,pad=.08",fc=C["orange"]+"20",ec=C["orange"],lw=2))
    ax.text(12.17,4.58,"AWR1843\n+ DCA1000",ha="center",va="center",color=C["ink"],fontsize=11)
    ax.add_patch(FancyArrowPatch((11.2,4.5),(8.64,3.5),arrowstyle="-|>",color=C["orange"],lw=2,mutation_scale=16))
    ax.text(2.25,.65,"Replaceable fault bearing",ha="center",fontsize=12,color=C["blue"])
    ax.text(7.5,.65,"Both modalities measure at the motor",ha="center",fontsize=12,color=C["teal"])
    ax.add_patch(FancyArrowPatch((3.15,2.12),(5.55,2.12),arrowstyle="-|>",color=C["red"],lw=2,mutation_scale=16))
    ax.text(4.35,1.75,"Mechanical propagation",ha="center",fontsize=10,color=C["red"])
    ax.text(.3,.1,"Fault label comes from the bearing specimen. The motor housing is a filtered, weaker observation.",fontsize=11,color=C["muted"])
    save(fig,"05_testbed")

def radar_timing():
    fig,axs=plt.subplots(3,1,figsize=(13,9),constrained_layout=True)
    t=np.arange(2)[:,None]*100+np.arange(192)[None,:]*.5
    ax=axs[0]
    ax.vlines(t.ravel(),0,1,color=C["teal"],lw=.65)
    ax.axvspan(96,100,color=C["red"],alpha=.2,label="4 ms nominal frame idle")
    ax.axvspan(196,200,color=C["red"],alpha=.2)
    ax.set(xlim=(0,200),ylim=(0,1.15),yticks=[],xlabel="Time (ms)",title="A. Chirp starts: 192 x 0.5 ms per frame, repeated every 100 ms")
    ax.legend(loc="upper right")
    ax=axs[1];tt=np.arange(92,104,.5);valid=(tt%100)<96
    ax.scatter(tt[valid],np.ones(valid.sum()),s=40,c=C["teal"],label="acquired slow-time sample")
    ax.scatter(tt[~valid],np.ones((~valid).sum()),s=55,c=C["red"],marker="x",label="missing grid position")
    ax.annotate("",xy=(100,1.16),xytext=(95.5,1.16),arrowprops={"arrowstyle":"<->","color":C["muted"]})
    ax.text(97.75,1.27,"4.5 ms between chirp starts",ha="center",fontsize=10,color=C["muted"])
    ax.set(xlim=(92,104),ylim=(.6,1.4),yticks=[],xlabel="Time (ms)",title="B. Eight missing positions; do not concatenate frames as a uniform sequence")
    ax.legend(loc="lower left",ncol=2)
    ax=axs[2];r=np.linspace(.1,2.2,300);fb=2*49.99e12*r/299792458/1000
    ax.plot(r,fb,c=C["blue"],lw=2,label="beat frequency from range")
    ax.axhline(175,c=C["orange"],ls="--",label="HPF1 corner (code 0)")
    ax.axhline(350,c=C["red"],ls="--",label="HPF2 corner (code 0)")
    ax.set(xlabel="Range (m)",ylabel="IF beat frequency (kHz)",title="C. IF high-pass filtering can attenuate close-range returns; measure SNR vs range")
    ax.legend(loc="upper left",fontsize=9);ax.grid(alpha=.15)
    fig.suptitle("Configuration arithmetic for AWR1843 + DCA1000\nNot a measurement of actual timing, phase coherence, or filter attenuation",fontsize=15,weight="bold")
    save(fig,"06_radar_timing")

if __name__=="__main__":
    for fn in [framework, alignment, observability, protocols, testbed, radar_timing]:
        fn()
    print(f"Saved 6 diagrams in PNG, SVG and PDF to {OUT}")
