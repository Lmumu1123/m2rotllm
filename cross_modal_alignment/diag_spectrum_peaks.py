"""雷达径向位移谱：列出所有显著峰，验证 26Hz 是否为转动基频。"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

DATA_ROOT = os.environ.get(
    "ALIGNMENT_DATA_DIR", os.path.dirname(os.path.abspath(__file__))
)
fp = os.environ.get("ALIGNMENT_FONT")
if fp and os.path.exists(fp):
    fm.fontManager.addfont(fp)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

BIN = os.environ.get(
    "RADAR_BIN", os.path.join(DATA_ROOT, "20260807-153437-1786088077053250.bin")
)
OUT = os.environ.get("ALIGNMENT_OUT_DIR", os.path.join(DATA_ROOT, "_rd_test_out"))
C, FC, LAM = 3e8, 77e9, 3e8/77e9
FS_LOOP = 1000.0/1.5; RX=4; NADC=512; CPL=3; TB=13

mm = np.memmap(BIN, dtype=np.int16, mode="r")
spb = RX*NADC; nch = len(mm)//(spb*2); nloop = nch//CPL
win = np.hanning(NADC).astype(np.float32)
twid = np.exp(-2j*np.pi*TB*np.arange(NADC)/NADC)
block=8192; zall=np.zeros((nloop,RX),dtype=np.complex128)
for lo in range(0,nloop,block):
    hi=min(lo+block,nloop); s=lo*CPL*spb
    a=np.asarray(mm[s:s+(hi-lo)*CPL*spb],dtype=np.float32).reshape((hi-lo)*CPL,RX,NADC)
    c0=a[0::CPL]; zall[lo:hi]=np.einsum("lra,a->lr",c0*win,twid)
ref=np.angle(zall[:1000].mean(axis=0)); zb=(zall*np.exp(-1j*ref)[None,:]).mean(axis=1)
disp=np.unwrap(np.angle(zb))*LAM/(4*np.pi)*1e6

# 去线性趋势，整段 FFT
sig=disp-np.polyval(np.polyfit(np.arange(len(disp))/FS_LOOP,disp,1),np.arange(len(disp))/FS_LOOP)
NFFT=1<<int(np.ceil(np.log2(len(sig)))); NFFT=min(NFFT,1<<20)
f=np.fft.rfftfreq(NFFT,1/FS_LOOP)
seg=np.zeros(NFFT); seg[:len(sig)]=sig[:NFFT]; seg[:len(sig)]*=np.hanning(len(sig))
spec=np.abs(np.fft.rfft(seg))*2/NFFT

# 去除直流附近(<1Hz)后找峰
spec_hf=spec.copy(); spec_hf[:int(1.0/(f[1]-f[0]))]=0
# 找所有局部极大，按幅度排序（不依赖 scipy）
def find_local_peaks(x, min_dist):
    """返回局部极大索引，要求与已选峰距离>=min_dist"""
    cand=np.where((x[1:-1]>x[:-2])&(x[1:-1]>x[2:]))[0]+1
    order=np.argsort(x[cand])[::-1]
    picked=[]
    for i in cand[order]:
        if all(abs(i-p)>=min_dist for p in picked):
            picked.append(i)
    return np.array(picked)

thr=np.median(spec_hf)*3
min_dist=int(2.0/(f[1]-f[0]))  # 峰之间至少隔2Hz
idx=find_local_peaks(spec_hf,min_dist)
idx=idx[spec_hf[idx]>=thr]
order=np.argsort(spec_hf[idx])[::-1]
print("=== 雷达位移谱主要峰（去直流后，按幅度排序）===")
print(f"{'rank':<5}{'freq(Hz)':<12}{'amp(um)':<12}")
for r,i in enumerate(order[:12]):
    print(f"{r+1:<5}{f[idx[i]]:<12.2f}{spec[idx[i]]:<12.3f}")

# 谐波/分谐波关系判断
peaks_f=[f[idx[i]] for i in order[:12]]
peaks_a=[spec[idx[i]] for i in order[:12]]
print("\n=== 谐波关系自检 ===")
base=peaks_f[0]
for j,pf in enumerate(peaks_f[:6]):
    ratio=pf/base
    print(f"  峰{j+1} {pf:.2f}Hz / 基{base:.2f}Hz = {ratio:.2f}"
          f"{'  <- 接近整数倍' if abs(ratio-round(ratio))<0.15 else ''}")

# 画图标注所有峰
fig,ax=plt.subplots(figsize=(13,6))
ax.semilogy(f,spec,lw=0.6)
ax.set_xlim(0,120)
thr=np.median(spec_hf)*3
ax.axhline(thr,color="green",ls=":",label=f"峰检测阈值={thr:.3f}um")
for r,i in enumerate(order[:8]):
    ax.annotate(f"{f[idx[i]]:.1f}Hz\n{spec[idx[i]]:.1f}um",
        xy=(f[idx[i]],spec[idx[i]]),
        xytext=(f[idx[i]]+2,spec[idx[i]]*1.5),
        arrowprops=dict(arrowstyle="->",color="red"),fontsize=8,color="red")
ax.axvline(26.0,color="orange",ls="--",lw=1,label="26Hz（疑似转动基频）")
# 电机工频关系：中国50Hz电网，2极电机同步转速3000rpm=50Hz，4极=1500rpm=25Hz
ax.axvline(25.0,color="purple",ls="--",lw=0.8,alpha=0.6,label="25Hz（4极电机同步转速）")
ax.set_xlabel("频率(Hz)"); ax.set_ylabel("位移幅值(um, log)")
ax.set_title("图7: 雷达位移谱全部峰 — 用谐波倍数关系判断哪个是转动基频")
ax.legend(fontsize=8); ax.grid(alpha=0.3,which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig7_spectrum_peaks.png"),dpi=110)
print(f"\nsaved -> {OUT}/fig7_spectrum_peaks.png")
