import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
rows=list(csv.DictReader(open("results/confirm200_ft_vs_zs.csv")))
Z=np.array([float(r["dice_zs"]) for r in rows])
F=np.array([float(r["dice_ft"]) for r in rows])
d=F-Z; N=len(d)
win=(d>1e-4).sum(); lose=(d<-1e-4).sum(); tie=N-win-lose
w_stat,w_p=stats.wilcoxon(F,Z)           # paired
t_stat,t_p=stats.ttest_rel(F,Z)
fig=plt.figure(figsize=(16,9)); fig.suptitle(
    f"Zero-shot vs Fine-tuned (epoch6)  |  N={N} ảnh ngẫu nhiên  |  box=cc-bbox, GT=curated",
    fontsize=15,fontweight="bold")
gs=fig.add_gridspec(2,3,hspace=0.32,wspace=0.28)
# 1 scatter
ax=fig.add_subplot(gs[0,0])
ax.scatter(Z,F,s=14,alpha=0.5,c=np.where(d>=0,"green","crimson"))
lim=[min(Z.min(),F.min())-0.02,1.005]
ax.plot(lim,lim,"k--",lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Dice ZERO-SHOT"); ax.set_ylabel("Dice FINE-TUNED")
ax.set_title("Mỗi điểm = 1 ảnh\n(dưới đường = FT kém hơn)")
ax.text(0.05,0.92,f"trên={win} (FT thắng)\nduoi={lose} (FT thua)",transform=ax.transAxes,
        va="top",fontsize=9,bbox=dict(boxstyle="round",fc="white",alpha=0.7))
# 2 histogram delta
ax=fig.add_subplot(gs[0,1])
ax.hist(d,bins=40,color="steelblue",edgecolor="white")
ax.axvline(0,color="k",ls="--",lw=1); ax.axvline(d.mean(),color="red",lw=2,label=f"mean={d.mean():+.4f}")
ax.set_xlabel("delta = Dice_FT - Dice_ZS"); ax.set_ylabel("số ảnh")
ax.set_title("Phân phối chênh lệch\n(<0 = FT kém hơn)"); ax.legend(fontsize=9)
# 3 sorted delta bars
ax=fig.add_subplot(gs[0,2])
ds=np.sort(d)
ax.bar(range(N),ds,color=np.where(ds>=0,"green","crimson"),width=1.0)
ax.axhline(0,color="k",lw=0.8)
ax.set_xlabel("ảnh (sắp theo delta)"); ax.set_ylabel("delta")
ax.set_title(f"Delta từng ảnh\n(đỏ {lose} > xanh {win})")
# 4 box/violin
ax=fig.add_subplot(gs[1,0])
parts=ax.violinplot([Z,F],showmedians=True)
ax.set_xticks([1,2]); ax.set_xticklabels(["Zero-shot","Fine-tuned"])
ax.set_ylabel("Dice"); ax.set_title("Phân phối Dice")
for i,arr in enumerate([Z,F],1):
    ax.text(i,np.median(arr),f" med={np.median(arr):.3f}",fontsize=9,va="center")
# 5 win/lose bar
ax=fig.add_subplot(gs[1,1])
b=ax.bar(["FT thắng","hòa","FT thua"],[win,tie,lose],color=["green","gray","crimson"])
for r in b: ax.text(r.get_x()+r.get_width()/2,r.get_height()+1,int(r.get_height()),ha="center",fontsize=11)
ax.set_ylabel("số ảnh"); ax.set_title("Đối đầu từng ảnh")
# 6 text verdict
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
verdict = "ZERO-SHOT TỐT HƠN" if (d.mean()<0 and w_p<0.05) else ("FT tốt hơn" if d.mean()>0 and w_p<0.05 else "không khác biệt rõ")
txt=(f"KẾT LUẬN: {verdict}\n\n"
     f"Zero-shot : median {np.median(Z):.4f} | mean {Z.mean():.4f}\n"
     f"Fine-tuned: median {np.median(F):.4f} | mean {F.mean():.4f}\n\n"
     f"delta (FT-ZS): mean {d.mean():+.4f}\n"
     f"               median {np.median(d):+.4f}\n\n"
     f"FT thắng/hòa/thua = {win}/{tie}/{lose}\n"
     f"FT hơn rõ(>+.02)={int((d>0.02).sum())}  tệ rõ(<-.02)={int((d<-0.02).sum())}\n\n"
     f"Wilcoxon paired p = {w_p:.2e}\n"
     f"t-test paired   p = {t_p:.2e}\n"
     f"(p<0.05 => khác biệt có ý nghĩa)\n\n"
     f"* GT do SAM hỗ trợ tạo -> vẫn cần\n  test vẽ tay để khách quan tuyệt đối")
ax.text(0.0,0.98,txt,transform=ax.transAxes,va="top",fontsize=11,family="monospace",
        bbox=dict(boxstyle="round",fc="#fff7e6",ec="orange"))
plt.savefig("results/confirm200_summary.png",dpi=130,bbox_inches="tight")
print("saved results/confirm200_summary.png")
print(f"verdict={verdict} | wilcoxon p={w_p:.2e} | mean delta={d.mean():+.4f} | win/lose={win}/{lose}")
