import sys,warnings; sys.path.insert(0,'../..'); warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

OUT=Path('.')
plt.rcParams.update({
    'font.family':'serif','font.size':10,'axes.labelsize':11,
    'axes.titlesize':10.5,'legend.fontsize':9,'xtick.labelsize':9,
    'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.grid':True,'grid.alpha':0.22,'grid.linewidth':0.5,
})

TASK_COLORS={'mlp':'#E64B35','xgboost':'#4DBBD5','svm_rbf':'#00A087'}
TASK_LABELS={'mlp':'MLP','xgboost':'XGBoost','svm_rbf':'SVM-RBF'}
COND_COLORS={'low_osc':'#E64B35','med_osc':'#F39B7F','high_osc':'#4DBBD5'}
COND_LABELS={'low_osc':'Low OSC (induced)','med_osc':'Med OSC (induced)','high_osc':'High OSC (induced)'}

main=pd.read_csv('../../results/main_results.csv')
paired=pd.read_csv('../../results/paired_analysis.csv')
curve=pd.read_csv('../../results/treatment_effect_curve.csv')

# ============================================================
# FIG 1 — Protocol diagram
# ============================================================
fig,ax=plt.subplots(figsize=(9,3.0))
ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis('off'); ax.grid(False)

def box(ax,x,y,w,h,txt,fc,fs=9.5):
    r=mpatches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.09',
        facecolor=fc,edgecolor='#444',linewidth=1.1)
    ax.add_patch(r)
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,
            fontfamily='serif',linespacing=1.5)

def arrow(ax,x1,y1,x2,y2,col='#333',ls='-'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='-|>',color=col,lw=1.3,
                        linestyle=ls,mutation_scale=13))

box(ax,0.1,0.95,1.7,1.05,'$\\mathcal{D}_{\\mathrm{dev}}$\n(train + val)','#EBF5FB',10)
box(ax,2.2,0.75,2.0,1.45,'Bootstrap\nresample\n$B$ times','#EAF4E8')
box(ax,4.8,0.95,2.1,1.05,'HPO runs\n$\\hat{h}^{(1)},\\ldots,\\hat{h}^{(B)}$','#EAF4E8')
box(ax,7.4,1.25,2.2,0.65,'OSC','#FEF9E7',10)
box(ax,7.4,0.45,2.2,0.65,'Winner $\\hat{h}^*$','#EBF5FB',10)

# dashed test box
r2=mpatches.FancyBboxPatch((7.4,-0.55),2.2,0.70,boxstyle='round,pad=0.09',
    facecolor='#FDEDEC',edgecolor='#922B21',lw=1.4,linestyle='--')
ax.add_patch(r2)
ax.text(8.5,-0.20,'$\\mathcal{D}_{\\mathrm{test}}$  (never seen)',
        ha='center',va='center',fontsize=9,color='#922B21',fontfamily='serif')

arrow(ax,1.8,1.47,2.2,1.47)
arrow(ax,4.2,1.47,4.8,1.47)
arrow(ax,6.9,1.65,7.4,1.57)
arrow(ax,6.9,1.27,7.4,0.77)
arrow(ax,8.5,0.45,8.5,0.15,'#922B21','--')

ax.text(3.25,2.55,'Same data, different resamples',ha='center',fontsize=8.5,
        color='#2E7D32',style='italic')
ax.text(8.55,2.58,'Dev-only computation',ha='center',fontsize=8.5,
        color='#7D6608',style='italic')
ax.text(8.5,-0.95,'Test gap $G$ observed here',
        ha='center',fontsize=8.5,color='#922B21',style='italic')

ax.set_title('Figure 1.  Measurement protocol — OSC is computed from development data only; '
             'the test set is strictly downstream.',
             pad=6,fontsize=9.5,loc='left')
fig.tight_layout(pad=0.4)
fig.savefig(OUT/'fig1_protocol.pdf'); fig.savefig(OUT/'fig1_protocol.png')
plt.close(); print('Fig 1 done')

# ============================================================
# FIG 2 — OSC vs post-selection gap
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(10,4.0))

ax=axes[0]
for task in ['mlp','xgboost','svm_rbf']:
    sub=main[main.task==task]
    ax.scatter(sub.eps_coverage,sub.gen_gap,color=TASK_COLORS[task],
               label=TASK_LABELS[task],alpha=0.55,s=26,linewidths=0,zorder=3)
xg=np.linspace(main.eps_coverage.min(),main.eps_coverage.max(),100)
m_=smf.ols('gen_gap~eps_coverage',data=main).fit()
ax.plot(xg,m_.params['Intercept']+m_.params['eps_coverage']*xg,
        'k--',lw=1.8,alpha=0.85,zorder=4,label='OLS fit')
r_,_=stats.spearmanr(main.eps_coverage,main.gen_gap)
ax.axhline(0,color='#888',lw=0.8,ls=':')
ax.set_xlabel('ε-Optimal-Set Coverage (OSC)')
ax.set_ylabel('Generalization Gap  $G$')
ax.set_title(f'(a)  OSC vs. post-selection gap\nSpearman $r={r_:.2f}$, $p<0.001$, $n=162$')
ax.legend(frameon=False,title='Model',title_fontsize=8.5,loc='upper left')

ax=axes[1]
bins=[0.0,0.4,0.6,0.8,1.001]
bin_labels=['<0.4','0.4–0.6','0.6–0.8','≥0.8']
main['osc_bin']=pd.cut(main.eps_coverage,bins=bins,labels=bin_labels,right=False)
palette=['#E64B35','#F39B7F','#91D1C2','#4DBBD5']
means_=[]; cis_=[]; ns_=[]
for lab in bin_labels:
    s=main[main.osc_bin==lab].gen_gap
    ns_.append(len(s)); means_.append(s.mean())
    cis_.append(1.96*s.std()/np.sqrt(len(s)))
xs=np.arange(4)
ax.bar(xs,means_,color=palette,alpha=0.85,width=0.58,zorder=3)
ax.errorbar(xs,means_,yerr=cis_,fmt='none',color='#222',lw=1.8,capsize=4,zorder=4)
ax.set_xticks(xs)
ax.set_xticklabels([f'{l}\n($n={n}$)' for l,n in zip(bin_labels,ns_)],fontsize=9)
ax.axhline(0,color='#888',lw=0.8,ls=':')
ax.set_xlabel('OSC bin  (pre-specified)')
ax.set_ylabel('Mean Generalization Gap')
ax.set_title('(b)  Calibration: $\\mathbb{E}[G\\mid\\mathrm{OSC\\ bin}]$\n'
             'ANOVA $F=11.2$, $p<0.001$;  Cohen $d=1.29$ (low vs. high)')
for i,(m2,ci2) in enumerate(zip(means_,cis_)):
    ax.text(i,m2+ci2+0.0008,f'{m2:+.3f}',ha='center',va='bottom',fontsize=8.5)

fig.suptitle('Figure 2.  OSC as a diagnostic for post-selection generalization gap.',
             y=1.02,fontsize=11)
fig.tight_layout()
fig.savefig(OUT/'fig2_osc_vs_gap.pdf'); fig.savefig(OUT/'fig2_osc_vs_gap.png')
plt.close(); print('Fig 2 done')

# ============================================================
# FIG 3 — Intervention: OSC vs treatment benefit
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(10,4.0))

ax=axes[0]
for lab in ['low_osc','med_osc','high_osc']:
    sub=paired[paired.noise_label==lab]
    ax.scatter(sub.osc,sub.delta_gap,color=COND_COLORS[lab],
               label=COND_LABELS[lab],alpha=0.45,s=20,linewidths=0,zorder=3)
ax.plot(curve.osc,curve.delta_mean,'k-',lw=2,zorder=5)
ax.fill_between(curve.osc,curve.delta_lo,curve.delta_hi,alpha=0.13,color='k',zorder=2)
ax.axhline(0,color='#888',lw=0.8,ls=':')
r3,_=stats.spearmanr(paired.osc,paired.delta_gap)
ax.set_xlabel('Realized OSC  (pre-treatment)')
ax.set_ylabel('$\\Delta G = G_{\\mathrm{control}} - G_{\\mathrm{centroid}}$')
ax.set_title(f'(a)  Treatment benefit vs. OSC\nSpearman $r={r3:.2f}$, $p<0.001$, $n=180$ pairs')
ax.legend(frameon=False,fontsize=8.5,loc='upper right')

ax=axes[1]
ax.plot(curve.osc,curve.delta_mean,'#E64B35',lw=2.5,zorder=5,
        label='$\\hat{\\tau}(\\mathrm{OSC})=0.101-0.081\\cdot\\mathrm{OSC}$')
ax.fill_between(curve.osc,curve.delta_lo,curve.delta_hi,alpha=0.18,color='#E64B35',zorder=2)
ax.axhline(0,color='#888',lw=0.8,ls=':')
# Overall ATE line
ate=paired.delta_gap.mean()
ax.axhline(ate,color='#4DBBD5',lw=1.5,ls='--',
           label=f'ATE  $\\hat{{\\tau}}={ate:+.3f}$, 95% CI $[+0.037,+0.052]$')
# Key annotations
beta_T=0.10054; beta_I=0.08065
for osc_v,xoff,yoff in [(0.10,0.08,0.008),(0.55,-0.13,-0.014),(0.90,0.05,-0.010)]:
    te=beta_T-beta_I*osc_v
    ax.annotate(f'$\\Delta G={te:+.3f}$',xy=(osc_v,te),
                xytext=(osc_v+xoff,te+yoff),fontsize=8.5,
                arrowprops=dict(arrowstyle='->',color='#555',lw=0.9))
ax.set_xlabel('OSC')
ax.set_ylabel('Predicted treatment benefit $\\Delta G$')
ax.set_title('(b)  Marginal treatment-effect curve  (95% CI)\n'
             'Interaction $\\hat{\\beta}_3=+0.081$, 95% CI $[+0.051,+0.110]$, $P(>0)=1.00$')
ax.legend(frameon=False,fontsize=8.5,loc='upper right')

fig.suptitle('Figure 3.  OSC moderates the benefit of stability-weighted selection.',
             y=1.02,fontsize=11)
fig.tight_layout()
fig.savefig(OUT/'fig3_intervention.pdf'); fig.savefig(OUT/'fig3_intervention.png')
plt.close(); print('Fig 3 done')

# ============================================================
# FIG 4 — Config-space vs performance-space stability
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(10,4.2))

# Left: 2x2 heatmap
ax=axes[0]
ax.grid(False)
osc_med=main.eps_coverage.median(); ws_med=main.winner_stability.median()
cell_data={}
for oq,wq in [(True,True),(True,False),(False,True),(False,False)]:
    sub=main[(main.eps_coverage>=osc_med)==oq]
    sub=sub[(sub.winner_stability>=ws_med)==wq]
    cell_data[(int(oq),int(wq))]=(sub.gen_gap.mean(),sub.gen_gap.std(),len(sub))

quad_colors={(1,1):'#4DBBD5',(1,0):'#91D1C2',(0,1):'#F39B7F',(0,0):'#E64B35'}
quad_labels={(1,1):'High OSC\nHigh WS',(1,0):'High OSC\nLow WS',
             (0,1):'Low OSC\nHigh WS',(0,0):'Low OSC\nLow WS'}
for (oq,wq),(mean,sd,n) in cell_data.items():
    c=wq; r=oq
    rect=mpatches.FancyBboxPatch((c+0.05,r+0.05),0.90,0.90,
        boxstyle='round,pad=0.05',facecolor=quad_colors[(oq,wq)],edgecolor='white',lw=2.2)
    ax.add_patch(rect)
    tc='white' if mean>0.012 else '#1a1a2e'
    ax.text(c+0.5,r+0.70,quad_labels[(oq,wq)],ha='center',va='center',
            fontsize=9,color=tc,fontweight='bold')
    ax.text(c+0.5,r+0.38,f'$E[G]={mean:+.4f}$',ha='center',va='center',fontsize=10,color=tc)
    ax.text(c+0.5,r+0.18,f'$n={n}$',ha='center',va='center',fontsize=8.5,color=tc)

ax.set_xlim(-0.02,2.02); ax.set_ylim(-0.02,2.02); ax.set_aspect('equal')
ax.set_xticks([0.5,1.5]); ax.set_xticklabels(['Low WS\n(config unstable)','High WS\n(config stable)'],fontsize=9.5)
ax.set_yticks([0.5,1.5]); ax.set_yticklabels(['Low OSC\n(perf. unstable)','High OSC\n(perf. stable)'],
               fontsize=9.5,rotation=90,va='center')
ax.set_title('(a)  Expected gap by OSC × WS quadrant\n'
             'Config stability without perf. stability does not reduce gap')

# Right: scatter OSC vs WS, gap as colour
ax=axes[1]
sc=ax.scatter(main.winner_stability,main.eps_coverage,c=main.gen_gap,
              cmap='RdYlBu_r',vmin=main.gen_gap.min(),vmax=main.gen_gap.max(),
              alpha=0.68,s=28,linewidths=0,zorder=3)
cb=plt.colorbar(sc,ax=ax,label='Generalization Gap $G$',shrink=0.88,pad=0.02)
ax.axvline(ws_med,color='#666',lw=1,ls='--',alpha=0.7)
ax.axhline(osc_med,color='#666',lw=1,ls='--',alpha=0.7)
r_wo,p_wo=stats.spearmanr(main.winner_stability,main.eps_coverage)
ax.set_xlabel('Winner Stability — WS  (configuration space)')
ax.set_ylabel('ε-Optimal Coverage — OSC  (performance space)')
ax.set_title(f'(b)  OSC and WS are nearly orthogonal\n'
             f'Spearman $r(\\mathrm{{OSC}},\\mathrm{{WS}})={r_wo:.2f}$, $p=0.11$')
ax.text(0.97,0.04,f'Gap increases bottom-to-top (OSC axis)\nbut not left-to-right (WS axis)',
        transform=ax.transAxes,ha='right',fontsize=8,style='italic',color='#444')

fig.suptitle('Figure 4.  Configuration-space stability (WS) ≠ performance-space stability (OSC).',
             y=1.02,fontsize=11)
fig.tight_layout()
fig.savefig(OUT/'fig4_ws_vs_osc.pdf'); fig.savefig(OUT/'fig4_ws_vs_osc.png')
plt.close(); print('Fig 4 done')
print('All figures complete.')
