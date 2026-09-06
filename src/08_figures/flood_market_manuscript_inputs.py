#!/usr/bin/env python3
"""Export validated aggregate composition evidence and three main figures."""
import argparse,hashlib,json,shutil
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj.transformer import TransformerGroup
from shapely.ops import transform

ROOT=Path(__file__).resolve().parents[2]
LABELS={'zero_improvement_share':'Zero-improvement share','pooled_log_price':'Pooled residential prices','improved_log_price':'Positive-improvement prices'}
COLORS={'positive':'#174A5B','zero':'#D68A3A','unknown':'#B7B7B7'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def effect(row):
    scale=(lambda x:100*x) if row.outcome=='zero_improvement_share' else (lambda x:100*np.expm1(x))
    unit=' pp' if row.outcome=='zero_improvement_share' else '%'
    if row.status!='estimated':return 'Not estimated'
    return f'{scale(row.estimate):+.1f}{unit} [{scale(row.ci_low):+.1f}, {scale(row.ci_high):+.1f}]'

def save(fig,path):
    fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def run(project_root=None,analysis_dir=None,manuscript_dir=None):
    root=Path(project_root or ROOT);src=Path(analysis_dir or root/'data_work/flood_market_composition');ms=Path(manuscript_dir or root/'manuscript_quarto')
    meta=json.loads((src/'run_metadata.json').read_text())
    if meta['bootstrap_draws']!=9999 or not meta['leaveouts_complete']:raise ValueError('Complete 9999-draw production run required')
    validation=pd.read_csv(src/'validation_checks.csv')
    if not validation.passed.all():raise ValueError('Scientific validation must pass before manuscript export')
    data=ms/'data';figures=ms/'figures';data.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    exported=[]
    for f in sorted(src.glob('*.csv')):
        if f.name.startswith('._'):continue
        dst=data/('flood_market_'+f.name);shutil.copyfile(f,dst)
        exported.append({'input':str(f),'output':str(dst),'sha256':sha(dst),'identical':sha(f)==sha(dst)})
    shutil.copyfile(src/'run_metadata.json',data/'flood_market_run_metadata.json')
    primary=pd.read_csv(src/'primary_estimates.csv');sens=pd.read_csv(src/'sensitivity_estimates.csv')
    table=[]
    selected=[(row,'All post years') for row in primary.itertuples()]
    for outcome,spec,label in [('zero_improvement_share','first_post_year','First post year'),('zero_improvement_share','later_post_years','March 2020 onward'),('zero_improvement_share','neighborhood_by_event_year','Neighborhood-by-year'),('pooled_log_price','neighborhood_by_event_year','Neighborhood-by-year'),('improved_log_price','neighborhood_by_event_year','Neighborhood-by-year')]:
        selected.append((next(sens[(sens.outcome==outcome)&(sens.specification==spec)].itertuples()),label))
    for row,label in selected:
        table.append({'Outcome / comparison':LABELS[row.outcome]+' / '+label,'Estimate [95% CI]':effect(row),'WCR11 p':f'{row.p_wild:.3f}',
                      'Holm p':f'{row.p_holm:.3f}' if hasattr(row,'p_holm') else '—','N / four-cell grids':f'{row.n_obs:,} / {row.four_cell_clusters}'})
    pd.DataFrame(table).to_csv(data/'flood_market_main_table.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#263238','text.color':'#263238','axes.edgecolor':'#9AA1A4'})
    d=pd.read_parquet(src/'restricted/eligible_deeds.parquet')
    d=d[d.inside.eq(1)|(d.inside.eq(0)&d.signed_dist_inund_m.between(0,1000))].copy()
    county=gpd.read_file(root/'data_work/douglas_county.gpkg')
    # The local PROJ installation lists unavailable NAD83 grid operations.
    # Select an explicitly available finite operation for the outline only.
    operation=None
    for candidate in TransformerGroup(county.crs,32615,always_xy=True).transformers:
        projected=county.geometry.map(lambda g:transform(candidate.transform,g))
        if projected.map(lambda g:g.is_valid and np.isfinite(g.bounds).all()).all():
            operation=candidate.description;county=county.set_geometry(projected).set_crs(32615,allow_override=True);break
    if operation is None:raise ValueError('No valid county-outline coordinate transformation')
    geom=county.geometry.union_all()
    inund=gpd.read_file(root/'data_work/inund_union.gpkg').to_crs(32615)
    sfha=gpd.read_file(root/'data_work/sfha_union.gpkg').to_crs(32615)
    inund['geometry']=inund.geometry.intersection(geom);sfha['geometry']=sfha.geometry.intersection(geom)
    parcels=d.drop_duplicates('parcel_id')
    fig=plt.figure(figsize=(9,5.4));gs=fig.add_gridspec(2,3,width_ratios=[1.1,1.1,1],hspace=.35,wspace=.26)
    overview=fig.add_subplot(gs[:,:2]);insets=[fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])]
    for ax in [overview]+insets:
        inund.plot(ax=ax,color='#BCDCE8',edgecolor='none',alpha=.8)
        sfha.boundary.plot(ax=ax,color='#85619A',linewidth=.38,alpha=.65)
        county.boundary.plot(ax=ax,color='#485258',linewidth=.75)
        for inside,marker,color,size in [(0,'s','#656C70',3),(1,'o','#B75A23',6)]:
            z=parcels[parcels.inside.eq(inside)];ax.scatter(z.utm_x,z.utm_y,marker=marker,c=color,s=size,linewidths=0,alpha=.72,zorder=4)
        ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    bounds=county.total_bounds;overview.set_xlim(bounds[0]-1300,bounds[2]+1300);overview.set_ylim(bounds[1]-1800,bounds[3]+1800)
    overview.set_title('Douglas County: expanded residential deed sample',loc='left',fontsize=10)
    for ax,sub,name in zip(insets,['3922','31343'],['Bluewater','The Preserve']):
        z=parcels[parcels.subdivision.eq(sub)];cx,cy=z.utm_x.median(),z.utm_y.median()
        radius=max(z.utm_x.max()-z.utm_x.min(),z.utm_y.max()-z.utm_y.min())*.65+250
        ax.set_xlim(cx-radius,cx+radius);ax.set_ylim(cy-radius,cy+radius);ax.set_title(name,loc='left',fontsize=9)
        overview.annotate(name,xy=(cx,cy),xytext=(cx+2300,cy+1800),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6},bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
    x0,y0=bounds[0]+2500,bounds[1]+2000
    overview.plot([x0,x0+10000],[y0,y0],color='#263238',lw=2);overview.text(x0+5000,y0+700,'10 km',ha='center',fontsize=8)
    overview.annotate('N',xy=(bounds[2]-2500,bounds[3]-1200),xytext=(bounds[2]-2500,bounds[3]-4500),ha='center',arrowprops={'arrowstyle':'-|>','lw':.8})
    handles=[Patch(facecolor='#BCDCE8',label='Inherited flood footprint'),Line2D([0],[0],color='#85619A',lw=1,label='SFHA boundary'),Line2D([0],[0],marker='o',color='none',markerfacecolor='#B75A23',markeredgecolor='none',markersize=5,label='Inside footprint'),Line2D([0],[0],marker='s',color='none',markerfacecolor='#656C70',markeredgecolor='none',markersize=4,label='Outside, within 1 km')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=8)
    fig.subplots_adjust(bottom=.12);save(fig,figures/'fig_flood_market_map.png')
    annual=pd.read_csv(src/'annual_composition.csv');years=np.arange(-4,3);labels=['2015–16','2016–17','2017–18','2018–19','2019–20','2020–21','2021–22']
    fig=plt.figure(figsize=(8,6.8));gs=fig.add_gridspec(2,2,height_ratios=[1,1.1],hspace=.35,wspace=.27)
    for inside,ax in [(1,fig.add_subplot(gs[0,0])),(0,fig.add_subplot(gs[0,1]))]:
        sub=annual[annual.inside.eq(inside)];bottom=np.zeros(7)
        for cat in ['positive','zero','unknown']:
            n=sub[sub.improvement_category.eq(cat)].set_index('event_year').deeds.reindex(years,fill_value=0)
            if not n.sum():continue
            ax.bar(years,n,bottom=bottom,color=COLORS[cat],label=cat,width=.72);bottom+=n.to_numpy()
        for x,y in zip(years,bottom):ax.text(x,y+max(bottom)*.015,str(int(y)),ha='center',fontsize=8)
        ax.axvline(-.5,color='#353A3D',ls='--',lw=.8);ax.set_xticks(years,labels,rotation=40,ha='right',fontsize=8)
        ax.set_ylim(0,max(bottom)*1.18);ax.set_ylabel('Recorded residential deeds');ax.set_title('Inside footprint' if inside else 'Outside, within 1 km',loc='left',fontsize=10)
    ax=fig.add_subplot(gs[1,:]);ax.axvspan(.5,2.5,color='#E6E8E9',zorder=0)
    for inside,color,marker in [(1,'#B75A23','o'),(0,'#174A5B','s')]:
        sub=annual[annual.inside.eq(inside)&annual.improvement_category.eq('zero')].set_index('event_year')
        y=sub.share.reindex(years,fill_value=0)*100
        ax.plot(years,y,marker=marker,color=color,label='Inside footprint' if inside else 'Outside, within 1 km',lw=1.8)
    ax.axvline(-.5,color='#353A3D',ls='--',lw=.8);ax.set_xticks(years,labels);ax.set_ylabel('Zero-improvement share (%)');ax.set_ylim(0,80);ax.set_xlim(-4.4,2.4)
    ax.text(.6,76,'March 2020 onward',fontsize=8);ax.set_title('Recorded zero-improvement share',loc='left',fontsize=10);ax.legend(frameon=False,loc='upper left',fontsize=8)
    fig.legend(handles=[Patch(facecolor=COLORS['positive'],label='Positive recorded improvement'),Patch(facecolor=COLORS['zero'],label='Zero recorded improvement')],loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.01))
    fig.subplots_adjust(bottom=.08);save(fig,figures/'fig_flood_market_annual.png')
    annual_price=pd.read_csv(src/'annual_prices.csv')
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),sharey=True)
    for cat,ax in zip(['positive','zero'],axes):
        ax.axvspan(.5,2.5,color='#E6E8E9',zorder=0)
        for inside,color,marker in [(1,'#B75A23','o'),(0,'#174A5B','s')]:
            sub=annual_price[annual_price.inside.eq(inside)&annual_price.improvement_category.eq(cat)].set_index('event_year')
            y=100*np.exp(sub.mean_log_price.reindex(years)-sub.loc[-1,'mean_log_price'])
            ax.plot(years,y,color=color,marker=marker,lw=1.5,label='Inside footprint' if inside else 'Outside, within 1 km')
        ax.axvline(-.5,color='#434A4D',ls='--',lw=.8);ax.set_xticks(years,labels,rotation=40,ha='right',fontsize=8)
        ax.set_title('Positive recorded improvement' if cat=='positive' else 'Zero recorded improvement',loc='left',fontsize=10)
        ax.set_xlim(-4.4,2.4);ax.axhline(100,color='#ABB2B4',lw=.6)
    axes[0].set_ylabel('Geometric mean price index (2018–19 = 100)');axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,figures/'fig_flood_market_annual_prices.png')
    dec=pd.read_csv(src/'decomposition.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),gridspec_kw={'width_ratios':[1,1.1]})
    ax=axes[0];ax.axvspan(-10,10,color='#EDF0F0');ax.axvspan(-5,5,color='#DCE4E5')
    prices=primary[primary.outcome.ne('zero_improvement_share')].reset_index(drop=True)
    for i,r in prices.iterrows():
        e,lo,hi=100*np.expm1([r.estimate,r.ci_low,r.ci_high]);ax.errorbar(e,1-i,xerr=[[e-lo],[hi-e]],fmt='o',color='#174A5B',capsize=3)
    ax.set_yticks([1,0],['Pooled','Positive improvement']);ax.set_ylim(-.6,1.6);ax.set_title('A. Adjusted price comparisons',loc='left',fontsize=10);ax.set_xlabel('Relative price change (%)');ax.axvline(0,color='#434A4D',ls='--',lw=.8)
    ax=axes[1];names=['Observed pooled','Fixed pre-event shares','Composition contribution']
    for i,r in dec.iterrows():
        e,lo,hi=100*np.array([r.estimate,r.ci_low,r.ci_high]);ax.errorbar(e,2-i,xerr=[[e-lo],[hi-e]],fmt='s',color='#B75A23',capsize=3)
    ax.set_yticks([2,1,0],names);ax.set_ylim(-.6,2.6);ax.set_title('B. Transaction composition accounting',loc='left',fontsize=10);ax.set_xlabel('Relative change (100 × log points)');ax.axvline(0,color='#434A4D',ls='--',lw=.8)
    fig.tight_layout(w_pad=2);save(fig,figures/'fig_flood_market_prices.png')
    values={'primary_deeds':len(d),'primary_parcels':d.parcel_id.nunique(),'inside_deeds':int(d.inside.sum()),'inside_parcels':d.loc[d.inside.eq(1),'parcel_id'].nunique(),'inside_grids':d.loc[d.inside.eq(1),'cluster_1km'].nunique(),
            'primary':primary.to_dict('records'),'decomposition':dec.to_dict('records')}
    (data/'flood_market_values.json').write_text(json.dumps(values,indent=2)+'\n')
    pd.DataFrame(exported).to_csv(data/'flood_market_export_manifest.csv',index=False)
    figure_manifest=[{'file':str(f.relative_to(ms)),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(figures.glob('fig_flood_market_*'))]
    pd.DataFrame(figure_manifest).assign(county_outline_operation=operation).to_csv(data/'flood_market_figure_manifest.csv',index=False)
    print(f'Exported {len(exported)} aggregate files and four figures (three main, one supplementary); restricted deeds were not copied.')
    return ms

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--project-root',type=Path);a.add_argument('--analysis-dir',type=Path);a.add_argument('--manuscript-dir',type=Path);v=a.parse_args();run(v.project_root,v.analysis_dir,v.manuscript_dir)
