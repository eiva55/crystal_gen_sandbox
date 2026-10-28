"""Картинки структур для отчёта: эталоны MP-20 рядом с образцами DiffCSP++ (рис. «HfO₂: эталоны и образцы»,
«чем модель подменяет цель», «шпинель: две раскладки»). Образцы берутся из архивов репозитория
(archive/pilot_cifs_2026-09.tar.gz, deg_v1_k100_cifs.tar.gz, spinel_n50_cifs.tar.gz), эталоны — из MP-20.
Образец выравнивается по ячейке эталона (StructureMatcher, стандартные допуски), чтобы картинки были сопоставимы.

Запуск из корня репозитория (окружение mp_lookup, нужен ase): python analysis/make_structure_figures.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize.plot import plot_atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK})
COL = {'O': '#e05a47', 'S': '#e3b505', 'Se': '#c98a00', 'Hf': '#2a78d6', 'Zr': '#2a78d6', 'Zn': '#6250d6',
       'Mg': '#1baf7a', 'Al': '#8a8985', 'Ce': '#2a78d6', 'Fe': '#eb6834', 'Si': '#6250d6', 'Sn': '#2a78d6',
       'Co': '#2a78d6', 'Rb': '#6250d6', 'Hf2': '#2a78d6', 'Ir': '#eb6834', 'B': '#1baf7a', 'La': '#2a78d6', 'V': '#1baf7a', 'P': '#e3b505'}
RAD = {'O': 0.55, 'S': 0.7, 'Se': 0.75, 'B': 0.45}
D = 'models/diffcsp_pp/data/mp_20'
_cif = None
def mp(mid):
    global _cif
    if _cif is None:
        _cif = pd.concat([pd.read_csv(f'{D}/{s}.csv', usecols=['material_id', 'cif', 'e_above_hull', 'spacegroup.number']) for s in ('train', 'val', 'test')]).set_index('material_id')
    return CifParser.from_str(_cif.cif[mid]).parse_structures(primitive=True)[0], _cif.loc[mid]
def conv(s):
    return SpacegroupAnalyzer(s, symprec=0.1).get_conventional_standard_structure()
def sg(s, p=0.1):
    return SpacegroupAnalyzer(s, symprec=p).get_space_group_symbol()
def draw(ax, s, title, sub='', rot='10x,-80y,0z', rep=(1, 1, 1), conventional=True):
    st = conv(s) if conventional else s.copy()
    st.make_supercell(rep)
    at = AseAtomsAdaptor.get_atoms(st)
    syms = at.get_chemical_symbols()
    plot_atoms(at, ax, rotation=rot, radii=[RAD.get(x, 0.9) for x in syms],
               colors=[COL.get(x, '#999999') for x in syms], show_unit_cell=2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11.5, color=INK, pad=6)
    if sub:
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha='center', va='top', fontsize=9.5, color=INK2)
def legend(fig, els, y=0.02):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='none', ms=10, mfc=COL[e], mec='#333', label=e) for e in els]
    fig.legend(handles=h, loc='lower center', ncol=len(els), frameon=False, bbox_to_anchor=(0.5, y), fontsize=10)
SM = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
RAD.update({'O': 0.42, 'S': 0.55, 'Se': 0.58, 'B': 0.38})
DEF_R = 0.68
SMA = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True)
def aligned(ref_conv, gen):
    try:
        s = SMA.get_s2_like_s1(ref_conv, gen)
        return s
    except Exception:
        return None
def draw2(ax, st, title, sub='', rot='10x,-80y,0z', rep=(1, 1, 1)):
    st = st.copy(); st.make_supercell(rep)
    at = AseAtomsAdaptor.get_atoms(st)
    syms = at.get_chemical_symbols()
    plot_atoms(at, ax, rotation=rot, radii=[RAD.get(x, DEF_R) for x in syms],
               colors=[COL.get(x, '#999999') for x in syms], show_unit_cell=2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11.5, color=INK, pad=6)
    if sub:
        ax.text(0.5, -0.03, sub, transform=ax.transAxes, ha='center', va='top', fontsize=9.5, color=INK2, linespacing=1.35)
SMA = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=False, attempt_supercell=True)
def aligned(ref_conv, gen):
    for g in (gen, gen.get_primitive_structure(), conv(gen)):
        try:
            s = SMA.get_s2_like_s1(ref_conv, g)
            if s is not None:
                return s
        except Exception:
            pass
    return None
def aligned(ref_conv, gen):
    for g in (gen, gen.get_primitive_structure(), conv(gen)):
        try:
            s = SMA.get_s2_like_s1(ref_conv, g)
            if s is not None:
                return Structure(ref_conv.lattice, s.species, np.mod(s.frac_coords, 1.0))
        except Exception:
            pass
    return None


def fig_hf():
    a,_=mp('mp-352'); b,_=mp('mp-685097'); ac=conv(a); bc=conv(b)
    g1=Structure.from_file(f'{TMP}/outputs/2026-08-21/14-47-21/hfo2_stable/1.cif')
    gh=Structure.from_file(f'{TMP}/outputs/2026-08-21/14-47-21/hfo2_meta/1.cif')
    gm=Structure.from_file(f'{TMP}/outputs/2026-08-21/14-47-21/hfo2_meta/5.cif')
    rot='-90x,0y,0z'
    fig,axs=plt.subplots(2,3,figsize=(12.5,8.4))
    draw2(axs[0,0],ac,'стабильная фаза, эталон','HfO₂ mp-352, P2₁/c\n0 мэВ/атом над оболочкой',rot)
    draw2(axs[0,1],aligned(ac,g1),'образец DiffCSP++','строго совпал с эталоном\n(так у 50 из 50)',rot)
    axs[0,2].set_axis_off()
    axs[0,2].text(0.0,0.55,'Один состав — две расстановки атомов.\nМодели задают группу и орбиты\nкаждой фазы отдельно, затем образец\nсравнивают с эталоном\n(StructureMatcher, строгие допуски).\nОбразцы повёрнуты в ячейку эталона.',fontsize=10.5,color=INK2,va='center')
    draw2(axs[1,0],bc,'метастабильная фаза, эталон','HfO₂ mp-685097, Pca2₁ (сегнетоэлектрик)\n29 мэВ/атом над оболочкой',rot)
    draw2(axs[1,1],aligned(bc,gh),'образец DiffCSP++','строго совпал с эталоном',rot)
    draw2(axs[1,2],aligned(bc,gm),'образец DiffCSP++','строго не совпал: полярное смещение\nпочти исчезло (при допуске 0,1 Å — Pbcn)',rot)
    legend(fig,['Hf','O'],y=0.0)
    fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig('docs/figures/fig16_hfo2_samples.png',dpi=150,bbox_inches='tight')



def fig_subst():
    from pathlib import Path
    COL.update({'Hf':'#2a78d6','Ir':'#d94f87','Ce':'#0b3d7a'})
    d=f'{TMP}/outputs/bench/deg_v1_k100_c02'; order=json.load(open(f'{d}/order.json'))
    cifs={int(p.stem):p for p in Path(d).glob('*.cif')}; off=min(cifs)
    idx=[i for i,o in enumerate(order) if o['material_id']=='mp-1025450' and o['k']==0][0]
    gen=Structure.from_file(str(cifs[idx+off]))
    lo,_=mp('mp-1025450'); hi,_=mp('mp-1025523'); loc,hic=conv(lo),conv(hi)
    assert SM.fit(gen,hi) and not SM.fit(gen,lo)
    pm,_=mp('mp-1018055'); cs,_=mp('mp-1002122'); pmc=conv(pm)
    a0=cs.lattice.a
    from pymatgen.core import Lattice
    csb=Structure(Lattice.orthorhombic(a0,a0*2**0.5,a0*2**0.5),['Hf','Hf','Ir','Ir'],[[0.5,0.75,0.75],[0.5,0.25,0.25],[0,0.25,0.75],[0,0.75,0.25]])
    assert StructureMatcher().fit(csb,cs)
    v1,_=mp('mp-19000'); v2,_=mp('mp-763482')
    fig,axs=plt.subplots(3,3,figsize=(12.5,13))
    r1='0x,90y,0z'
    draw2(axs[0,0],loc,'цель: нижний полиморф','CeFeSi₂ mp-1025450, Cmcm\n0 мэВ/атом',r1,rep=(1,1,2))
    draw2(axs[0,1],hic,'другой полиморф того же шаблона','mp-1025523, Cmcm\n62 мэВ/атом',r1,rep=(1,1,2))
    draw2(axs[0,2],aligned(hic,gen),'образец DiffCSP++','совпал с верхним полиморфом:\nусловие их не различает',r1,rep=(1,1,2))
    r2='0x,90y,0z'
    draw2(axs[1,0],pmc,'цель','HfIr mp-1018055, Pmma\n19 мэВ/атом (test)',r2,rep=(1,2,2))
    draw2(axs[1,1],csb,'что выдала модель','HfIr mp-1002122, тип CsCl, Pm-3m\n42 мэВ/атом (train)',r2,rep=(1,2,2))
    axs[1,2].set_axis_off(); axs[1,2].text(0.0,0.5,'Переход к более симметричной\nструктуре: свободные координаты\nшаблона Pmma сошлись к частным\nзначениям, и смещения атомов\nисчезли. Шаблон это допускает.\nПример из пробного прогона;\nв основном — ZnS, CoO₂, HfCo₃B₂, Rb.',fontsize=10.5,color=INK2,va='center')
    r3='0x,0y,0z'
    draw2(axs[2,0],v1,'цель','VPO₅ mp-19000, P4/nmm\n15 мэВ/атом',r3)
    draw2(axs[2,1],aligned(v1,v2) or v2,'что выдала модель','VPO₅ mp-763482, P4/n\n10 мэВ/атом (train)',r3)
    axs[2,2].set_axis_off(); axs[2,2].text(0.0,0.5,'Граница допуска: две записи базы\nпочти одинаковы (у второй немного\nповёрнуты кислородные группы) и\nсовпадают между собой при\nстандартных допусках. Строгое\nсравнение относит образец к «соседу».',fontsize=10.5,color=INK2,va='center')
    for ax,lab in zip(axs[:,0],['общий шаблон','более симметричная структура','граница допуска']):
        ax.text(-0.14,0.5,lab,transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=12,color=INK,fontweight='bold')
    legend(fig,['Ce','Fe','Si','Hf','Ir','V','P','O'],y=0.0)
    fig.tight_layout(rect=(0.02,0.03,1,1)); fig.savefig('docs/figures/fig17_substitution_examples.png',dpi=150,bbox_inches='tight')



def fig_spin():
    from pathlib import Path
    def cn(s):
        out=[]
        for i,site in enumerate(s):
            e=site.specie.symbol
            if e=='O': out.append('O'); continue
            n=sum(1 for nb in s.get_neighbors(site,2.6) if nb.specie.symbol=='O')
            out.append(f'{e}{n}')
        return out
    files=sorted(Path(f'{TMP}/spinel_x05_disordered').glob('*.cif'),key=lambda p:int(p.stem))
    ordd=anti=None
    for p in files:
        s=Structure.from_file(str(p)); c=cn(s)
        mg=sorted(x for x in c if x.startswith('Mg')); al=sorted(x for x in c if x.startswith('Al'))
        if set(mg)=={'Mg4'} and ordd is None: ordd=(p,s,c)
        if 'Mg6' in mg and anti is None: anti=(p,s,c)
    C={'Mg4':'#1baf7a','Mg6':'#0b6b49','Al6':'#b8b7b2','Al4':'#eb6834','O':'#e05a47'}
    R={'Mg4':0.55,'Mg6':0.55,'Al6':0.5,'Al4':0.5,'O':0.3}
    def drawc(ax,s,c,title,sub,rot):
        at=AseAtomsAdaptor.get_atoms(s)
        plot_atoms(at,ax,rotation=rot,radii=[R[x] for x in c],colors=[C[x] for x in c],show_unit_cell=2)
        ax.set_axis_off(); ax.set_title(title,fontsize=11.5,pad=6)
        ax.text(0.5,-0.03,sub,transform=ax.transAxes,ha='center',va='top',fontsize=9.5,color=INK2,linespacing=1.35)
    rot='-90x,30y,0z'
    fig,axs=plt.subplots(1,2,figsize=(11,7.2))
    drawc(axs[0],ordd[1],ordd[2],'образец: упорядоченная раскладка','Mg в тетраэдрах (4 соседа O), Al в октаэдрах (6);\nниже по энергии; в обучении 0 из 6 таких',rot)
    drawc(axs[1],anti[1],anti[2],'образец: антисайтная раскладка','на каждые 14 атомов один Mg в октаэдре и один Al\nв тетраэдре; на 50 мэВ/атом выше; в обучении 6 из 6',rot)
    from matplotlib.lines import Line2D
    lab={'Mg4':'Mg, 4 соседа O','Mg6':'Mg, 6 соседей O (антисайт)','Al6':'Al, 6 соседей O','Al4':'Al, 4 соседа O (антисайт)','O':'O'}
    h=[Line2D([],[],marker='o',ls='none',ms=10,mfc=C[k],mec='#333',label=lab[k]) for k in ['Mg4','Al6','Mg6','Al4','O']]
    fig.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=9.5,bbox_to_anchor=(0.5,-0.04))
    fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig('docs/figures/fig18_spinel_samples.png',dpi=150,bbox_inches='tight')



if __name__ == '__main__':
    import tarfile, tempfile, json
    from pathlib import Path
    TMP = tempfile.mkdtemp()
    for a in ('pilot_cifs_2026-09', 'deg_v1_k100_cifs', 'spinel_n50_cifs'):
        with tarfile.open(f'archive/{a}.tar.gz') as t:
            t.extractall(TMP)
    fig_hf(); fig_subst(); fig_spin()
    print('ok: fig16, fig17, fig18')
