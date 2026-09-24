#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hero3d_animation.py — ASLCD opening hero animation
==================================================================
Racconto visivo "tessuto -> rete":
  A) si apre sull'istologia H&E reale (solo tessuto, niente pallini)
  B) l'H&E diventa un po' trasparente e le cellule (colorate per tipo) appaiono
  C) la camera zooma su una tasca del tessuto
  D) gli archi KNN si disegnano
  E) il grafo si solleva dal piano: l'ALTEZZA dei nodi e' la BETWEENNESS
     centrality (lo "scheletro" del tessuto, cfr. Part 1); l'H&E svanisce
  F) la rete 3D orbita dolcemente

Rende tutto off-screen e scrive un MP4 pronto per PowerPoint.

Uso:   python hero3d_animation.py
Nota:  i dati restano sulla tua macchina; lo script li carica via data_io.
       Tutte le manopole sono nella sezione CONFIG qui sotto.
"""

# ============================================================================
# 0) HEADLESS — display virtuale se siamo senza X (server/SSH).
#    Se usi la build 'vtk-osmesa' non serve: il blocco fallisce in silenzio.
# ============================================================================
import os, sys, platform
_vdisplay = None
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    try:
        from xvfbwrapper import Xvfb
        _vdisplay = Xvfb(width=1920, height=1080, colordepth=24)
        _vdisplay.start()
    except Exception as e:
        print("[warn] Xvfb non avviato (", e, ") — ok se usi vtk-osmesa.")

# ============================================================================
# 1) CONFIG — TUTTE le manopole sono qui
# ============================================================================
# --- percorsi progetto ---
PROJECT_ROOT = ""                 # "" = autodetect (cerca ./src+./cache, poi ../). Oppure path assoluto.
HE_DIR    = "/group/sottoriva/00-LAB/IMAGES_SCANS/MINERVA/public/he_images_omezarr"
SAMPLE    = 47
SAMPLE_ID = "U03_011_B1_S1_T2"    # = sample 47 (nome del file H&E)

# --- regione "hero" (coerente con la title slide) ---
CENTER = (34700.0, 22104.0)       # centro finestra hero (pixel livello-0)
FRAC   = 0.22                     # lato finestra = 0.22 * estensione del sample

# --- tasca = il sotto-grafo che si solleva (valori validati = composizione approvata) ---
POCKET_MANUAL_CENTER = (32282.0, 20370.0)   # metti None per ri-cercare la tasca piu' diversificata
POCKET_SIDE          = 2200.0
POCKET_TARGET        = (350, 650)                    # range nodi (solo auto-search)
POCKET_SIDES         = (2200., 2700., 3200., 3700.)  # lati candidati (auto-search)
POCKET_GRID          = 30                            # risoluzione griglia (auto-search)

# --- rilievo betweenness (asse z della rete) ---
SMOOTH_ITERS = 2      # smussa l'altezza lungo il grafo -> creste continue (scheletro)
RELIEF_EXP   = 0.7    # contrasto del rilievo (<1 schiaccia gli estremi)
Z_MAX        = 3.2    # altezza massima del rilievo
BASE_LIFT    = 1.0    # quanto la rete si stacca dal piano

# --- H&E ---
HE_LEVEL = 3          # livello piramide (0=full; 3 = /8: texture nitida e leggera)
HE_FLIP  = False      # orientamento texture (validato: False)

# --- dimensioni pallini (interpolate tra le fasi) ---
PT_ESTAB = 10         # taglia sul tessuto (apertura / inizio zoom)
PT_BIG   = 40         # taglia "pop" dopo lo zoom
PT_NET   = 40         # taglia nella rete fluttuante (dopo il lift-off)
CTX_PT   = 16         # taglia cellule di contesto (le altre del campo largo)

# --- opacita' ---
CTX_OPACITY  = 1.00   # opacita' cellule di contesto quando appaiono
TISSUE_DIM   = 0.55   # opacita' "un po' trasparente" dell'H&E prima che sparisca
EDGE_OPACITY = 0.30   # opacita' archi

# --- archi: tubi (robusti, spessore sempre rispettato) oppure linee ---
USE_TUBES   = False
EDGE_RADIUS = 0.02    # spessore archi se USE_TUBES (0.015 sottile, 0.04 grosso)
EDGE_WIDTH  = 4.0     # spessore archi se NON tubi (ignorato in software rendering!)
EDGE_COLOR  = "#8fa6c8"

# --- camera ---
ELEV        = 33.0    # elevazione (gradi) in obliqua/orbita
R_ORBIT     = 15.0    # distanza camera in orbita
AZ0         = -25.0   # azimut iniziale orbita
LIFT_DRIFT  = 12.0    # deriva azimut durante il lift-off (gradi)
ORBIT_SWEEP = 165.0   # ampiezza dell'orbita (gradi)

# --- sfondo ---
BG_COLOR = "black"    # nero pieno. Per gradiente: BG_COLOR="#0c1726" e BG_TOP="#15273F"
BG_TOP   = None       # None = tinta piatta

# --- output / render ---
OUT       = "hero3d_final.mp4"
RES       = (1920, 1080)
FPS       = 30
QUALITY   = 9         # 0..10 (qualita' encoder)
ANTIALIAS = "ssaa"    # "ssaa" (qualita') | "fxaa" (veloce) | None

# --- TEMPI (secondi per fase) — alza per RALLENTARE ---
T_HE     = 1.2        # A: solo H&E
T_APPEAR = 1.6        # B: H&E semi-trasparente + pallini appaiono
T_ZOOM   = 3.0        # C: zoom nella tasca
T_HOLD     = 3.0      # C2: dopo lo zoom, quanto restano visibili i pallini di contesto
T_CTX_FADE = 1.0      # C3: durata della dissolvenza del contesto
T_EDGES  = 1.2        # D: archi compaiono
T_LIFT   = 3.0        # E: lift-off (betweenness) + H&E sparisce
T_ORBIT  = 8.0        # F: orbita

# --- colori/etichette classi (tab10[:8], ordine alfabetico) ---
COLORS = ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F"]
NAMES  = ["B","Endothelial","Fibroblasts","Myeloid","Normal-epi","SMC","T","Tumor"]

# ============================================================================
# 2) IMPORTS
# ============================================================================
import numpy as np
import pyvista as pv
import networkx as nx
import scipy.sparse as sp
import matplotlib.colors as mcolors
import zarr

# ============================================================================
# 3) DATI — carico G, sample 47, finestra hero, tasca
# ============================================================================
if not PROJECT_ROOT:
    for _cand in (".", ".."):
        if os.path.isdir(os.path.join(_cand, "src")) and os.path.isdir(os.path.join(_cand, "cache")):
            PROJECT_ROOT = _cand; break
    if not PROJECT_ROOT:
        PROJECT_ROOT = "."
SRC   = os.path.join(PROJECT_ROOT, "src")
CACHE = os.path.join(PROJECT_ROOT, "cache")
sys.path.insert(0, SRC)
import data_io, graph_construction

print("Carico il grafo globale…")
G    = data_io.load_global(CACHE)
sg   = graph_construction.build_for_sample(SAMPLE, global_arrays=G)
gidx = sg.global_idxs
pos_s = np.asarray(G.positions[gidx], float)        # (n_s,2) pixel livello-0
ct_s  = np.asarray(G.cell_type[gidx]).astype(int)
ei    = np.asarray(sg.edge_index)                   # (2,E) indici locali al sample

# finestra hero
half   = FRAC * (pos_s.max(0) - pos_s.min(0)) / 2.0
lo, hi = np.array(CENTER) - half, np.array(CENTER) + half
inwin  = ((pos_s[:, 0] >= lo[0]) & (pos_s[:, 0] <= hi[0]) &
          (pos_s[:, 1] >= lo[1]) & (pos_s[:, 1] <= hi[1]))
sel    = np.where(inwin)[0]
remap  = -np.ones(pos_s.shape[0], np.int64); remap[sel] = np.arange(sel.size)
emask  = inwin[ei[0]] & inwin[ei[1]]
und    = np.unique(np.sort(remap[ei[:, emask]], axis=0), axis=1)  # archi non orientati (hero-local)
pos    = pos_s[sel]                                  # (Nhero,2) cellule regione hero
ct     = ct_s[sel]                                   # (Nhero,)

# helper finestra/diversita' per la scelta della tasca
def win_mask(cx, cy, side):
    h = side / 2
    return (pos[:, 0] >= cx - h) & (pos[:, 0] < cx + h) & (pos[:, 1] >= cy - h) & (pos[:, 1] < cy + h)

def diversity(mask):
    n = int(mask.sum())
    if n == 0:
        return 0.0
    p = np.bincount(ct[mask], minlength=8) / n; p = p[p > 0]
    return float(-(p * np.log(p)).sum())

# scelta tasca
if POCKET_MANUAL_CENTER is not None:
    CX, CY, SIDE = POCKET_MANUAL_CENTER[0], POCKET_MANUAL_CENTER[1], POCKET_SIDE
else:
    best = None
    for side in POCKET_SIDES:
        xs = np.linspace(pos[:, 0].min() + side / 2, pos[:, 0].max() - side / 2, POCKET_GRID)
        ys = np.linspace(pos[:, 1].min() + side / 2, pos[:, 1].max() - side / 2, POCKET_GRID)
        for cx in xs:
            for cy in ys:
                n = int(win_mask(cx, cy, side).sum())
                if POCKET_TARGET[0] <= n <= POCKET_TARGET[1]:
                    H = diversity(win_mask(cx, cy, side))
                    if best is None or H > best[0]:
                        best = (H, cx, cy, side)
    assert best is not None, "Nessuna finestra nel range: allarga POCKET_TARGET/POCKET_SIDES."
    _, CX, CY, SIDE = best

m    = win_mask(CX, CY, SIDE)                         # maschera tasca su 'pos'
ssel = np.where(m)[0]
rmap = -np.ones(pos.shape[0], np.int64); rmap[ssel] = np.arange(ssel.size)
em   = m[und[0]] & m[und[1]]
E    = rmap[und[:, em]]                               # (2,Msub) archi tasca (pocket-local)
P    = pos[ssel].astype(float)                        # (Nsub,2) nodi tasca
C    = ct[ssel]                                       # (Nsub,)
print(f"Tasca: centro=({CX:.0f},{CY:.0f}) lato={SIDE:.0f}  ->  N={P.shape[0]} nodi, M={E.shape[1]} archi")

# ============================================================================
# 4) COORDINATE UNIFICATE (pixel livello-0 -> mondo) + RILIEVO betweenness
# ============================================================================
ORIGIN = np.array([CX, CY], float)   # centro tasca = origine del mondo (rete centrata in 0,0)
SCALE  = 5.0 / (SIDE / 2)            # la tasca occupa circa [-5, 5]

def to_world(xy):
    w = (np.asarray(xy, float) - ORIGIN) * SCALE
    w = w.copy(); w[:, 1] *= -1       # pixel y-giu' -> mondo y-su'
    return w

WXY = to_world(pos)                   # (Nhero,2) regione larga
XY  = to_world(P)                     # (Nsub,2)  tasca

# betweenness -> campo d'altezza smussato lungo il grafo
g = nx.Graph(); g.add_nodes_from(range(P.shape[0])); g.add_edges_from(E.T.tolist())
bc = nx.betweenness_centrality(g, normalized=True)
bc = np.array([bc[i] for i in range(P.shape[0])]); bc /= bc.max()
A  = sp.coo_matrix((np.ones(E.shape[1] * 2), (np.r_[E[0], E[1]], np.r_[E[1], E[0]])),
                   shape=(P.shape[0],) * 2).tocsr()
deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
zf = bc.copy()
for _ in range(SMOOTH_ITERS):
    zf = 0.5 * zf + 0.5 * (A @ zf) / deg
zf = ((zf - zf.min()) / (zf.max() - zf.min() + 1e-9)) ** RELIEF_EXP
zfull = BASE_LIFT + zf * Z_MAX        # altezza finale dei nodi

# ============================================================================
# 5) PIANO H&E (texture sulla regione larga)
# ============================================================================
f = 2 ** HE_LEVEL
series = zarr.open_group(f"{HE_DIR}/{SAMPLE_ID}.ome.zarr", mode="r")["0"]
arr    = series[str(HE_LEVEL)]                        # (1,3,1,Y,X) uint8
x0, y0 = pos.min(0); x1, y1 = pos.max(0)
xs0, xs1 = int(x0 // f), int(np.ceil(x1 / f))
ys0, ys1 = int(y0 // f), int(np.ceil(y1 / f))
crop = np.asarray(arr[0, :, 0, ys0:ys1, xs0:xs1]).transpose(1, 2, 0)   # (h,w,3)
if HE_FLIP:
    crop = np.flipud(crop)
pxc = np.array([[xs0 * f, ys0 * f], [xs1 * f, ys0 * f],
                [xs0 * f, ys1 * f], [xs1 * f, ys1 * f]], float)
wc  = to_world(pxc); wx0, wy0 = wc.min(0); wx1, wy1 = wc.max(0)
plane = pv.Plane(center=((wx0 + wx1) / 2, (wy0 + wy1) / 2, -0.05), direction=(0, 0, 1),
                 i_size=wx1 - wx0, j_size=wy1 - wy0)
tex = pv.numpy_to_texture(np.ascontiguousarray(crop))

# ============================================================================
# 6) MESH — nuvola di contesto + nodi/archi della tasca
# ============================================================================
oth = ~m
rgb_oth = (np.array([mcolors.to_rgb(COLORS[c]) for c in ct[oth]]) * 255).astype(np.uint8)
cloud = pv.PolyData(np.hstack([WXY[oth], np.zeros((oth.sum(), 1))])); cloud["rgb"] = rgb_oth

rgb   = (np.array([mcolors.to_rgb(COLORS[c]) for c in C]) * 255).astype(np.uint8)
lines = np.column_stack([np.full(E.shape[1], 2, np.int64), E[0], E[1]]).ravel()
mesh  = pv.PolyData(np.hstack([XY, np.zeros((P.shape[0], 1))]), lines=lines)
mesh["rgb"] = rgb

def make_tube():
    t = mesh.tube(radius=EDGE_RADIUS); t.clear_data()   # tolgo 'rgb' -> vince il colore solido
    return t

# ============================================================================
# 7) CAMERA — helper e keyframe
# ============================================================================
def cam(az, elev, r, fz):
    a, e = np.radians(az), np.radians(elev)
    return [(r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), fz + r * np.sin(e)),
            (0, 0, fz), (0, 0, 1)]

def lerp_cam(c0, c1, t):
    return [tuple((1 - t) * np.array(a) + t * np.array(b)) for a, b in zip(c0, c1)]

def smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

def den(n):
    return max(n - 1, 1)

wcx, wcy = WXY.mean(0)
R_WIDE   = (np.ptp(WXY, axis=0).max() / 2) / np.tan(np.radians(15)) * 1.15
CAM_WIDE = [(wcx, wcy, R_WIDE), (wcx, wcy, 0.0), (0.0, 1.0, 0.0)]   # top-down sul campo largo
fz       = zfull.mean() * 0.55
CAM_OBL  = cam(AZ0, ELEV, R_ORBIT, fz)                              # obliqua sulla tasca

# frame per fase (da secondi)
NA, NB, NC, NHOLD, NCTX, ND, NE, NF = (round(t * FPS) for t in
    (T_HE, T_APPEAR, T_ZOOM, T_HOLD, T_CTX_FADE, T_EDGES, T_LIFT, T_ORBIT))

# ============================================================================
# 8) PLOTTER + ATTORI
# ============================================================================
pl = pv.Plotter(off_screen=True, window_size=RES)
if BG_TOP:
    pl.set_background(BG_COLOR, top=BG_TOP)
else:
    pl.set_background(BG_COLOR)
if ANTIALIAS:
    try:
        pl.enable_anti_aliasing(ANTIALIAS)
    except Exception as e:
        print("[warn] anti-aliasing off:", e)

plane_a = pl.add_mesh(plane, texture=tex, opacity=1.0)
cloud_a = pl.add_mesh(cloud, style="points", render_points_as_spheres=True,
                      point_size=CTX_PT, scalars="rgb", rgb=True, opacity=0.0)
if USE_TUBES:
    edge_geom = make_tube()
    edge_a = pl.add_mesh(edge_geom, color=EDGE_COLOR, opacity=0.0)
else:
    edge_geom = None
    edge_a = pl.add_mesh(mesh, style="wireframe", color=EDGE_COLOR,
                         line_width=EDGE_WIDTH, opacity=0.0)
node_a = pl.add_mesh(mesh, style="points", render_points_as_spheres=True,
                     point_size=PT_ESTAB, scalars="rgb", rgb=True,
                     smooth_shading=True, opacity=0.0)

def zset(z):
    mesh.points = np.hstack([XY, np.asarray(z, float)[:, None]])

def refresh_edges():
    if USE_TUBES:
        edge_geom.copy_from(make_tube())   # i tubi non seguono i punti: vanno rigenerati

zset(np.zeros(P.shape[0]))

# ============================================================================
# 9) RENDER — le 6 fasi
# ============================================================================
print(f"Render: {NA + NB + NC + ND + NE + NF} frame @ {FPS}fps  ->  {OUT}")
pl.open_movie(OUT, framerate=FPS, quality=QUALITY)

# A — solo H&E (nessun pallino)
for _ in range(NA):
    pl.camera_position = CAM_WIDE
    pl.write_frame()

# B — H&E -> semi-trasparente + pallini APPAIONO (camera ferma, vista larga)
for i in range(NB):
    t = smooth(i / den(NB))
    plane_a.prop.opacity = 1 - (1 - TISSUE_DIM) * t
    cloud_a.prop.opacity = CTX_OPACITY * t
    node_a.prop.opacity  = t
    pl.camera_position = CAM_WIDE
    pl.write_frame()
plane_a.prop.opacity = TISSUE_DIM; cloud_a.prop.opacity = CTX_OPACITY; node_a.prop.opacity = 1.0

# C — zoom nella tasca (il contesto RESTA, i nodi crescono)
for i in range(NC):
    t = smooth(i / den(NC))
    pl.camera_position = lerp_cam(CAM_WIDE, CAM_OBL, t)
    node_a.prop.point_size = PT_ESTAB + (PT_BIG - PT_ESTAB) * t
    pl.write_frame()
node_a.prop.point_size = PT_BIG

# C2 — hold: il contesto resta visibile per T_HOLD secondi
for _ in range(NHOLD):
    pl.camera_position = CAM_OBL
    pl.write_frame()

# C3 — il contesto sparisce (restano i nodi della tasca = futuro grafo)
for i in range(NCTX):
    cloud_a.prop.opacity = CTX_OPACITY * (1 - smooth(i / den(NCTX)))
    pl.camera_position = CAM_OBL
    pl.write_frame()
cloud_a.prop.opacity = 0.0

# D — archi compaiono
for i in range(ND):
    edge_a.prop.opacity = EDGE_OPACITY * smooth(i / den(ND))
    pl.camera_position = CAM_OBL
    pl.write_frame()
edge_a.prop.opacity = EDGE_OPACITY

# E — lift-off: tessuto SPARISCE, grafo si alza, nodi -> PT_NET
for i in range(NE):
    t = smooth(i / den(NE))
    zset(t * zfull); refresh_edges()
    plane_a.prop.opacity   = TISSUE_DIM * (1 - t)
    node_a.prop.point_size = PT_BIG + (PT_NET - PT_BIG) * t
    pl.camera_position = cam(AZ0 + LIFT_DRIFT * t, ELEV, R_ORBIT, fz)
    pl.write_frame()
plane_a.prop.opacity = 0.0; node_a.prop.point_size = PT_NET; zset(zfull); refresh_edges()

# F — orbita
for i in range(NF):
    az = (AZ0 + LIFT_DRIFT) + ORBIT_SWEEP * (i / den(NF))
    pl.camera_position = cam(az, ELEV, R_ORBIT, fz)
    pl.write_frame()

pl.close()
print(f"FATTO: {os.path.abspath(OUT)}  ({round(os.path.getsize(OUT) / 1e6, 1)} MB)")

# chiudo l'eventuale display virtuale
if _vdisplay is not None:
    try:
        _vdisplay.stop()
    except Exception:
        pass