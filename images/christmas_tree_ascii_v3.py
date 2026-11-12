#!/usr/bin/env python3
"""Animated 3-D ASCII/particle Christmas tree: progressive trunk, reactive light pool, sparkling star.

Deterministic, fully procedural geometry. Creates an MP4 and a GitHub-ready GIF.
The reference video informs the timing and look, but none of its frames are copied.

Install: pip install numpy opencv-python pillow
Run: python christmas_tree_ascii_v3.py --output christmas_tree_ascii_v3.mp4
     python christmas_tree_ascii_v3.py --seconds 19.5 --fps 10 --width 520

Requires ffmpeg on PATH.  Images use mathematically projected points, not a
stock image or animation generated from prompts.
"""
from __future__ import annotations
import argparse
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SEED = 20261112
TREE_HEIGHT = 4.55
BASE_RADIUS = 1.54
ELEVATION = math.radians(17.5)


def ease(x: float) -> float:
    x = np.clip(x,0.,1.)
    return x*x*(3-2*x)


def in_range(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def foliage(rng: np.random.Generator):
    n = 25800
    y = rng.uniform(0.08, TREE_HEIGHT, n)
    a = rng.uniform(0, 2*math.pi, n)
    # Jagged needles and fourteen horizontal branch tiers: conical, not smooth.
    branch = (y / TREE_HEIGHT * 15.0) % 1.0
    radius = BASE_RADIUS*(1-y/TREE_HEIGHT)
    radius *= 0.92 + 0.12*(1-branch)**1.4
    radius *= 1 + .035*np.sin(a*9 + y*11) + .02*np.cos(a*17-y*16)
    inside = np.where(rng.random(n)<.19, rng.uniform(.28,.97,n), rng.uniform(.89,1.07,n))
    r = radius*inside
    x = r*np.sin(a)
    z = r*np.cos(a)
    xyz = np.column_stack((x,y,z)).astype(np.float32)
    # Colored ASCII 'pixels' with some white frost / Christmas-lights glints.
    palette = np.array([[9,44,23],[12,64,32],[15,82,41],[20,100,48],
                        [31,114,57],[49,136,72],[78,157,94],[123,190,142]],np.uint8)
    choices = rng.choice(len(palette),size=n,p=[.12,.16,.19,.2,.15,.1,.06,.02])
    col = palette[choices].copy()
    glints = rng.random(n) < .028
    col[glints] = rng.choice(np.array([[158,211,167],[205,225,211],[252,241,195]],np.uint8), size=glints.sum())
    sizes = rng.choice([0,1,2],n,p=[.26,.55,.19]).astype(np.uint8)
    phase = rng.uniform(0,2*math.pi,n).astype(np.float32)
    # Needles have individual activation times. Correlated angular waves plus independent
    # jitter avoid a rigid, horizontal ring rising along the trunk.
    progress=y/TREE_HEIGHT
    birth=(2.04+2.10*progress**1.05+0.58*rng.random(n)
           +0.12*np.sin(a*6.0+y*4.6)+0.10*np.sin(a*13.0-y*7.0))
    return xyz,col,sizes,phase,birth.astype(np.float32)



def trunk(rng: np.random.Generator):
    """Full-height wooden core growing as asynchronous points, not a rigid pillar wipe."""
    n=3200
    height=TREE_HEIGHT-.19
    y=rng.uniform(.018,height,n)
    angle=rng.uniform(0,2*np.pi,n)
    # Surface of a gently tapering wood cable; uneven shading suggests fibers.
    radius=(.062-.022*y/height)*rng.uniform(.72,1.26,n)
    xyz=np.column_stack((radius*np.cos(angle),y,radius*np.sin(angle))).astype('float32')
    tone=np.array([[64,34,19],[93,46,24],[126,70,34],
                   [159,99,53],[105,56,30],[185,125,73]],dtype='uint8')
    colors=tone[rng.choice(6,n,p=[.15,.28,.22,.09,.22,.04])]
    sizes=rng.choice([0,1,2],n,p=[.25,.61,.14]).astype('uint8')
    # Spread independent activations across the entire height; no perfectly flat front.
    progress=y/height
    birth=(.69+1.00*progress**1.12+.13*rng.random(n)
           +.04*np.sin(y*10+angle*4)+.023*np.cos(y*18-angle*7))
    return xyz,colors,sizes,birth.astype('float32')


def garland(rng: np.random.Generator):
    # Fine golden fabric winding ~4.15 turns from the lowest branches to the top.
    n=2950
    u = rng.random(n)
    y=.28+u*(TREE_HEIGHT-.51)
    theta = -2*math.pi*(4.2*u)+.52
    r = BASE_RADIUS*(1-y/TREE_HEIGHT)*(1.015+.015*rng.normal(size=n))
    # Golden ribbon has small width, made of horizontal scattered highlights.
    y += rng.normal(0,.034,n)
    r += rng.normal(0,.027,n)
    xyz=np.stack((r*np.sin(theta),y,r*np.cos(theta)),axis=1).astype(np.float32)
    cols=np.array([[247,189,50],[245,221,105],[171,111,40],
                   [255,242,154],[184,144,50]],np.uint8)
    color=cols[rng.choice(len(cols),n,p=[.3,.27,.21,.09,.13])]
    sizes=rng.choice([0,1,2],n,p=[.2,.63,.17]).astype(np.uint8)
    return xyz,color,sizes,u


def bulb_positions(rng):
    n = 116
    y = rng.uniform(.2,TREE_HEIGHT-.35,n)
    a = rng.uniform(0,2*math.pi,n)
    r = BASE_RADIUS*(1-y/TREE_HEIGHT)*1.08
    xyz = np.column_stack((r*np.sin(a),y,r*np.cos(a))).astype('float32')
    colors=np.array([[249,220,99],[255,245,205],[240,192,75],
                     [223,63,54],[156,228,201]],np.uint8)
    idx=rng.choice(5,n,p=[.35,.28,.19,.12,.06]); return xyz,colors[idx],rng.uniform(0,6.28,n)


def ornaments(rng):
    n=37
    y=rng.uniform(.38,3.55,n)
    a=rng.uniform(0,2*math.pi,n)
    r=BASE_RADIUS*(1-y/TREE_HEIGHT)*1.12
    xyz=np.column_stack((r*np.sin(a),y,r*np.cos(a))).astype('float32')
    colors=np.array([[196,27,34],[239,57,43],[252,201,56],[245,165,50],[238,237,207],[164,45,39]],np.uint8)
    size=rng.uniform(.042,.09,n)
    return xyz,colors[rng.choice(6,n,p=[.19,.24,.18,.19,.08,.12])],size


def presents(rng):
    all_pts=[];all_colors=[];all_sizes=[];ids=[]
    specs=[
     (1.55,.28,0.44,0.38,0.40,(173,26,31),(255,205,73)),
     (-1.7,-.32,0.41,0.32,0.38,(230,217,191),(200,48,40)),
     (1.28,1.22,0.37,0.50,0.36,(22,69,149),(239,214,92)),
     (-1.05,1.54,0.40,0.37,0.43,(170,32,29),(231,217,86)),
     (.15,1.85,0.50,0.35,0.36,(27,89,69),(221,186,63)),
     (-.22,-1.74,0.46,0.38,0.42,(189,31,36),(242,193,75)),
     (1.05,-1.32,.32,.27,.28,(225,211,202),(206,42,39)),
    ]
    # Each gift is a tiny colored ASCII cuboid with crossed ribbons and a bow.
    for gift_id,(x,z,w,h,d,body,ribbon) in enumerate(specs):
      x,z=x*.89,z*.89
      w,h,d=w*.85,h*.85,d*.85
      for face in range(5):
        n=155 if face!=4 else 240
        u=rng.uniform(-.5,.5,n); v=rng.uniform(-.5,.5,n)
        if face==0: xx=x+u*w; yy=.025+(v+.5)*h; zz=np.full(n,z-d/2)
        elif face==1: xx=x+w/2+u*.0; yy=.025+(v+.5)*h; zz=z+u*d
        elif face==2: xx=x-w/2+u*.0; yy=.025+(v+.5)*h; zz=z+u*d
        elif face==3: xx=x+u*w; yy=.025+(v+.5)*h; zz=np.full(n,z+d/2)
        else: xx=x+u*w; yy=np.full(n,h+.025); zz=z+v*d
        pts=np.column_stack((xx,yy,zz)).astype(np.float32)
        norm_u=(xx-x)/w;norm_z=(zz-z)/d
        if face==4: isribbon=(abs(norm_u)<.095)|(abs(norm_z)<.095)
        elif face in (0,3):isribbon=(abs(norm_u)<.095)
        else: isribbon=(abs(norm_z)<.095)
        cc=np.array(body)[None,:]*rng.uniform(.55,1.08,(n,1))
        cc=np.clip(cc,0,255).astype('uint8')
        cc[isribbon]=ribbon
        all_pts.append(pts);all_colors.append(cc)
        all_sizes.append(rng.choice([0,1,2],n,p=[.22,.65,.13]).astype('uint8'))
        ids.append(np.full(n,gift_id))
      # little ribbon bow above the box
      u=np.linspace(0,2*np.pi,72)
      for sign in (-1,1):
       bx=x+sign*.10+np.cos(u)*.105
       by=np.full(72,h+.1)+np.sin(u)*.065
       bz=np.full(72,z)
       all_pts.append(np.column_stack((bx,by,bz)).astype('float32'))
       all_colors.append(np.tile(np.array(ribbon,dtype='uint8'),(72,1)))
       all_sizes.append(np.full(72,1,dtype='uint8'))
       ids.append(np.full(72,gift_id))
    return (np.concatenate(all_pts),np.concatenate(all_colors),np.concatenate(all_sizes),np.concatenate(ids))


class Scene:
 def __init__(self,width=520,height=None,seed=SEED):
    self.W=width; self.H=height or 2*round((width*361/341)/2)
    self.rng=np.random.default_rng(seed)
    self.tree=foliage(self.rng)
    self.trunk=trunk(self.rng)
    self.garland=garland(self.rng)
    self.bulbs=bulb_positions(self.rng)
    self.ornaments=ornaments(self.rng)
    self.gifts=presents(self.rng)
    self.scale=self.W/520*89
    self.cx=self.W*.5; self.ground_y=self.H*.895
    self.sky=self.make_sky()
    self.snow=self.make_snow()
    self.flare_birth=self.rng.uniform(6.25,8.03,len(self.ornaments[0]))
    self.gift_birth=np.array([9.70,9.45,10.05,9.95,10.35,9.50,10.58],dtype=np.float32)
    self.zoom=1.0
    self._brush_cache={}


 def make_sky(self):
    y=np.linspace(0,1,self.H)[:,None,None]
    base=np.array([5,10,19],float)[None,None,:] + y*np.array([0,3,5],float)[None,None,:]
    # dark teal glow is barely perceptible around the base
    a=np.empty((self.H,self.W,3),dtype=np.uint8)
    a[:]=np.clip(base,0,255).astype('uint8')
    rr=np.random.default_rng(3939)
    xs=rr.integers(1,self.W-1,210)
    ys=rr.integers(0,self.H-6,210)
    b=rr.choice([25,34,45,60,81,125,180],size=len(xs),p=[.1,.20,.24,.19,.14,.09,.04])
    for x,yy,v in zip(xs,ys,b):
      a[yy,x]=[int(v*.72),int(v*.85),int(v)]
    self.stars_xy=(xs,ys,b)
    return a

 def make_snow(self):
    rng=np.random.default_rng(86042026)
    count=305
    layer=rng.random(count)
    return dict(x=rng.uniform(0,self.W,count), y=rng.uniform(0,self.H,count),
       velocity=np.where(layer<.67,rng.uniform(13,44,count),rng.uniform(45,87,count))*(self.W/520),
       drift=rng.uniform(-15,15,count)*(self.W/520),
       amplitude=rng.uniform(1.5,12.5,count)*(self.W/520),
       rate=rng.uniform(.3,1.4,count),phase=rng.uniform(0,2*np.pi,count),
       bright=np.where(layer<.67,rng.uniform(70,160,count),rng.uniform(124,242,count)),
       radius=np.where(layer<.76,0,np.where(layer<.975,1,2)))

 def glow(self,img,x,y,radius=14,strength=.7,color=(255,252,232)):
    """Localized additive bloom in RGB, avoiding full-frame blurs per ornament."""
    r=max(3,int(radius)); k=min(30,r)
    if k not in self._brush_cache:
      yy,xx=np.mgrid[-k:k+1,-k:k+1]
      self._brush_cache[k]=np.exp(-.5*(xx*xx+yy*yy)/(max(1,k*.43)**2)).astype('float32')
    xa=max(0,x-k); xb=min(self.W,x+k+1); ya=max(0,y-k);yb=min(self.H,y+k+1)
    if xb<=xa or yb<=ya:return
    ka=xa-(x-k);kb=ka+(xb-xa);la=ya-(y-k);lb=la+(yb-ya)
    sub=img[ya:yb,xa:xb]
    gl=self._brush_cache[k][la:lb,ka:kb,None]
    rgb=np.asarray(color,dtype='float32')[None,None,:]
    sub[:]=np.clip(sub.astype('float32')+strength*gl*rgb,0,255).astype('uint8')

 def snowfall(self,img,t):
    """Slow, fast, swaying and almost straight falling flakes at several depths."""
    sn=self.snow
    xs=np.mod(sn['x']+sn['drift']*t+sn['amplitude']*np.sin(sn['rate']*t+sn['phase'])
              +.23*sn['amplitude']*np.sin(2.4*sn['rate']*t+sn['phase']*1.7),self.W)
    ys=np.mod(sn['y']+sn['velocity']*t,self.H)
    for x,y,b,rad in zip(xs.astype(int),ys.astype(int),sn['bright'],sn['radius']):
      if y<1 or y>=self.H-2 or x<2 or x>=self.W-2:continue
      val=int(b*(.90+.10*math.cos(t*1.1+x*.012)))
      if rad==0:img[y,x]=[int(val*.82),int(val*.91),val]
      elif rad==1:
        img[y,x]=[int(val*.90),int(val*.95),val]
        img[y+1,x]=[int(val*.42),int(val*.48),int(val*.56)]
      else:
        img[y,x]=[min(255,val),min(255,val),min(255,val)]
        img[y-1,x]=[val//2,val//2,val//2]
        img[y+1,x]=[val//3,val//3,val//3]

 def project(self, pts,angle):
    ca,sa=math.cos(angle),math.sin(angle)
    depth=pts[:,0]*sa+pts[:,2]*ca
    side=pts[:,0]*ca-pts[:,2]*sa
    perspective=np.clip(11/(11-depth),.45,2.25)
    x=self.cx+side*self.scale*perspective
    y=self.ground_y-(pts[:,1]*math.cos(ELEVATION)-depth*math.sin(ELEVATION))*self.scale*perspective
    return x,y,depth

 def floor(self,img,angle,t):
    # Dotted mesh in perspective; retains a printed-terminal horizontal texture.
    ca,sa=math.cos(angle),math.sin(angle)
    for depth in np.arange(-7.5,6.6,.29):
      xs=np.arange(-7,7.03,.15)
      pts=np.stack((xs*ca+depth*sa,np.zeros_like(xs),-xs*sa+depth*ca),axis=1)
      xx,yy,dd=self.project(pts,angle)
      shade=int(21+19*(depth+7.5)/14)
      for x,y in zip(xx.astype(int),yy.astype(int)):
       if 0<=x<self.W-2 and int(self.H*.69)<y<self.H:
        img[y,x:x+2]=[shade,shade+5,shade+9]
    for sidex in np.arange(-6.5,7,.52):
      z=np.arange(-8,7,.12)
      pts=np.stack((sidex*ca+z*sa,np.zeros_like(z),-sidex*sa+z*ca),axis=1)
      xx,yy,dd=self.project(pts,angle)
      for x,y in zip(xx.astype(int),yy.astype(int)):
       if 0<=x<self.W-1 and int(self.H*.69)<y<self.H:
        img[y,x]=[19,27,36]

 def print_points(self,img,pts,cols,sizes,angle,mask=None,twinkle=None):
    if mask is not None:
     pts=pts[mask];cols=cols[mask];sizes=sizes[mask]
     if twinkle is not None: twinkle=twinkle[mask]
    if len(pts)==0: return
    xx,yy,depth=self.project(pts,angle)
    x=np.rint(xx).astype('int32'); y=np.rint(yy).astype('int32')
    use=(x>=2)&(x<self.W-3)&(y>=2)&(y<self.H-3)
    x=x[use];y=y[use];depth=depth[use]; cols=cols[use];sizes=sizes[use]
    if twinkle is not None: twinkle=twinkle[use]
    # Draw back to front. Short dash + plus signs are the 'ASCII' strokes.
    order=np.argsort(depth)
    for j in order:
      xi,yi=int(x[j]),int(y[j]); cc=cols[j]
      if twinkle is not None:
       flash=twinkle[j]
       cc=np.clip(cc.astype('float32')*flash,0,255).astype('uint8')
      s=int(sizes[j])
      if s==0: img[yi,xi]=cc
      elif s==1: img[yi,xi:xi+2]=cc
      else:
       img[yi,xi:xi+3]=cc
       img[yi-1,xi+1]=cc

 def ground_ring(self,img,angle,t):
    """Persistent breathing white pool, brightest under the tree, dim at its rim.

    Its elliptical footprint is a physical circle projected onto the 3-D ground.
    Colored illumination grows in as the pine, ribbon, ornaments and gifts appear.
    """
    reveal=ease((t-.09)/.67)
    if reveal<=0:return
    breathing=1.0+.080*math.sin(t*1.13+.30)+.028*math.sin(t*2.17+.45)
    radius=2.08*(.15+.85*reveal)*breathing
    aa=np.linspace(0,2*math.pi,256,endpoint=False)
    p=np.column_stack((radius*np.cos(aa),np.full_like(aa,.018),radius*np.sin(aa)))
    xx,yy,_=self.project(p,angle)
    poly=np.column_stack((xx,yy)).astype('int32')
    x0=max(0,int(min(xx)-22)); x1=min(self.W,int(max(xx)+23))
    y0=max(0,int(min(yy)-22)); y1=min(self.H,int(max(yy)+23))
    if x0>=x1 or y0>=y1:return
    hh=y1-y0;ww=x1-x0
    localpoly=poly-np.array([x0,y0],dtype='int32')
    fill=np.zeros((hh,ww),dtype='uint8')
    cv2.fillPoly(fill,[localpoly],255,cv2.LINE_AA)
    # Only blur a cropped region, so full animation stays inexpensive.
    soft=cv2.GaussianBlur(fill,(0,0),4.5).astype('float32')/255.0
    ys,xs=np.mgrid[y0:y1,x0:x1]
    cx,cy=self.cx,self.ground_y
    rx=max(1,(max(xx)-min(xx))*.50)
    ry=max(1,(max(yy)-min(yy))*.50)
    ux=(xs-cx)/rx; uy=(ys-cy)/ry
    rho2=ux*ux+uy*uy
    core=np.exp(-3.25*rho2)
    midway=np.exp(-1.25*rho2)
    # The introductory flash fades into a visible ambient light pool; it never vanishes.
    intro=1-ease((t-1.42)/1.48)
    ambience=(.91*intro+.39*(1-intro))*reveal
    ambience*=.95+.065*math.sin(t*1.13+.30)
    # White heart and soft, darker outside, with a faint creamy edge.
    white=(15.0+177.0*core+19.0*midway)*soft*ambience
    tint=np.zeros((hh,ww,3),dtype='float32')
    tint+=white[:,:,None]*np.array([.89,.94,1.0],dtype='float32')
    green=ease((t-2.08)/1.75)
    gold=ease((t-4.75)/1.15)
    red=ease((t-6.2)/2.1)
    gifts=ease((t-9.5)/1.3)
    # Soft multi-color reflections around the trunk, following camera orientation.
    gl=np.exp(-3.7*((ux+.20)**2+(uy+.07)**2))
    gl2=np.exp(-3.2*((ux-.32)**2+(uy-.07)**2))
    gl3=np.exp(-4.4*((ux+.05)**2+(uy-.43)**2))
    gl4=np.exp(-4.0*((ux-.45)**2+(uy+.39)**2))
    tint+=soft[:,:,None]*gl[:,:,None]*green*np.array([8,33,12],dtype='float32')
    tint+=soft[:,:,None]*gl2[:,:,None]*gold*np.array([44,29,5],dtype='float32')
    tint+=soft[:,:,None]*gl3[:,:,None]*red*np.array([31,4,5],dtype='float32')
    tint+=soft[:,:,None]*gl4[:,:,None]*gifts*np.array([6,13,35],dtype='float32')
    img[y0:y1,x0:x1]=np.clip(img[y0:y1,x0:x1].astype('float32')+tint,0,255).astype('uint8')
    # Perimeter has a handful of ASCII dashes; it never becomes a hard white ring.
    rim=int(np.clip((52*intro+25)*reveal,0,255))
    for ix,iy in zip(xx.astype(int)[::4],yy.astype(int)[::4]):
      if 2<ix<self.W-3 and 2<iy<self.H-2:
        img[iy,ix:ix+2]=np.maximum(img[iy,ix:ix+2],np.array([rim,rim,rim+4],dtype='uint8'))

 def trunk_completion_flash(self,img,angle,t):
    """Short white flash travelling around the finished wooden core, cueing foliage."""
    energy=ease((t-1.74)/.16)*(1-ease((t-2.15)/.42))
    if energy<.002:return
    length=TREE_HEIGHT-.19
    y=np.linspace(.08,length,45)
    xyz=np.column_stack((np.zeros_like(y),y,np.zeros_like(y))).astype('float32')
    xx,yy,_=self.project(xyz,angle)
    # Flickering long thread along the trunk, with small traveling white sparks.
    for j in range(len(y)):
      x=int(xx[j]); cy=int(yy[j]);
      if not(5<x<self.W-5 and 4<cy<self.H-4):continue
      phase=t*19+j*1.31
      flick=.57+.43*math.sin(phase)**2
      f=energy*flick
      if j%3==0:self.glow(img,x,cy,10,f*.30,(244,252,255))
      if j%2==0:
        v=min(255,int(145*f))
        img[cy,x-2:x+3]=np.maximum(img[cy,x-2:x+3],np.array([v,v,v],dtype='uint8'))
    x0=int(xx[-1]); y0=int(yy[-1]);
    if 8<x0<self.W-8 and 8<y0<self.H-8:
      self.glow(img,x0,y0,25,.82*energy,(253,252,248))
      r=int(9+6*energy)
      for a in np.arange(0,2*math.pi,math.pi/4):
        dx=int(math.cos(a)*r);dy=int(math.sin(a)*r)
        value=int(214*energy)
        cv2.line(img,(x0-dx//3,y0-dy//3),(x0+dx,y0+dy),
                 (value,value,min(255,value+15)),1,cv2.LINE_AA)

 def growth_shimmers(self,img,angle,t):
    """Several independent green/white tips; never a single rising disk."""
    xyz,col,sz,ph,birth=self.tree
    new=(birth<=t)&(birth>t-.25)
    ids=np.flatnonzero(new)
    if len(ids)==0:return
    # Only a few fresh tips flicker white; foliage grows point-by-point.
    ids=ids[::max(1,len(ids)//130)]
    x,y,depth=self.project(xyz[ids],angle)
    for xi,yi,dep in zip(x.astype(int),y.astype(int),depth):
      if 3<xi<self.W-3 and 3<yi<self.H-3 and dep>-.35:
        img[yi,xi:xi+2]=[168,208,182]

 def glowing_star(self,img,angle,t):
    """Slowly breathing five-point gold star with short, irregular white glints."""
    visible=ease((t-8.47)/.72)
    if visible<=0:return
    pos=np.array([[0,TREE_HEIGHT+.20,0]],dtype='float32')
    x,y,_=self.project(pos,angle)
    x=int(x[0]);y=int(y[0])
    pulse=1.0+.15*math.sin(2*math.pi*.73*t+.4)+.055*math.sin(2*math.pi*.27*t)
    pulse=max(.76,pulse)
    # Isolated twinkles, rather than continual strobing.
    burst_times=(9.65,11.90,14.45,16.65,18.55)
    burst=max(math.exp(-((t-v)/.23)**2) for v in burst_times)
    halo=.26*visible*(.77+.23*pulse)+.32*burst*visible
    self.glow(img,x,y,25,halo,(255,197,67))
    # Five visible golden points with pale center, slightly resizing each frame.
    outer=max(5,int(12.6*visible*pulse))
    inner=max(2,int(outer*.40))
    vertices=[]
    for k in range(10):
      theta=-math.pi/2+k*math.pi/5
      r=outer if k%2==0 else inner
      vertices.append([int(x+r*math.cos(theta)),int(y+r*math.sin(theta))])
    poly=np.array(vertices,dtype='int32')
    cv2.fillPoly(img,[poly],(214,169,54),cv2.LINE_AA)
    cv2.polylines(img,[poly],True,(255,224,115),1,cv2.LINE_AA)
    ray=int(self.W*.045*visible*pulse*(1+.25*burst))
    for j,theta in enumerate(np.linspace(0,2*math.pi,8,endpoint=False)):
      x2=int(x+math.cos(theta)*ray);y2=int(y+math.sin(theta)*ray)
      alpha=.55+.45*math.sin(t*1.23+j*.65)**2
      rgb=(int(230*visible*alpha),int(191*visible*alpha),int(80*visible*alpha))
      cv2.line(img,(x,y),(x2,y2),rgb,1,cv2.LINE_AA)
    cv2.circle(img,(x,y),max(2,int(4.2*visible*pulse)),(255,235,153),-1,cv2.LINE_AA)
    cv2.circle(img,(x,y),max(1,int(2.2*visible*pulse)),(255,254,236),-1,cv2.LINE_AA)
    if burst>.045:
      # Short white flashes, plus detached sparks drifting 8–28 pixels out.
      for j,phi in enumerate((.19,1.25,2.55,3.75,5.05)):
        r=(outer+6+j*3)*(.78+.32*math.sin(t*10+j))
        px=int(x+math.cos(phi)*r);py=int(y+math.sin(phi)*r)
        if 4<px<self.W-4 and 4<py<self.H-4:
          weight=burst*(.72+.25*math.sin(j*3+5*t)**2)
          lum=int(248*weight)
          cv2.line(img,(px-2,py),(px+2,py),(lum,lum,lum),1,cv2.LINE_AA)
          cv2.line(img,(px,py-2),(px,py+2),(lum,lum,lum),1,cv2.LINE_AA)
      ray2=int(ray*(1+.26*burst))
      for theta in (0,math.pi/2,math.pi,3*math.pi/2):
        dx=int(math.cos(theta)*ray2);dy=int(math.sin(theta)*ray2)
        cv2.line(img,(x,y),(x+dx,y+dy),
                 (int(255*burst),int(247*burst),int(213*burst)),1,cv2.LINE_AA)

 def frame(self,t):
    # Slowly orbiting viewpoint + small simultaneous in/out camera zoom.
    angle=.34+t*.173+.026*math.sin(t*.43)
    self.zoom=1.0+.018*math.sin(t*.98)+.012*math.sin(t*2.16+1.2)
    img=self.sky.copy()
    self.floor(img,angle,t)
    self.ground_ring(img,angle,t)

    # Wood fibers appear at different moments from bottom to tip.
    tp,tc,ts,tbirth=self.trunk
    alive_trunk=tbirth<=t
    if np.any(alive_trunk):
      pts=tp[alive_trunk]; colors=tc[alive_trunk].astype('float32')
      # The finished foliage naturally covers the wooden cable from the outside.
      # Dim already-covered regions without making the early trunk disappear.
      covered=ease((t-(2.05+2.08*pts[:,1]/TREE_HEIGHT))/.78)
      colors*=(1-.67*covered)[:,None]
      ages=t-tbirth[alive_trunk]
      tip=(1-ease(np.clip(ages/.23,0,1)))[:,None]
      colors=colors*(1-.31*tip)+np.array([224,186,138],dtype='float32')[None,:]*(.31*tip)
      self.print_points(img,pts,np.clip(colors,0,255).astype('uint8'),ts[alive_trunk],angle)
    self.trunk_completion_flash(img,angle,t)

    # Each green needle has its own birth time and short white-green sprout phase.
    tree,col,sizes,phase,birth=self.tree
    alive=birth<=t
    if np.any(alive):
      age=t-birth[alive]
      sprout=1-np.clip(age/.52,0,1)
      sprout=ease(sprout)
      green=col[alive].astype('float32')
      new_color=green*(1-.69*sprout[:,None])+np.array([210,235,215])[None,:]*(.69*sprout[:,None])
      settling=.63+.37*ease(np.minimum(1,age/.4))
      new_color*=settling[:,None]
      sparkle=.88+.11*np.sin(1.6*t+phase[alive])
      self.print_points(img,tree[alive],np.clip(new_color,0,255).astype('uint8'),sizes[alive],angle,twinkle=sparkle)
      if 1<t<5.3:self.growth_shimmers(img,angle,t)

    # Ribbon: its leading edge flares WHITE, then the material warms to GOLD.
    pp,base,ss,u=self.garland
    ages=t-(4.75+1.12*u)
    active=ages>=0
    if np.any(active):
      a=ages[active]
      gold=base[active].astype('float32')
      whitening=(1-ease(np.minimum(1,a/.36)))[:,None]
      color=gold*(1-whitening)+np.array([255,253,242])[None,:]*whitening
      self.print_points(img,pp[active],np.clip(color,0,255).astype('uint8'),ss[active],angle)
      # Bloom is concentrated around the new ribbon, not across its entire length.
      fresh=(ages>=0)&(ages<.19)
      IDs=np.flatnonzero(fresh)[::max(1,int(np.sum(fresh)//23))]
      if len(IDs)>0:
        x,y,z=self.project(pp[IDs],angle)
        for xi,yi,zz in zip(x.astype(int),y.astype(int),z):
          if 3<xi<self.W-3 and 3<yi<self.H-3 and zz>-.2:
            self.glow(img,xi,yi,9,.24,(255,252,232))

    # Electric fairy lights emerge one after another, some flickering softly.
    lights=ease((t-5.65)/1.75)
    if lights>0:
      pts,cols,ph=self.bulbs
      ids=np.arange(len(pts))
      born=5.78+1.45*((ids*31)%len(pts))/len(pts)
      mask=t>=born
      x,y,z=self.project(pts[mask],angle)
      for xx,yy,dep,c,p in zip(x.astype(int),y.astype(int),z,cols[mask],ph[mask]):
        if not(3<xx<self.W-3 and 3<yy<self.H-3) or dep<-.2:continue
        flick=.74+.26*math.sin(t*3.1+p)**2
        color=tuple(int(v*flick) for v in c)
        cv2.line(img,(xx-1,yy),(xx+1,yy),color,1,cv2.LINE_AA)
        img[yy-1,xx]=color

    # Each ornament flashes white, then materializes into a shaded/glossy ball.
    pos,colors,radw=self.ornaments
    ages=t-self.flare_birth
    mask=(ages>=0)
    if np.any(mask):
      x,y,z=self.project(pos[mask],angle)
      idx=np.flatnonzero(mask)
      for xx,yy,dep,k in zip(x.astype(int),y.astype(int),z,idx):
        if dep<-.12 or not(6<xx<self.W-6 and 6<yy<self.H-6):continue
        age=float(ages[k]); transition=ease(age/.43)
        rad=max(2,int(radw[k]*self.scale*.74))
        basec=colors[k].astype('float32')
        mix=basec*transition+np.array([253,251,241])*(1-transition)
        cc=tuple(int(v) for v in mix)
        dark=tuple(int(v*.54) for v in cc)
        cv2.circle(img,(xx,yy),rad,dark,-1,cv2.LINE_AA)
        cv2.circle(img,(xx-1,yy-1),max(1,rad-1),cc,-1,cv2.LINE_AA)
        # Fine gold-red texture and white specular highlight on the shell.
        if age>.22:
          cv2.circle(img,(xx+rad//3,yy),1,tuple(int(v*.7) for v in cc),-1)
          cv2.circle(img,(xx-1,yy-rad//2),1,(255,243,204),-1)
          if rad>=4: img[yy+rad//2,xx]=[int(v*.72) for v in cc]
        if age<.53:
          self.glow(img,xx,yy,10+rad*2,.94*(1-ease(age/.53)),(255,255,246))
          if age<.21:
            cv2.line(img,(xx-6,yy),(xx+6,yy),(240,241,236),1,cv2.LINE_AA)
            cv2.line(img,(xx,yy-6),(xx,yy+6),(240,241,236),1,cv2.LINE_AA)

    self.glowing_star(img,angle,t)

    # Presents materialize one by one as white halos, then reveal their pigments,
    # ribbon crosses and little gold bows.
    gp,gc,gs,gid=self.gifts
    ages=t-self.gift_birth[gid]
    active=ages>=0
    if np.any(active):
      age=np.maximum(0,ages[active])
      transition=ease(np.minimum(1,age/.58))
      base=gc[active].astype('float32')
      white=np.array([253.,251.,241.])[None,:]
      cols=white*(1-transition[:,None])+base*transition[:,None]
      self.print_points(img,gp[active],np.clip(cols,0,255).astype('uint8'),gs[active],angle)
      for gift_id in range(7):
        age=t-self.gift_birth[gift_id]
        if 0<=age<.58:
          points=gp[gid==gift_id]
          center=np.mean(points,axis=0)[None,:]
          xx,yy,_=self.project(center,angle)
          self.glow(img,int(xx[0]),int(yy[0]),18,.80*(1-ease(age/.58)),(255,250,233))

    # Film grain / CRT-like raster, matching the character-based reference.
    img[1::4]=(img[1::4].astype('float32')*.94).astype('uint8')

    # Breathing zoom is applied to the entire rendered scene, not just the tree.
    M=cv2.getRotationMatrix2D((self.W*.5,self.H*.46),0,self.zoom)
    img=cv2.warpAffine(img,M,(self.W,self.H),flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=(4,9,17))
    # Snow falls throughout the scene, with curved, drifting trajectories.
    self.snowfall(img,t)
    return img


def convert(video, gif, width=420, fps=10):
    video=Path(video); gif=Path(gif)
    if not video.exists():raise FileNotFoundError(video)
    palette_path=gif.with_suffix('.palette.png')
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(video),
                    '-vf',f'fps={fps},scale={width}:-1:flags=lanczos,palettegen=max_colors=160:stats_mode=full',str(palette_path)],check=True)
    pal=np.array(Image.open(palette_path).convert('RGB')).reshape(-1,3)
    # PaletteGen reserves at least one slot for transparency; actual sampled colors are shared.
    pal=np.unique(pal,axis=0)
    pal=pal.astype(np.uint8)
    if len(pal)>255:pal=pal[:255]
    if len(pal)<2:raise RuntimeError('Palette unexpectedly empty')
    print('palette colors',len(pal),flush=True)
    # 32768-bin fast nearest-neighbor table. Each pixel can then be mapped in O(1).
    all_r,all_g,all_b=np.mgrid[0:32,0:32,0:32]
    centers=np.stack((all_r,all_g,all_b),axis=-1).reshape(-1,3).astype(np.int32)*8+4
    bestdist=np.full(len(centers),2147483647,dtype=np.int32)
    lut=np.zeros(len(centers),dtype=np.uint8)
    for k,col in enumerate(pal):
        d=np.sum((centers-col.astype(np.int32))**2,axis=1)
        update=d<bestdist
        lut[update]=k;bestdist[update]=d[update]
    colors=np.zeros((256,3),dtype=np.uint8)
    colors[:len(pal)]=pal
    palette_flat=colors.ravel().tolist()
    cap=cv2.VideoCapture(str(video))
    frames=[]
    while True:
        success,frame=cap.read()
        if not success:break
        h,w=frame.shape[:2]
        nh=round(h*width/w)
        if w!=width:frame=cv2.resize(frame,(width,nh),interpolation=cv2.INTER_AREA)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        b0=frame[:,:,0].astype(np.uint16)>>3
        b1=frame[:,:,1].astype(np.uint16)>>3
        b2=frame[:,:,2].astype(np.uint16)>>3
        index=lut[b0*1024+b1*32+b2]
        output=Image.fromarray(index,mode='P')
        output.putpalette(palette_flat)
        frames.append(output)
        if len(frames)%40==0:print('GIF frames decoded:',len(frames),flush=True)
    cap.release()
    if not frames:raise RuntimeError('no input video frames')
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=int(round(1000/fps)),
                   loop=0,optimize=False,disposal=2)
    palette_path.unlink(missing_ok=True)
    print(f'GIF written {gif} ({gif.stat().st_size/1048576:.2f} MiB, {len(frames)} frames)',flush=True)


def run(output, width, fps, seconds, preview):
    if not shutil.which('ffmpeg'):
      raise RuntimeError('ffmpeg not found. Install ffmpeg and try again.')
    scene=Scene(width)
    command=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo',
             '-pix_fmt','rgb24','-s',f'{scene.W}x{scene.H}',
             '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
             '-pix_fmt','yuv420p','-crf','17','-preset','medium',str(output)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
      for i in range(round(seconds*fps)):
        frame=scene.frame(i/fps)
        if i==round(11.5*fps):cv2.imwrite(str(preview),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        proc.stdin.write(frame.tobytes())
        if i%20==0:print(f'Frame {i+1}/{round(seconds*fps)}',flush=True)
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg video encoder failed')
    print(f'Wrote {output}',flush=True)


def scene_ratio(width):
    return round(width*361/341)/width

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='christmas_tree_ascii_v3.mp4')
    p.add_argument('--preview',default='christmas_tree_v3_preview.png')
    p.add_argument('--seconds',type=float,default=19.5)
    p.add_argument('--gif-output',default='christmas_tree_ascii_v3.gif',
                   help='Destination for the README GIF; pass empty string to skip')
    p.add_argument('--fps',type=int,default=10)
    p.add_argument('--width',type=int,default=520)
    a=p.parse_args()
    run(Path(a.output),a.width,a.fps,a.seconds,Path(a.preview))
    if a.gif_output:
      # Stable global-palette export with a precomputed RGB lookup table.
      # Much faster than running Pillow quantize against 128 colors per frame.
      convert(a.output,a.gif_output,width=min(420,a.width),fps=a.fps)

if __name__=='__main__':main()
