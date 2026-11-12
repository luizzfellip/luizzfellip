"""Fast, reproducible global-palette GIF encoder for ASCII tree MP4.

It avoids extremely slow per-frame Pillow quantize(palette=...) while keeping a
stable color table and exactly one frame per source video frame.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess
import shutil


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


if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--video',default='christmas_tree_ascii_v3.mp4');a.add_argument('--gif',default='christmas_tree_ascii_v3.gif');a.add_argument('--width',type=int,default=420);a.add_argument('--fps',type=int,default=10);s=a.parse_args()
 convert(s.video,s.gif,s.width,s.fps)
