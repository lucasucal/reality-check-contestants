import sys
import numpy as np
from PIL import Image
from pathlib import Path
hdr, metal, out = map(Path, sys.argv[1:])
with hdr.open('rb') as f:
    assert f.readline().startswith(b'#?RADIANCE')
    while f.readline().strip(): pass
    dims = f.readline().split(); h, w = int(dims[1]), int(dims[3])
    data = np.empty((h,w,4),dtype=np.uint8)
    for row in range(h):
        a = f.read(4); assert a[:2] == b'\x02\x02' and int.from_bytes(a[2:], 'big') == w
        for channel in range(4):
            i = 0
            while i < w:
                n = f.read(1)[0]
                if n > 128:
                    count=n-128; data[row,i:i+count,channel]=f.read(1)[0]
                else:
                    count=n; data[row,i:i+count,channel]=np.frombuffer(f.read(count),dtype=np.uint8)
                i += count
rgb = data[:,:,:3].astype(np.float32) * np.exp2(data[:,:,3].astype(np.int16)-136)[:,:,None]
# Fixed photographic exposure and filmic compression; encode sRGB for MuJoCo.
rgb = rgb * 0.85
rgb = rgb / (1 + rgb)
rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*np.power(rgb,1/2.4)-.055)
rgb = np.uint8(np.clip(rgb*255,0,255))
size=1024
u,v=np.meshgrid((np.arange(size)+.5)*2/size-1,(np.arange(size)+.5)*2/size-1)
one=np.ones_like(u)
# OpenGL cubemap faces: direction at each output pixel, v increases downwards.
faces=[(one,-v,-u),(-one,-v,u),(u,one,v),(u,-one,-v),(u,-v,one),(-u,-v,-one)]
# Coordinates above are OpenGL Y-up; equirectangular panorama is Z-up after conversion.
grid=Image.new('RGB',(size*3,size*2))
for i,(x,y,z) in enumerate(faces):
    lon=np.arctan2(z,x); lat=np.arctan2(y,np.sqrt(x*x+z*z))
    px=np.mod(((lon/(2*np.pi)+.5)*w).astype(int),w)
    py=np.clip(((.5-lat/np.pi)*h).astype(int),0,h-1)
    grid.paste(Image.fromarray(rgb[py,px]),((i%3)*size,(i//3)*size))
out.mkdir(parents=True,exist_ok=True)
grid.save(out/'workshop_sky.png',optimize=True)
Image.open(metal).convert('RGB').save(out/'workbench_metal.png',optimize=True)
