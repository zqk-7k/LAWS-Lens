"""Nonblank checks and an inspection sheet for independently rebuilt plots."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps

p=argparse.ArgumentParser()
p.add_argument('--figures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
files=sorted(a.figures.glob('*.png'))
sheet=Image.new('RGB',(1500,4*355),'white')
draw=ImageDraw.Draw(sheet)
checks=[]
for i,path in enumerate(files):
    with Image.open(path) as raw:
        im=raw.convert('RGB')
        pixels=np.asarray(im)
        nonwhite=float((pixels.min(axis=2)<240).mean())
        passed=im.width>=600 and im.height>=250 and .015<nonwhite<.9
        checks.append({'file':path.name,'width':im.width,'height':im.height,'nonwhite_fraction':nonwhite,'passed':passed})
        small=ImageOps.contain(im,(490,320))
        x,y=(i%3)*500,(i//3)*355
        sheet.paste(small,(x+(500-small.width)//2,y))
        draw.text((x+5,y+327),path.stem,fill='black')
sheet.save(a.output/'CONTACT_SHEET.png')
status='PASS' if len(files)==11 and all(x['passed'] for x in checks) else 'FAIL'
(a.output/'REPORT.json').write_text(json.dumps({'status':status,'figures':checks,'pixel_identity_with_paper_not_claimed':True},indent=2))
print(status)
raise SystemExit(0 if status=='PASS' else 1)
