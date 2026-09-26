"""Batch pipeline: cut crops, build clothing masks, composite edits back.
usage: pipeline.py cut | mask | composite | grid
"""
import sys, os, json, subprocess, cv2, numpy as np
ORIG = '/home/user/audio'
OUT  = '/home/user/audio/retusche'
# name: (image, box x0,y0,x1,y1, aspect, description)
CROPS = {
 '161C': ('BAU-161', (400,2700,2200,3900), '3:2', 'lower torso and grey work trousers of a man in a dark grey-blue t-shirt with crossed arms, scaffolding behind'),
 '017A': ('BAU-017', (2000,1750,3400,3500), '4:5', 'two men in dark grey-blue company t-shirts and grey work trousers standing in front of a white van with a big blue logo, one hand on hip'),
 '017B': ('BAU-017', (3250,1750,5350,3430), '5:4', 'three men in dark grey-blue company t-shirts and grey work trousers in front of a white van, arms crossed or on hips'),
 '040A': ('BAU-040', (1900,1600,4033,4000), '8:9', 'two men in dark grey-blue company t-shirts and grey work trousers with tool pouches, arms crossed, white van with blue lettering behind'),
 '066A': ('BAU-066', (2900,1750,4700,4000), '4:5', 'a man in a dark grey-blue t-shirt holding a blue clipboard, brown leather belt, grey trousers, white van with blue and red logo behind'),
 '181A': ('BAU-181', (1200,1450,2500,3400), '2:3', 'a crouching man in a dark grey-blue t-shirt and grey work trousers on a wooden roof deck'),
 '181B': ('BAU-181', (3250,1400,4650,3500), '2:3', 'a man in a dark grey-blue t-shirt with company logo and grey work trousers with a leather belt, standing next to a dark brick chimney'),
 '233A': ('BAU-233', (1700,2500,3300,4900), '2:3', 'a man in a black polo shirt with hands on hips and light grey jeans standing on a scaffold deck, cement sacks and buckets behind'),
 '247A': ('BAU-247', (700,1750,1937,3400), '3:4', 'an older man in a dark grey-blue t-shirt with company logo, holding a trowel, dark bricks in front'),
 '247B': ('BAU-247', (2800,1700,4525,4000), '3:4', 'a man in a dark grey-blue t-shirt with company logo and grey work trousers, one arm extended holding a trowel, scaffolding behind'),
 '250A': ('BAU-250', (870,1900,2070,3400), '4:5', 'an older man in a dark grey-blue t-shirt with company logo holding a trowel over dark bricks'),
 '250B': ('BAU-250', (2950,1650,4713,4000), '3:4', 'a man in a dark grey-blue t-shirt with company logo and grey work trousers, arm extended holding a trowel, scaffolding behind'),
 '461A': ('BAU-461', (1100,1450,2337,3650), '9:16', 'a man in a dark grey-blue polo shirt with company logo and grey work trousers standing next to a yellow wheel loader'),
 '461B': ('BAU-461', (2100,1100,3000,2450), '2:3', 'a man in a dark grey-blue t-shirt and dark work trousers sitting in the cab of a yellow wheel loader'),
 '461C': ('BAU-461', (2600,1200,3866,3450), '9:16', 'a young man in a dark grey-blue t-shirt with company logo and grey work trousers standing in front of a yellow wheel loader, dusty work boots'),
 '461D': ('BAU-461', (3680,1250,5086,3750), '9:16', 'two men in dark grey-blue company t-shirts and grey work trousers, one with a foot on a concrete block, giving thumbs up'),
}
PROMPT = ("Photo retouching task on a crop of a real photograph: {desc}. "
 "Make the clothing look clean and tidy: remove all dust, cement, paint and dirt stains from the t-shirts and polo shirts so they look freshly washed; "
 "make the work trousers about 70 percent cleaner, keeping a little natural wear; tuck any untucked shirt hem neatly and completely into the trousers so the waistband and belt are visible; "
 "remove any clip-on microphones or similar stray objects on the shirts. "
 "Keep everything else pixel-identical: same framing and pose, same skin, hands and arm hair, same watches and belts, same company logos with the same text and position, same background, same colours, same natural photographic grain and sharpness. "
 "Do not beautify, do not change the composition, do not add anything.")

def cut():
    from PIL import Image
    cache = {}
    for k,(img,b,ar,_) in CROPS.items():
        im = cache.setdefault(img, Image.open(f'{ORIG}/{img}.jpg'))
        c = im.crop(b); c.save(f'crop_{k}.png'); print(k, c.size, ar)

def cloth_mask(crop, vmax_blue=215, vmax_grey=140, smax_grey=70):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); h,s,v = cv2.split(hsv)
    # logos / strongly saturated prints stay original
    logo = ((s>=110)&(v>=90)).astype(np.uint8)*255
    logo = cv2.dilate(logo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    skin = ((h<=25)&(s>=45)&(v>=80)).astype(np.uint8)*255
    skin = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    cloth = (((h>=95)&(h<=135)&(v<=vmax_blue)) | ((s<=smax_grey)&(v<=vmax_grey))).astype(np.uint8)*255
    cloth[skin>0]=0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61))
    cloth = cv2.morphologyEx(cloth, cv2.MORPH_CLOSE, k)
    # fill enclosed holes (bright dust spots inside clothing)
    inv = cv2.bitwise_not(cloth); ff = inv.copy(); hh,ww = inv.shape
    m2 = np.zeros((hh+2,ww+2),np.uint8); cv2.floodFill(ff, m2, (0,0), 0)
    for pt in [(ww-1,0),(0,hh-1),(ww-1,hh-1)]:
        m2[:]=0; cv2.floodFill(ff, m2, pt, 0)
    cloth = cv2.bitwise_or(cloth, ff)
    cloth = cv2.morphologyEx(cloth, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    n,lab,st,_ = cv2.connectedComponentsWithStats(cloth); keep=np.zeros_like(cloth)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA] > 0.01*cloth.size: keep[lab==i]=255
    keep[skin>0]=0; keep[logo>0]=0
    return cv2.erode(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))

_SC={'017A':1540,'017B':1480,'181A':1560,'233A':2200,'461A':1980,'461B':1215,'461C':2025,'461D':2250}
def mask():
    for k in CROPS:
        c = cv2.imread(f'crop_{k}.png')
        m = cloth_mask(c, 230, 205, 80) if k in ('247A','247B','250A','250B') else cloth_mask(c)
        if k in _SC: m[_SC[k]:,:]=0
        cv2.imwrite(f'mask_{k}.png', m)
        ov = c.copy(); ov[m>0] = (0.5*ov[m>0] + (0,0,127)).astype(np.uint8)
        small = cv2.resize(ov, None, fx=600/ov.shape[0], fy=600/ov.shape[0]); cv2.imwrite(f'maskov_{k}.jpg', small, [cv2.IMWRITE_JPEG_QUALITY,75])
        print(k, round(m.mean()/255,2))

def composite():
    by_img = {}
    for k,(img,b,ar,_) in CROPS.items(): by_img.setdefault(img,[]).append((k,b))
    os.makedirs(OUT, exist_ok=True)
    for img, items in by_img.items():
        src = f'{ORIG}/{img}.jpg'
        if img=='BAU-161': src = 'BAU-161_retusche.jpg'   # already carries A and B
        cur = src
        for i,(k,b) in enumerate(items):
            if not os.path.exists(f'ed_{k}.png'): print('missing', k); continue
            nxt = f'tmp_{img}_{i}.png'
            r = subprocess.run(['python3','composite.py',cur,nxt,'box=%d,%d,%d,%d'%b,f'edited=ed_{k}.png',f'mask=mask_{k}.png','feather=30'],capture_output=True,text=True)
            print(k, r.stdout.strip().splitlines()[0], r.stdout.strip().splitlines()[-1]); cur = nxt
        im = cv2.imread(cur); cv2.imwrite(f'{OUT}/{img}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY,97])
        # before/after per crop
        o = cv2.imread(f'{ORIG}/{img}.jpg')
        for k,b in items:
            x0,y0,x1,y1=b; a=o[y0:y1,x0:x1]; c=im[y0:y1,x0:x1]
            ba=np.hstack([a,np.full((a.shape[0],20,3),255,np.uint8),c]); sc=1400/ba.shape[1]
            cv2.imwrite(f'{OUT}/{img}_vorher_nachher_{k[-1]}.jpg', cv2.resize(ba,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY,88])

if __name__=='__main__':
    globals()[sys.argv[1]]()
