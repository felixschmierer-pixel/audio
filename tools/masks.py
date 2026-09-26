import cv2, numpy as np
def cloth_mask(path, out, extra_poly=None, hue=(95,135), smax=None, vmax=140, close=25):
    im = cv2.imread(path); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    m = (h>=hue[0])&(h<=hue[1])&(v<=vmax)
    if smax: m &= s<=smax
    m = m.astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close,close))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    # keep only large components
    n,lab,st,_ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA] > 0.01*m.size: keep[lab==i]=255
    if extra_poly is not None:
        cv2.fillPoly(keep, [np.array(extra_poly,np.int32)], 255)
    keep = cv2.erode(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    cv2.imwrite(out, keep)
    ov = im.copy(); ov[keep>0] = (0.5*ov[keep>0] + (0,0,127)).astype(np.uint8)
    cv2.imwrite(out.replace('.png','_ov.jpg'), ov, [cv2.IMWRITE_JPEG_QUALITY,80])
    print(out, keep.mean()/255)
# A: shirt; force-include mic area (mic is black, hue undefined)
cloth_mask('161_cropA.png','maskA.png', extra_poly=[(300,270),(470,270),(470,430),(300,430)])
# B: shirt + trousers + belt (belt is brown: add second pass)
cloth_mask('161_cropB.png','maskB.png', vmax=150)
im=cv2.imread('161_cropB.png'); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); h,s,v=cv2.split(hsv)
belt=((h<=25)&(s>60)&(v<200)).astype(np.uint8)*255
belt=cv2.morphologyEx(belt,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
# restrict belt to belt band rows (approx y 700..1050 in crop) and exclude right edge (jeans/background)
bb=np.zeros_like(belt); bb[680:1080, 120:1550]=255; belt&=bb
mB=cv2.imread('maskB.png',0); mB=np.maximum(mB,belt)
# exclude right-side neighbour's jeans and the arm region top
mB[:, 1650:]=0
cv2.imwrite('maskB.png',mB)
ov=im.copy(); ov[mB>0]=(0.5*ov[mB>0]+(0,0,127)).astype(np.uint8); cv2.imwrite('maskB_ov.jpg',ov,[cv2.IMWRITE_JPEG_QUALITY,80])
