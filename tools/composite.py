"""Fit an AI-edited crop back into the original pixel-exactly.
usage: composite.py ORIG OUT box=x0,y0,x1,y1 edited=path [feather=px] [mask=path]
Steps: resize edited to crop size -> ECC affine alignment on luminance -> per-channel
mean/std colour match inside the mask -> re-add original film grain -> feathered mask blend.
"""
import sys, cv2, numpy as np
from PIL import Image

def ecc_align(ref, mov):
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    g2 = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    warp = np.eye(2,3,dtype=np.float32)
    try:
        crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        cc, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error as e:
        print('ECC failed, identity used', e); cc=0
    out = cv2.warpAffine(mov, warp, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LANCZOS4+cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return out, cc, warp

def color_match(src, ref, mask):
    src = src.astype(np.float32); ref = ref.astype(np.float32)
    m = mask > 0.5
    out = src.copy()
    for c in range(3):
        s, r = src[...,c][m], ref[...,c][m]
        out[...,c] = (src[...,c]-s.mean())/(s.std()+1e-6)*r.std()+r.mean()
    return np.clip(out,0,255)

def grain_of(ref, ksize=3):
    blur = cv2.GaussianBlur(ref.astype(np.float32),(0,0),1.0)
    return ref.astype(np.float32)-blur

def main():
    orig_p, out_p = sys.argv[1], sys.argv[2]
    kw = dict(a.split('=',1) for a in sys.argv[3:])
    x0,y0,x1,y1 = map(int, kw['box'].split(','))
    feather = int(kw.get('feather', 40))
    orig = cv2.imread(orig_p, cv2.IMREAD_COLOR)
    crop = orig[y0:y1, x0:x1]
    ed = cv2.imread(kw['edited'], cv2.IMREAD_COLOR)
    ed = cv2.resize(ed, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA if ed.shape[0]>crop.shape[0] else cv2.INTER_LANCZOS4)
    al, cc, warp = ecc_align(crop, ed)
    print(f'ECC corr={cc:.4f} warp=\n{warp}')
    h,w = crop.shape[:2]
    if 'mask' in kw:
        mask = cv2.imread(kw['mask'], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        mask = cv2.resize(mask,(w,h))
    else:
        mask = np.ones((h,w),np.float32)
    # keep a border of the crop untouched and feather
    mask[:feather,:]=0; mask[-feather:,:]=0; mask[:,:feather]=0; mask[:,-feather:]=0
    mask = cv2.GaussianBlur(mask,(0,0),feather/2)
    al = color_match(al, crop, mask)
    # grain: replace AI high-frequency with original's high-frequency in mask, scaled
    hf_orig = grain_of(crop); hf_ai = grain_of(al)
    s_o = hf_orig.std(); s_a = hf_ai.std()
    print(f'grain std orig={s_o:.2f} ai={s_a:.2f}')
    if s_a < s_o:   # add synthetic grain with the original's statistics
        rng = np.random.default_rng(161)
        noise = rng.normal(0, 1, crop.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise,(0,0),0.6)
        noise *= (np.sqrt(max(s_o**2 - s_a**2,0)) / (noise.std()+1e-6))
        al = np.clip(al + noise, 0, 255)
    m3 = mask[...,None]
    blended = crop.astype(np.float32)*(1-m3) + al*m3
    res = orig.copy(); res[y0:y1,x0:x1] = np.clip(blended,0,255).astype(np.uint8)
    cv2.imwrite(out_p, res, [cv2.IMWRITE_JPEG_QUALITY, 97] if out_p.endswith('.jpg') else [])
    diff = np.abs(res.astype(int)-orig.astype(int)).sum(-1)>0
    print('changed pixels:', diff.sum(), 'of', diff.size, f'({100*diff.sum()/diff.size:.2f}%)')
main()
