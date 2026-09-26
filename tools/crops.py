from PIL import Image
im=Image.open('/home/user/audio/BAU-161.jpg')
CROPS={'A':(850,1800,1874,2824),   # left man: mic + shirt, 1:1
       'B':(2100,2800,3900,4000)}  # middle man: belly/belt/trousers, 3:2
for k,b in CROPS.items():
    im.crop(b).save(f'161_crop{k}.png')
    print(k, b, im.crop(b).size)
