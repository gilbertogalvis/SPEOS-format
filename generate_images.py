# generate_images.py
import cv2 as cv2
import numpy as np

def create_color_bar(color_map, max_val, min_val):
    img1 = np.zeros((255, 25,3))
    img2 = (np.ones((255, 50,3))*255).astype(np.uint8)
    img3 = 255*np.ones((15, 75,3))
    


    for n in range(0,255):    
        img1[n, :]= [255-n, 255-n, 255-n]   

    img1 = img1.astype(np.uint8)

    imgb = cv2.applyColorMap(img1, color_map)
    font = cv2.FONT_HERSHEY_SIMPLEX

    vis = np.concatenate((imgb, img2), axis=1)
    vis = np.concatenate((img3, vis), axis=0)
    vis = np.concatenate((vis, img3), axis=0)

    for l in range(0,17):
        idx = "{:.1f}".format(max_val*(16-l)*16/256)
        cv2.putText(vis, '{}'.format(idx), (30,15 + l*16), font, 0.4, (0, 0, 0), 1, cv2.LINE_4)
    return vis

def norm(x, lim_min, lim_max):
    
    idx = x <= lim_min
    x[idx] = lim_min

    idx = x >= lim_max
    x[idx] = lim_max

    return x

def statistic2(x, lim_min, lim_max ):
    all_pix = x.shape[0]*x.shape[1]
    print ("Total number of pixels", all_pix)    
    print ("{} x > {}".format(lim_min,lim_max),( (x>=lim_min) & (x <=lim_max) ).sum()," % ",
        (( (x >= lim_min) & (x <= lim_max) ).sum()/all_pix)*100 )

def get_image_set(path,brightness, map_id, max_val, min_val):
    print (path)
    img1 = cv2.imread(path)    
    # print ("Shape of read image ##",img1.shape)
    
    img2 = cv2.resize(img1, (400,240), interpolation = cv2.INTER_AREA)
    
    ref2 = cv2.convertScaleAbs(img2, alpha=1, beta = brightness)
    cv2.imwrite("images/A1.png", ref2)

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    if (max_val ==0) and (min_val==0):
        max_val = np.max(gray)
        min_val = np.min(gray)

    statistic2(gray, min_val ,max_val)

    gray = norm(gray, min_val, max_val)
    gray = cv2.convertScaleAbs(gray, alpha=1, beta = brightness)

    img_maped = cv2.applyColorMap(gray, map_id)

    cv2.imwrite("images/A2.png", img_maped)

    color_bar = create_color_bar(map_id, max_val, min_val)
    cv2.imwrite("images/A1_bar.png", color_bar)
    print ("Set of images uploads")

def get_image_set2(path,brightness, map_id, max_val, min_val):
    
    name_file = path.split('/')[-1]
    
    img1 = cv2.imread(path)    
    
    ref2 = cv2.convertScaleAbs(img1, alpha=1, beta = brightness)
    cv2.imwrite("images/A1.png", ref2)

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)    

    if (max_val ==0) and (min_val==0):
        max_val = np.max(gray)
        min_val = np.min(gray)    

    gray = norm(gray, min_val, max_val)
    gray = cv2.convertScaleAbs(gray, alpha=1, beta = brightness)

    img_maped = cv2.applyColorMap(gray, map_id)

    cv2.imwrite("images/"+name_file, img_maped)

    return "images/"+name_file


def save_map(Y, path, map_id):
    
    cv2.imwrite(path, Y)
    
    # name_file = path.split('/')[-1]
    
    img1 = cv2.imread(path)    
    
    # ref2 = cv2.convertScaleAbs(img1, alpha=1, beta = brightness)
    # cv2.imwrite("images/A1.png", ref2)

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)    

    img_maped = cv2.applyColorMap(gray, map_id)
    
    cv2.imwrite("gray.png", gray)
    cv2.imwrite(path, img_maped)