# short_test2.py

import os
import cv2
import numpy as np
from xyz_adjust import *
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2xyz, xyz2rgb
# from utils2 import *

# from numpy import asarray
from numpy import savetxt


# my_file = '/home/jorge/work/A_speos/guiv_4.1/GUI_v3/Y_matrix_correct.png'
# base = os.path.splitext(my_file)[0]
# (my_file, base + '.bin')

# os.rename(my_file, base + '.bin')
def statistic_full(x, lim_min, lim_max):
    all_pix = x.shape[0]*x.shape[1]
    print ("Total number of pixels", all_pix)
    print ("{}< x < {}".format(lim_min,lim_max*0.2),( (x>=lim_min) & (x <=lim_max*0.2) ).sum()," % ",
        (( (x >= lim_min) & (x <= lim_max*0.2) ).sum()/all_pix)*100 )
    
    print ("{}< x < {}".format(lim_max*0.2,lim_max*0.4),( (x>=lim_max*0.2) & (x <=lim_max*0.4) ).sum()," % ",
        (( (x >= lim_max*0.2) & (x <= lim_max*0.4) ).sum()/all_pix)*100 )
    
    print ("{}< x < {}".format(lim_max*0.4,lim_max*0.6),( (x>=lim_max*0.4) & (x <=lim_max*0.6) ).sum()," % ",
        (( (x >= lim_max*0.4) & (x <= lim_max*0.6) ).sum()/all_pix)*100 )

    print ("{}< x < {}".format(lim_max*0.6,lim_max*0.8),( (x>=lim_max*0.6) & (x <=lim_max*0.8) ).sum()," % ",
        (( (x >= lim_max*0.6) & (x <= lim_max*0.8) ).sum()/all_pix)*100 )

    print ("{}< x < {}".format(lim_max*0.8,lim_max),( (x>=lim_max*0.8) & (x <=lim_max) ).sum()," % ",
        (( (x >= lim_max*0.8) & (x <= lim_max) ).sum()/all_pix)*100 )

def cleanxyzV2(X,Y,Z, lim_min, lim_max):
    X = delete_outliers(X, lim_min, lim_max)
    Y = delete_outliers(Y, lim_min, lim_max)
    Z = delete_outliers(Z, lim_min, lim_max)
    return X,Y,Z

def normalization(X,Y,Z, valnorm):
    X = valnorm*(X/np.max(X))#.astype(np.uint8)
    Y = valnorm*(Y/np.max(Y))#.astype(np.uint8)
    Z = valnorm*(Z/np.max(Z))#.astype(np.uint8)
    return X,Y,Z

def show2(img1, img2):
  fig, axes = plt.subplots(1, 2, figsize=(80, 60))
  ax = axes.ravel()
  ax[0].imshow(img1)
  ax[1].imshow(img2)  
  plt.show()

def show4(img1, img2, img3, img4):
  fig, axes = plt.subplots(2, 2, figsize=(80, 60))
  ax = axes.ravel()
  ax[0].imshow(img1)
  ax[1].imshow(img2)  
  ax[2].imshow(img3)
  ax[3].imshow(img4)
  plt.show()

def show6(img1, img2, img3, img4,img5,img6):
  fig, axes = plt.subplots(2, 3, figsize=(80, 60))
  ax = axes.ravel()
  ax[0].imshow(img1)
  ax[1].imshow(img2)  
  ax[2].imshow(img3)
  ax[3].imshow(img4)
  ax[4].imshow(img5)
  ax[5].imshow(img6)
  plt.show()
def show8(img1, img2, img3, img4,img5,img6, img7, img8):
  fig, axes = plt.subplots(2, 4, figsize=(80, 60))
  ax = axes.ravel()
  ax[0].imshow(img1)
  ax[1].imshow(img2)  
  ax[2].imshow(img3)
  ax[3].imshow(img4)
  ax[4].imshow(img5)
  ax[5].imshow(img6)
  ax[6].imshow(img7)
  ax[7].imshow(img8)
  plt.show()
def curve_color2(X, Y, Z, xr, yr, xg, yg, xb, yb, degree):	

	xr= xr.reshape(-1, 1)
	yr= yr.reshape(-1, 1)

	xg= xg.reshape(-1, 1)
	yg= yg.reshape(-1, 1)	

	xb= xb.reshape(-1, 1)
	yb= yb.reshape(-1, 1)

	polyreg1 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	polyreg2 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	polyreg3 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	
	polr = polyreg1.fit(xr,yr)
	polg = polyreg2.fit(xg,yg)
	polb = polyreg3.fit(xb,yb)

	r = X.reshape(-1, 1)
	# r = np.linspace(1,255, 255)
	newr = (polr.predict(r)).astype(np.uint8)
	newr = newr.reshape(X.shape[0], X.shape[1])
	

	# #channel g
	g = Y.reshape(-1, 1)    
	newg = (polg.predict(g)).astype(np.uint8)    
	newg = newg.reshape(X.shape[0], X.shape[1])

	#channel b    
	b = Z.reshape(-1, 1)
	newb = (polb.predict(b)).astype(np.uint8)    
	newb = newb.reshape(X.shape[0], X.shape[1])

	new_img = cv2.merge([newb, newg, newr])

	return newr, newg, newb

####################################### speos data #######################################
folder_path = '/home/jorge/work/speos/GUI_v3/a_speos_data/'
with open(folder_path+'/X.npy', 'rb') as f:
    Xs = np.load(f)

with open(folder_path+'/Y.npy', 'rb') as f:
    Ys = np.load(f)

with open(folder_path+'/Z.npy', 'rb') as f:
    Zs = np.load(f)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> second folder XYZ #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

folder_path = '/home/jorge/work/speos/GUI_v3/a_XYZ_data/'
with open(folder_path+'X.npy', 'rb') as f:
	X = np.load(f)

with open(folder_path+'Y.npy', 'rb') as f:
	Y = np.load(f)

with open(folder_path+'Z.npy', 'rb') as f:
	Z = np.load(f)

print("DONE!")
print("Statistic for MEASURED matrix>>>>>>>>>>>>>>>>>>>>>>>>>")

print("Verification Xs", np.min(Xs), np.max(Xs), Xs.shape)
print("Verification X", np.min(X), np.max(X), X.shape)

lim_minM, lim_maxM = 0, 30
lim_minS, lim_maxS = 0, 70
valnorm = 255

X,Y,Z = cleanxyzV2(X,Y,Z, lim_minM, lim_maxM)

Xs,Ys,Zs = cleanxyzV2(Xs,Ys,Zs, lim_minS, lim_maxS)

print("After eliminate delete_outliers Xs 0 70", Xs.shape, np.max(Xs), np.mean(Xs), np.std(Xs))
print("After eliminate delete_outliers Ys 0 70", Ys.shape, np.max(Ys), np.mean(Ys), np.std(Ys))
print("After eliminate delete_outliers Zs 0 70", Zs.shape, np.max(Zs), np.mean(Zs), np.std(Zs))

X,Y,Z = normalization(X,Y,Z, valnorm)
Xs,Ys,Zs = normalization(Xs,Ys,Zs, valnorm)

maxx1 = np.max([np.max(Xs), np.max(Ys), np.max(Zs)])

img1 = cv2.merge((Xs/maxx1, Ys/maxx1, Zs/maxx1))
img1 = xyz2rgb(img1)

maxx2 = np.max([np.max(X), np.max(Y), np.max(Z)])
img2 = cv2.merge((X/maxx2 , Y/maxx2 , Z/maxx2 ))
img2 = xyz2rgb(img2)


print("img2.shape", img2.shape)
print("img2, type", img2[10,10])

img2 = img2.astype(np.uint8)

gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
savetxt("test.csv", gray, delimiter=',')


img2 = img2.astype(np.uint8)
img2 = cv2.medianBlur(img2,5)

################# apliying XYZ correction #################
degree = 2
xr = np.array([0,125,255])
yr = np.array([0,200,255])

xg = np.array([0,125,255])
yg = np.array([0,200,255])

xb = np.array([0,125,255])
yb = np.array([0,200,255])

print("before Ys shape", Ys.shape, np.max(Ys), np.mean(Ys))


Xs, Ys, Zs = curve_color2(Xs, Ys, Zs, xr, yr, xg, yg, xb, yb, degree)


Xs2 = cv2.medianBlur(Xs,5)


print("After Ys shape", Ys.shape, np.max(Ys), np.mean(Ys))

maxx1 = np.max([np.max(Xs), np.max(Ys), np.max(Zs)])

img3 = cv2.merge((Xs/maxx1, Ys/maxx1, Zs/maxx1))
img3 = xyz2rgb(img3)
img3 = img3.astype(np.uint8)
# img3 = cv2.medianBlur(img3,5)
# maxx2 = np.max([np.max(X), np.max(Y), np.max(Z)])
# img4 = cv2.merge((X/maxx2 , Y/maxx2 , Z/maxx2 ))
# img4 = xyz2rgb(img4)

Xs = ((Xs)/255)*70
Ys = ((Ys)/255)*70
Zs = ((Zs)/255)*70

print("After reescaled Xs 0 70", Xs.shape, np.max(Xs), np.mean(Xs), np.std(Xs))
print("After reescaled Ys 0 70", Ys.shape, np.max(Ys), np.mean(Ys), np.std(Ys))
print("After reescaled Zs 0 70", Zs.shape, np.max(Zs), np.mean(Zs), np.std(Zs))

maxx2 = np.max([np.max(Xs), np.max(Ys), np.max(Zs)])
print("?maxx2", maxx2)

img2_corrected = cv2.merge((Xs/maxx2 , Ys/maxx2 , Zs/maxx2 ))
img2_corrected = xyz2rgb(img2_corrected)

img2_corrected = img2_corrected.astype(np.uint8)
print("img2_corrected.shape", img2_corrected.shape)

# img2_corrected = img2_corrected.medianBlur(img2_corrected,5)
#read image png
# imgrgb = cv2.imread(folder_path+'XYZ_BGR_white_adjust.png')
# imgrgb = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2RGB)

# imgxyz = rgb2xyz(imgrgb)

# recx,recy,recz = cv2.split(imgxyz)
# recx,recy,recz = normalization(recx,recy,recz, valnorm)
# show2(img1, img2)

show4(img2, img1, img3, Xs2)
# show6(img1, img2, imgxyz, recx,recy,recz)

# show8(img1, Xs, Ys, Zs, imgrgb, recx, recy, recz)
