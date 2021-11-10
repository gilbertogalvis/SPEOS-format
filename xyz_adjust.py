# xyz_adjust.py
import os
import numpy as np
import cv2 as cv2
from skimage import data
from skimage.color import rgb2xyz, xyz2rgb
from utils2 import *

def get_XYZ(path):
	
	path_files = '/'.join(path.split('/')[0:-1])	
			
	with open(path_files+'/X.npy', 'rb') as f:
		X = np.load(f)

	with open(path_files+'/Y.npy', 'rb') as f:
		Y = np.load(f)

	with open(path_files+'/Z.npy', 'rb') as f:
		Z = np.load(f)

	with open(path_files+'/cx.npy', 'rb') as f:
		cx = np.load(f)

	with open(path_files+'/cy.npy', 'rb') as f:
		cy = np.load(f)	

	return X, Y, Z, cx, cy

def statistic(X, Y, Z):
	print ("Statistics feactures")
	print (np.max(X))
	print (np.min(X))

	print (np.max(Y))
	print (np.min(Y))

	print (np.max(Z))
	print (np.min(Z))

def delete_outliers(x, lim_min, lim_max):    
    idx = x <= lim_min
    x[idx] = lim_min

    idx = x >= lim_max
    x[idx] = lim_max
    return x   #/np.max(x[:])

def statistic2(x, lim_min, lim_max ):
    all_pix = x.shape[0]*x.shape[1]
    print ("Total number of pixels", all_pix)
    print ("{} x > {}".format(lim_min,lim_max),( (x>=lim_min) & (x <=lim_max) ).sum()," % ",
        (( (x >= lim_min) & (x <= lim_max) ).sum()/all_pix)*100 )

def speos_RGB(img1):
	# img1 = cv2.merge((X, Y, Z))
	img2 = xyz2rgb(img1)
	R, G , B = cv2.split(img2)
	img = cv2.merge((B,G,R))*255
	return img

def cleanxyz(X, Y, Z, Xs, Ys, Zs):
	
	Xsn = delete_outliers(Xs, 0, 70)
	Ysn = delete_outliers(Ys, 0, 70)
	Zsn = delete_outliers(Zs, 0, 70)

	X = delete_outliers(X, 0, 10)
	Y = delete_outliers(Y, 0, 10)
	Z = delete_outliers(Z, 0, 10)
	return X, Y, Z,Xsn, Ysn, Zsn

def crop_savematrix(Xsnew, Ysnew, Zsnew, cx, cy, p1y, p2y, p1x,p2x, folder):
	Xsnew = Xsnew[p1y:p2y , p1x:p2x]
	Ysnew = Ysnew[p1y:p2y , p1x:p2x]
	Zsnew = Zsnew[p1y:p2y , p1x:p2x]

	with open(folder + '/results/X.npy', 'wb') as f:
	  np.save(f, Xsnew)

	with open(folder + '/results/Y.npy', 'wb') as f:
	  np.save(f, Ysnew)

	with open(folder + '/results/Z.npy', 'wb') as f:
	    np.save(f, Zsnew)

	with open(folder + '/results/cx.npy', 'wb') as f:
	    np.save(f, cx)

	with open(folder + '/results/cy.npy', 'wb') as f:
	    np.save(f, cy)    

	return Xsnew, Ysnew, Zsnew

def cleanxyzV2(X,Y,Z, lim_min, lim_max):
    X = delete_outliers(X, lim_min, lim_max)
    Y = delete_outliers(Y, lim_min, lim_max)
    Z = delete_outliers(Z, lim_min, lim_max)
    return X, Y, Z

def normalization(X,Y,Z, valnorm):
    X = valnorm*(X/np.max(X))#.astype(np.uint8)
    Y = valnorm*(Y/np.max(Y))#.astype(np.uint8)
    Z = valnorm*(Z/np.max(Z))#.astype(np.uint8)
    return X,Y,Z
# path_xyz = '/home/jorge/work/A_speos/GUI/v6/a_XYZ_data/XYZ_BGR_white_adjust.png'
# path_speos = '/home/jorge/work/A_speos/GUI/v6/a_speos_data/speos_color_correction.png'

# X, Y, Z = get_XYZ(path_xyz)
# Xs, Ys, Zs = get_XYZ(path_speos)
# Xsn, Ysn, Zsn = cleanxyz(X, Y, Z, Xs, Ys, Zs)

# img1 = cv2.merge((Xsn, Ysn, Zsn))
# img = speos_RGB(img1)

# x_list = [2142, 2913]
# y_list = [3100, 3143]

# x_list2 = [1350, 1460]
# y_list2 = [1040, 1035]

# xr = np.array([0,20,70, 100, 120, 140, 160, 255])#np.zeros(len(x_list))
# yr = np.array([0,40,110,140, 170, 190, 210, 255])#np.zeros(len(x_list))

# xg = np.array([0,20,70, 100, 120, 140, 160, 255])#np.zeros(len(x_list))
# yg = np.array([0,30,90, 120, 150, 160, 180, 255])#np.zeros(len(x_list))

# xb = np.array([0,20,70, 100, 120, 140, 160, 255])#np.zeros(len(x_list))
# yb = np.array([0,30,90, 120, 150, 160, 180, 255])#np.zeros(len(x_list))

# xr = np.array(np.zeros(len(x_list)))
# yr = np.array(np.zeros(len(x_list)))
# xg = np.array(np.zeros(len(x_list)))
# yg = np.array(np.zeros(len(x_list)))
# xb = np.array(np.zeros(len(x_list)))
# yb = np.array(np.zeros(len(x_list)))

# for i in range(0, len(x_list)):

# 	xr[i] = X[x_list[i], y_list[i]]
# 	xg[i] = Y[x_list[i], y_list[i]]
# 	xb[i] = Z[x_list[i], y_list[i]]	

# 	yr[i] = Xsn[x_list2[i], y_list2[i]]
# 	yg[i] = Ysn[x_list2[i], y_list2[i]]
# 	yb[i] = Zsn[x_list2[i], y_list2[i]]

# print ("xr", xr)
# print ("Yr", yr)
# print ("Xg", xg)
# print ("Yg", yg)

# print ("Xb", xb)
# print ("Yb", yb)

# fl1 = True
# print ("Max Xsn", np.max(Xsn) )
# print ("Max Ysn",np.max(Ysn) )
# print ("Max Zsn",np.max(Zsn) )

# degree = 2
# Xsnew, Ysnew, Zsnew = curve_color2(Xsn, Ysn, Zsn, xr, yr, xg, yg, xb, yb, degree)

# maxx = np.max([np.max(X), np.max(Y), np.max(Z)])
# img1 = cv2.merge((Xsnew/maxx, Ysnew/maxx, Zsnew/maxx))

# img2 = xyz2rgb(img1)
# R, G , B = cv2.split(img2)
# img = cv2.merge((B,G,R))*255
# cv2.imwrite("XYZ.png", img)