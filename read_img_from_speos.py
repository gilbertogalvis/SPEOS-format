import numpy as np
from skimage import data
from skimage.color import rgb2xyz, xyz2rgb
import cv2 as cv2
import matplotlib.pyplot as plt
# from utils2 import *
import os

def norm(x):
	idx = x >= 150
	x[idx] = np.mean(x[idx==False])
	return x/np.max(x[:])
	
def read_txtSpeos(path1, folderout, filename):

	with open(path1) as f:
	    lines = f.readlines()

	line_size = 19584
	height = 3264

	line_size = 11200
	height = 2000

	
	i2 = 0
	X = np.zeros([height,int(line_size/4)])
	Y = np.zeros([height,int(line_size/4)])
	Z = np.zeros([height,int(line_size/4)])

	cx = np.zeros([height, int(line_size/4)])
	cy = np.zeros([height, int(line_size/4)])


	# for i in range(8,height+9):
	for i in range(0,height):

		lact = lines[i].split()

		lsize = len(lines[i].split())
		if lsize == 11200:
		# if lsize == line_size:
			
			for n in range(0, int(line_size/4)):
				n2 = n*4
				Y[i2, n] = lact[n2+1]
				X[i2, n] = lact[n2]
				Z[i2, n] = lact[n2+3]

				if X[i2, n]+Y[i2, n]+Z[i2, n]!=0:

					cx[i2, n] = float(X[i2, n])/float(X[i2, n]+Y[i2, n]+Z[i2, n])
					cy[i2, n] = float(Y[i2, n])/float(X[i2, n]+Y[i2, n]+Z[i2, n])

			i2 = i2 +1


	with open(os.path.join(folderout,'X.npy'), 'wb') as f:
		np.save(f, X)

	with open(os.path.join(folderout,'Y.npy'), 'wb') as f:
		np.save(f, Y)

	with open(os.path.join(folderout,'Z.npy'), 'wb') as f:
		np.save(f, Z)

	with open(os.path.join(folderout,'cx.npy'), 'wb') as f:
		np.save(f, cx)

	with open(os.path.join(folderout,'cy.npy'), 'wb') as f:
		np.save(f, cy)

	Y2 = Y
	Xn = norm(X)

	Yn = norm(Y)

	Zn = norm(Z)

	img1 = cv2.merge((Xn, Yn, Zn))

	img2 = xyz2rgb(img1)

	R, G , B = cv2.split(img2)	

	img3 = cv2.merge((B,G,R))*255	

	cv2.imwrite(os.path.join(folderout, filename), img3)

	print ("Image from  Speos generated!")

	return os.path.join(folderout, filename)

# path1 = '/home/jorge/work/speos/GUI_v5.2/txt_files/speos_txt/sim.txt'
# folderout = '/home/jorge/work/speos/GUI_v5.2/txt_files/out'
# filename = 'speos_out.png'

# read_txtSpeos(path1, folderout, filename)
