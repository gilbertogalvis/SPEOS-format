# import pandas as pd
import numpy as np
from skimage import data
from skimage.color import xyz2rgb,rgb2xyz
import cv2 as cv2
from colormath.color_objects import XYZColor, sRGBColor
from colormath.color_conversions import convert_color
import os

# path_files = os.getcwd()
# os.path.join(path_files, ) 

def txt2XYZ(path1, path2, path3, folderout, filename, height,width):

	with open(path1) as f:
	    linesX = f.readlines()

	with open(path2) as f:
	    linesY = f.readlines()

	with open(path3) as f:
	    linesZ = f.readlines()

	Ylist = []
	Xi = []
	Yi = []

	Cxlist = []
	Cylist = []

	fl = []	

	xi = 0
	# height,width = 3264, 4896

	line_size = width*4
	data = np.zeros([height,width,3])
	new_line = ''

	linesX = linesX[8:len(linesX)]
	linesY = linesY[8:len(linesY)]
	linesZ = linesZ[8:len(linesZ)]

	xn,yn,zn = 0,0,0
	Xa = np.zeros([height, width])
	Ya = np.zeros([height, width])
	Za = np.zeros([height, width])

	cx = np.zeros([height, width])
	cy = np.zeros([height, width])
	imgr = np.zeros([height, width, 3])

	print("Generating X Y Z arrays")

	for i in range(0,height):

			new_line = ''

			lineX = linesX[i]
			lineY = linesY[i]
			lineZ = linesZ[i]
			
			rx = lineX.split()
			ry = lineY.split()
			rz = lineZ.split()

			for yi in range(0,width-1):			
				X = float(rx[yi])
				Y = float(ry[yi])
				Z = float(rz[yi])			

				Xa[xi, yi] = X
				Ya[xi, yi] = Y
				Za[xi, yi] = Z

				cx[xi, yi] = X/(X+Y+Z)
				cy[xi, yi] = Y/(X+Y+Z)
			xi = xi +1	

	print("Savings Arrays")

	with open(os.path.join(folderout,'X.npy'), 'wb') as f:
		np.save(f, Xa)

	with open(os.path.join(folderout,'Y.npy'), 'wb') as f:
		np.save(f, Ya)

	with open(os.path.join(folderout,'Z.npy'), 'wb') as f:
		np.save(f, Za)

	with open(os.path.join(folderout,'cx.npy'), 'wb') as f:
		np.save(f, cx)

	with open(os.path.join(folderout,'cy.npy'), 'wb') as f:
		np.save(f, cy)

	
	Ya2 = Ya
	img1 = cv2.merge((Xa/np.max(Xa), Ya/np.max(Ya), Za/np.max(Za)))

	imgrgb1 = xyz2rgb(img1)

	################## noramlized image using function ###############

	out = np.zeros(Xa.shape, np.double)
	Xa = cv2.normalize(Xa, out, 1.0, 0.0, cv2.NORM_MINMAX)

	out = np.zeros(Ya.shape, np.double)
	Ya = cv2.normalize(Ya, out, 1.0, 0.0, cv2.NORM_MINMAX)

	out = np.zeros(Za.shape, np.double)
	Za = cv2.normalize(Za, out, 1.0, 0.0, cv2.NORM_MINMAX)

	img2 = cv2.merge((Ya, Xa, Za))## un poco de azul similar a la imagen simulada

	imgrgb2 = xyz2rgb(img2)

	R, G, B = cv2.split(imgrgb2)
	imgrgb3 = cv2.merge((B, G, R))

	imgrgb3 = imgrgb3*255
	print(os.path.join(folderout, filename))
	cv2.imwrite(os.path.join(folderout, filename), imgrgb3)
	return os.path.join(folderout, filename)

# path1 = '/home/jorge/work/speos/GUI_v5.2/txt_files/xyz_files/Akamai3_X.txt'

# path2 = '/home/jorge/work/speos/GUI_v5.2/txt_files/xyz_files/Akamai3_Y.txt'

# path3 = '/home/jorge/work/speos/GUI_v5.2/txt_files/xyz_files/Akamai3_Z.txt'
# height,width = 3264, 4896

# folderout = '/home/jorge/work/speos/GUI_v5.2/txt_files/out'
# filename = 'RGB.png'

# txt2XYZ(path1, path2, path3, folderout, filename, height,width)

