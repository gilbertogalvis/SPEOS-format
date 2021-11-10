# adjust_colors.py

import cv2 as cv2
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image, ImageQt


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from utils import white_balance

def sort_array(xr, yr):
	xr = np.array(xr)
	yr = np.array(yr)
	xrcopy = xr.copy()
	yrcopy = yr.copy()
	idr = np.argsort(xr)
	for n, nidx in enumerate(idr):	
		xr[n] = xrcopy[nidx]
		yr[n] = yrcopy[nidx]	
	
	xr = xr.reshape(xr.shape[0],1).reshape(-1, 1)
	yr = yr.reshape(yr.shape[0],1).reshape(-1, 1)	
	return xr, yr

def curve_color(imgpath, xr, yr, xg, yg, xb, yb, degree):
	polyreg1 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	polyreg2 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	polyreg3 = make_pipeline(PolynomialFeatures(degree),LinearRegression())

	polr = polyreg1.fit(xr,yr)
	polg = polyreg2.fit(xg,yg)
	polb = polyreg3.fit(xb,yb)	

	print ("path for read image inside function color curve #####", imgpath)
	img = cv2.imread(imgpath)

	r, g, b = cv2.split(img)

	# #channel R
	r = r.reshape(-1, 1)
	newr = (polr.predict(r)).astype(np.uint8)
	newr = newr.reshape(img.shape[0], img.shape[1])
	
	# #channel g
	g = g.reshape(-1, 1)    
	newg = (polg.predict(g)).astype(np.uint8)    
	newg = newg.reshape(img.shape[0], img.shape[1])

	#channel b    
	b = b.reshape(-1, 1)    
	newb = (polb.predict(b)).astype(np.uint8)    
	newb = newb.reshape(img.shape[0], img.shape[1])

	new_img = cv2.merge([newb, newg, newr])

	return new_img

def get_pixels_values(img1, img2,x2, y2, x3, y3):
	xr,yr = [0], [0]
	xg,yg = [0], [0]
	xb,yb = [0], [0]

	for n in range (0,npoints):
		pix1 = img1[x2[n],y2[n]]
		pix2 = img2[x3[n],y3[n]]

		xr.append(pix1[0])
		xg.append(pix1[1])
		xb.append(pix1[2])

		yr.append(pix2[0])
		yg.append(pix2[1])
		yb.append(pix2[2])

	xr.append(255)
	xg.append(255)
	xb.append(255)

	yr.append(255)
	yg.append(255)
	yb.append(255)

	xr, yr = sort_array(xr, yr)

	xg, yg = sort_array(xg, yg)

	xb, yb = sort_array(xb, yb)
	return xr, yr, xg, yg, xb, yb

def pitxels_array(pixels1, pixels2):
	pixels1 = (np.array(pixels1)*255).astype(int)
	pixels2 = (np.array(pixels2)*255).astype(int)
	
	xr,yr = [0], [0]
	xg,yg = [0], [0]
	xb,yb = [0], [0]

	if pixels1.shape == pixels2.shape:

		for n in range (0, pixels1.shape[0]):

			xr.append(pixels1[n][0])
			xg.append(pixels1[n][1])
			xb.append(pixels1[n][2])

			yr.append(pixels2[n][0])
			yg.append(pixels2[n][1])
			yb.append(pixels2[n][2])

		xr.append(255)
		xg.append(255)
		xb.append(255)

		yr.append(255)
		yg.append(255)
		yb.append(255)

		xr, yr = sort_array(xr, yr)

		xg, yg = sort_array(xg, yg)

		xb, yb = sort_array(xb, yb)
	else:
		print ("Take equal number of points")
		return None, None, None, None, None, None, False
	return xr, yr, xg, yg, xb, yb, True

def white_balance(img, lim_):
	balanced_img = np.zeros_like(img) #Initialize final image
	# lim_ = 0.0005
	# lim_ = 0.005
	for i in range(3): #i stands for the channel index 
	    hist, bins = np.histogram(img[..., i].ravel(), 256, (0, 256))
	    bmin = np.min(np.where(hist>(hist.sum()*lim_)))
	    bmax = np.max(np.where(hist>(hist.sum()*lim_)))
	    balanced_img[...,i] = np.clip(img[...,i], bmin, bmax)
	    balanced_img[...,i] = (balanced_img[...,i]-bmin) / (bmax - bmin) * 255
	return balanced_img

def get_circle_vals(xo,yo,r,img):
	all_points = 0
	sum_all = 0
	j = 0
	for m in range(xo-r,xo+r):
		for n in range(yo-r,yo+r):
			if ((m-xo)**2 + (n-yo)**2 <= r**2):				
				all_points = all_points+1				
				sum_all = sum_all + img[m, n]
				j = j +1
	
	avr_val = sum_all/all_points
	return avr_val
	
def luminance_circles(outimg, listcoor, Y, cx, cy, s1, dist, path):
	# with open('luminance.csv') as f:
 #    	ROI = f.readlines()

	line1 = "Point, X, Y,radius, Lx mean,Cx, Cy \n"
	with open(path+"luminance.csv", "w") as file_object:
		file_object.write(line1)

	font = cv.FONT_HERSHEY_SIMPLEX

	for i, cord in enumerate(listcoor):
		lx = cx[cord[1], cord[0]]
		r = cord[2]
		cv.circle(outimg, (cord[0],cord[1]), r, [0, 255, 0],thickness=2, lineType=8, shift=0)
		cv.putText(outimg, 'Point {}'.format(i), (cord[0]+r,cord[1]), font, s1, (0, 255, 0), 2, cv.LINE_4)
		# print ("lx[0]",lx[0],"lx[1]",lx[1],"radius#",r)
		Y_mean = get_circle_vals(cord[1],cord[0],r,Y)
		# Yv = Y[cord[1],cord[0]]
		Yv = "{:.2f}".format(Y_mean)

		cv.putText(outimg, 'Lx mean {} cd/m2'.format(Yv), (cord[0]+int(r*1.1),cord[1]+dist),font, s1, (0, 255, 0), 2, cv.LINE_4 )
		
		Cxv = cx[cord[1],cord[0]]
		Cxv = "{:.4f}".format(Cxv)
		cv.putText(outimg, 'Cx {}'.format(Cxv), (cord[0]+int(r*1.1),cord[1]+int(2*dist)),font, s1, (0, 255, 0), 2, cv.LINE_4 )
		
		Cyv = cy[cord[1],cord[0]]
		Cyv = "{:.4f}".format(Cyv)
		cv.putText(outimg, 'Cy {}'.format(Cyv), (cord[0]+int(r*1.1),cord[1]+int(3*dist)),font, s1, (0, 255, 0), 2, cv.LINE_4 )
		
		line = "{}, {}, {}, {}, {}, {}, {}".format(i, cord[1], cord[0], r, Y_mean, Cxv, Cyv) +"\n"
		with open(path+"luminance.csv", "a") as file_object:
			file_object.write(line)
	return outimg

def create_color_bar(color_map):
    img1 = np.zeros((255, 50,3))
    img2 = (np.ones((255, 40,3))*255).astype(np.uint8)

    for n in range(0,255):    
        img1[n, :]= [255-n, 255-n, 255-n]   

    img1 = img1.astype(np.uint8)

    imgb = cv2.applyColorMap(img1, color_map)
    font = cv2.FONT_HERSHEY_SIMPLEX

    vis = np.concatenate((imgb, img2), axis=1)

    for l in range(1,13):
        idx = "{:.0f}".format((13-l)*20)
        cv2.putText(vis, '{}'.format(idx), (60,l*20), font, 0.4, (0, 0, 0), 1, cv2.LINE_4)
    return vis

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

def curve_colorY(X, Y, Z, xr, yr, xg, yg, xb, yb, degree):	

	# xr= xr.reshape(-1, 1)
	# yr= yr.reshape(-1, 1)

	xg= xg.reshape(-1, 1)
	yg= yg.reshape(-1, 1)	

	# xb= xb.reshape(-1, 1)
	# yb= yb.reshape(-1, 1)

	# polyreg1 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	polyreg2 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	# polyreg3 = make_pipeline(PolynomialFeatures(degree),LinearRegression())
	
	# polr = polyreg1.fit(xg,yg)
	polg = polyreg2.fit(yg, xg)
	# polb = polyreg3.fit(xg,yg)

	r = X.reshape(-1, 1)
	newr = (polg.predict(r)).astype(np.uint8)
	newr = newr.reshape(X.shape[0], X.shape[1])
	
	# #channel g
	g = Y.reshape(-1, 1)    
	newg = (polg.predict(g)).astype(np.uint8)    
	newg = newg.reshape(X.shape[0], X.shape[1])

	#channel b    
	b = Z.reshape(-1, 1)
	newb = (polg.predict(b)).astype(np.uint8)    
	newb = newb.reshape(X.shape[0], X.shape[1])

	new_img = cv2.merge([newb, newg, newr])

	return newr, newg, newb
def pitxels_array2(pixels1, pixels2):
	pixels1 = np.array(pixels1)
	pixels2 = np.array(pixels2)
	
	xr,yr = [0], [0]
	xg,yg = [0], [0]
	xb,yb = [0], [0]

	if pixels1.shape == pixels2.shape:

		for n in range (0, pixels1.shape[0]):

			xr.append(pixels1[n][0])
			xg.append(pixels1[n][1])
			xb.append(pixels1[n][2])

			yr.append(pixels2[n][0])
			yg.append(pixels2[n][1])
			yb.append(pixels2[n][2])

		xr.append(255)
		xg.append(255)
		xb.append(255)

		yr.append(255)
		yg.append(255)
		yb.append(255)

		xr, yr = sort_array(xr, yr)

		xg, yg = sort_array(xg, yg)

		xb, yb = sort_array(xb, yb)
	else:
		print ("Take equal number of points")
		return None, None, None, None, None, None, False
	return xr, yr, xg, yg, xb, yb, True

print ('Done!')
print ("DONE new update!!!")