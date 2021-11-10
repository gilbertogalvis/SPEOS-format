
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPainter, QColor, QPen
from PIL import Image, ImageQt
from PyQt5.QtGui import QImage
import numpy as np
from xyz_adjust import *
from generate_images import save_map

class Label(QtWidgets.QWidget):
    def __init__(self, path, parent=None):
        super(Label, self).__init__(parent)
        self.points = []
        self.pixels = []
        self.pcount = 1
        self.image = QtGui.QPixmap(path)
        # self.ui.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.image = self.image.scaledToWidth(400)
        # self.image.setScaledContents(True)
        self.image = self.image.scaled(350, 200)

        self.drawing = False
        self.lastPoint = QtCore.QPoint()

    def paintEvent(self, event):
        
        painter = QtGui.QPainter(self)
        painter.drawPixmap(QtCore.QPoint(), self.image)
        painter.setPen(QtGui.QColor(0,255,0))
        pen = QtGui.QPen()
        pen.setWidth(30)
        painter.setPen(pen)        
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            painter = QtGui.QPainter(self.image)
            painter.setPen(QtGui.QPen(QtCore.Qt.green, 3, QtCore.Qt.SolidLine))
            r = 15
            painter.drawEllipse(event.pos(),r,r)
            painter.drawText(event.pos().x(), event.pos().y(), "Point {}".format(str(self.pcount)) )            
            self.drawing = True
            self.lastPoint = event.pos()
            self.points.append([event.pos().x(), event.pos().y(), self.pcount])

            print (">>>>>>>>>>>>>>event.pos().x()",event.pos().x(),"event.pos().y()",  event.pos().y())
            x = event.pos().x()
            y = event.pos().y()
            c = self.image.toImage().pixel(x,y)
            colors = QColor(c).getRgbF()
            self.pixels.append(colors)            
            self.pcount = self.pcount +1
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
            painter = QtGui.QPainter(self.image)
            painter.setPen(QtGui.QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))            
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == QtCore.Qt.LeftButton:
            self.drawing = False

    def sizeHint(self):
        return self.image.size()
############################## >>>>>>>>>>>>>>>>>>>>>>>>>> LABEL2 <<<<<<<<<<<<<<<<<<<<<<<<<<< ################################
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
class Label2(QtWidgets.QWidget):
	def __init__(self, path, parent=None):
		super(Label2, self).__init__(parent)
		self.points = []
		self.pixels = []
		self.pcount = 1
		self.image = QtGui.QPixmap(path)
		
		print ("  >>>>>>>>>>self.image Label2	SIZE 1 ###", self.image.size())
		size1 = self.image.size()

		
		new_height = 500

		self.rate_x = (self.image.height()/new_height)

		self.image = self.image.scaledToHeight(new_height)

		self.image = self.image.scaled(800, new_height)

		print ("  >>>>>>>>>>self.image Label2	SIZE 1 ###", self.image.size())

		print ("New image size", self.image.size())

		print ("self.rate_x", self.rate_x)

		self.drawing = False
		self.lastPoint = QtCore.QPoint()
		self.textEdit = "Point, X, Y, radius, Lx mean, Cx,Cy \n"
		# path_files = "/home/jorge/work/A_speos/GUI/files_completes/XYZ_data/"
		path_files = "a_XYZ_data/"

		
		with open(path_files+'Y.npy', 'rb') as f:
			Ya = np.load(f)

		with open(path_files+'cx.npy', 'rb') as f:
			cx = np.load(f)
		
		with open(path_files+'cy.npy', 'rb') as f:
			cy = np.load(f)
		
				
		self.luminance_array = Ya
		self.cx = cx
		self.cy = cy

	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawPixmap(QtCore.QPoint(), self.image)
		painter.setPen(QtGui.QColor(0,255,0))
		pen = QtGui.QPen()
		pen.setWidth(30)
		painter.setPen(pen)        
		painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

	def mousePressEvent(self, event):
		if event.button() == QtCore.Qt.LeftButton:
			painter = QtGui.QPainter(self.image)
			painter.setPen(QtGui.QPen(QtCore.Qt.green, 3, QtCore.Qt.SolidLine))
			r = self.RADIUS
			line_s = 10
			painter.drawEllipse(event.pos(),r,r)
			painter.drawText(event.pos().x()+r, event.pos().y(), "Point {}".format(str(self.pcount)) )

			print ("Label 2>>>>event.pos().x()",event.pos().x(),"event.pos().y()",  event.pos().y())
			# luminance_val = self.luminance_array[event.pos().x(), event.pos().y()]
			x_, y_ = event.pos().x(), event.pos().y()
			print ("x_, y_", x_, y_)

			nx, ny = int(x_*self.rate_x), int(y_*self.rate_x)
			print ("Nx, Ny",nx, ny)

			lm_val = get_circle_vals(ny ,nx,r,self.luminance_array)
			cx_val = get_circle_vals(ny ,nx,r,self.cx)
			cy_val = get_circle_vals(ny ,nx,r,self.cy)

			lm_val = "{:.2f}".format(lm_val)
			cx_val = "{:.2f}".format(cx_val)
			cy_val = "{:.2f}".format(cy_val)
			
			painter.drawText(event.pos().x()+r, event.pos().y()+line_s, "Lx mean {} cd/m2".format(str(lm_val)) )
			painter.drawText(event.pos().x()+r, event.pos().y()+2*line_s, "Cx mean {}".format(str(cx_val)) )
			painter.drawText(event.pos().x()+r, event.pos().y()+3*line_s, "Cy mean {}".format(str(cy_val)) )

			self.textEdit = self.textEdit + ("{}, {}, {}, {}, {}, {}, {} \n".format(self.pcount,
				x_,y_,r,lm_val,cx_val, cy_val))

			self.drawing = True
			self.lastPoint = event.pos()
			self.points.append([event.pos().x(), event.pos().y(), self.pcount])
			x = event.pos().x()
			y = event.pos().y()
			c = self.image.toImage().pixel(x,y)
			colors = QColor(c).getRgbF()
			self.pixels.append(colors)
			self.pcount = self.pcount +1
			self.update()

	def mouseMoveEvent(self, event):
		if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
			painter = QtGui.QPainter(self.image)
			painter.setPen(QtGui.QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))            
			self.lastPoint = event.pos()
			self.update()

	def mouseReleaseEvent(self, event):
		if event.button == QtCore.Qt.LeftButton:
			self.drawing = False
################################################   LABEL 3 #####################################################################
class Label3(QtWidgets.QWidget):
	def __init__(self, path,lim_min=0, lim_max=255, parent=None, fl_crop=False, pts = [0,0,0,0], new_width = 400, new_height = 250):
		super(Label3, self).__init__(parent)
		self.points = []
		self.pixels = []
		self.pcount = 1
		self.image = QtGui.QPixmap(path)
		size1 = self.image.size()		
		self.textEdit = "Point, X, Y, Z, radius, Lx mean, Cx,Cy \n"
		self.X1, self.Y1, self.Z1, self.cx, self.cy = get_XYZ(path)
		self.pixelsout = (self.Y1 >lim_max).sum()
		self.all_pix = self.Y1.shape[0]*self.Y1.shape[1]
    	
		valnorm = 255
		self.X, self.Y, self.Z = cleanxyzV2(self.X1, self.Y1, self.Z1, lim_min, lim_max)
		self.X, self.Y, self.Z = normalization(self.X, self.Y, self.Z, valnorm)
		

		img_size = self.X.shape
		if fl_crop:
			self.p1x, self.p1y = pts[0], pts[1]
			self.p2x, self.p2y = pts[1], pts[3]
			width_  = (pts[2]-pts[0])
			height_ = (pts[3]-pts[1])
			
		else:			
			self.p1x, self.p1y = 0 , 0
			self.p2x, self.p2y = self.X.shape[1], self.X.shape[0]
			# pts[px1, py1, px2, py2]
			width_ = self.X.shape[1]
			height_ = self.X.shape[0]

		self.rate_x = (width_/new_width)
		self.rate_y = (height_/new_height)

		self.image = self.image.scaled(new_width, new_height)

		self.drawing = False
		self.pt1 = False
		self.lastPoint = QtCore.QPoint()
		self.textEdit = "Point, X, Y, Cx,Cy \n"


	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawPixmap(QtCore.QPoint(), self.image)
		painter.setPen(QtGui.QColor(0,255,0))
		pen = QtGui.QPen()
		pen.setWidth(30)
		painter.setPen(pen)        
		painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

	def mousePressEvent(self, event):

		if event.button() == QtCore.Qt.LeftButton:
			# print("inside mouse event>>>>>>>>>>>>>>>>")
			# print(np.mean(self.Y), np.std(self.Y), np.max(self.Y))
			painter = QtGui.QPainter(self.image)
			painter.setPen(QtGui.QPen(QtCore.Qt.green, 3, QtCore.Qt.SolidLine))
			# luminance_val = self.luminance_array[event.pos().x(), event.pos().y()]
			x_, y_ = (event.pos().x()), (event.pos().y())
			
			new_event = True

			if self.flag_sqr:

				if self.pt1 and new_event:
					
					painter.drawEllipse(event.pos(),3,3)
					self.p2x, self.p2y = x_, y_					
					width  = self.p2x - self.p1x
					height = self.p2y - self.p1y
					
					painter.drawRect(self.p1x, self.p1y, width, height)
					self.pt1 = False
					new_event = False					
					self.p1x, self.p1y = int(self.p1x*self.rate_x), int(self.p1y*self.rate_y)
					self.p2x, self.p2y = int(self.p2x*self.rate_x), int(self.p2y*self.rate_y)
					self.drawing = True
					self.update()
			
				if ~self.pt1 and new_event:					
					self.p1x, self.p1y = x_, y_
					painter.drawEllipse(event.pos(),3,3)
					self.pt1 = True
					self.drawing = True
					self.update()
			else:
				r = 10 #self.RADIUS
				line_s = 12
				painter.drawEllipse(event.pos(),r,r)		

				Xn = self.X[self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x)]
				Yn = self.Y[self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x)]
				Yn1 = self.Y1[self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x)]
				Zn = self.Z[self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x)]
				cx = self.cx[self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x)]
				cy = self.cy[self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x)]

				Xn = "{:4.2f}".format(Xn)
				Yn = "{:4.2f}".format(Yn)
				# Yn1 = "{:4.2f}".format(Yn1)
				print("Yn value", Yn)
				Zn = "{:4.2f}".format(Zn)

				r = 15
				cx_mean = get_circle_vals(self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x),r,self.cx)
				cy_mean = get_circle_vals(self.p1y + int(y_*self.rate_y), self.p1x + int(x_*self.rate_x),r,self.cy)


				cx = "{:4.2f}".format(cx_mean)
				cy = "{:4.2f}".format(cy_mean)
				
				# self.textEdit = self.textEdit + ("{}, {}, {}, {}, {}, {}, {} \n".format(self.pcount,
				# x_,y_,r,lm_val,cx_val, cy_val))
				self.textEdit = self.textEdit + ("{}, {}, {}, {}, {}, {} \n".format(self.pcount, Xn,Yn,Zn, cx, cy))

				# painter.drawText(event.pos().x()+r, event.pos().y() + line_s, "X {}".format(str(Xn)) )
				painter.drawText(event.pos().x()+r, event.pos().y(), "Y {}".format(str(Yn)) )
				# painter.drawText(event.pos().x()+r, event.pos().y()+line_s, "Y {}".format(str(Yn1)) )
				# painter.drawText(event.pos().x()+r, event.pos().y()+3*line_s, "Z {}".format(str(Zn)) )


				self.drawing = True
				# self.lastPoint = event.pos()
				# self.points.append([event.pos().x(), event.pos().y(), self.pcount])
				# x = event.pos().x()
				# y = event.pos().y()
				# c = self.image.toImage().pixel(x,y)
				# colors = QColor(c).getRgbF()
				self.pixels.append([Xn, Yn, Zn])
				self.pcount = self.pcount +1
				self.update()

	def mouseMoveEvent(self, event):
		if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
			painter = QtGui.QPainter(self.image)
			painter.setPen(QtGui.QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))            
			self.lastPoint = event.pos()
			self.update()

	def mouseReleaseEvent(self, event):
		if event.button == QtCore.Qt.LeftButton:
			self.drawing = False
###################### Label for show matrix Y 
class Label_ymatrix(QtWidgets.QWidget):
	def __init__(self, path, ymatrix, parent=None, new_width = 400, new_height = 250, list_coor = []):
		super(Label_ymatrix, self).__init__(parent)		
		self.list_coor = list_coor
		self.points = []
		self.pixels = []
		self.pcount = 1
		self.image = QtGui.QPixmap(path)
		self.Y = cv2.imread(path)
		self.ymatrix = ymatrix
		size1 = self.image.size()
		width_, height_ = self.image.width(), self.image.height()
		self.rate_x = (width_/new_width)
		self.rate_y = (height_/new_height)
		self.image = self.image.scaled(new_width, new_height)
		self.drawing = False
		self.lastPoint = QtCore.QPoint()

	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawPixmap(QtCore.QPoint(), self.image)
		painter.setPen(QtGui.QColor(100,255,0))
		pen = QtGui.QPen()
		pen.setWidth(30)
		painter.setPen(pen)        
		painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

	def mousePressEvent(self, event):
		if event.button() == QtCore.Qt.LeftButton:
			painter = QtGui.QPainter(self.image)
			painter.setPen(QtGui.QPen(QtCore.Qt.green, 3, QtCore.Qt.SolidLine))
			x_, y_ = (event.pos().x()), (event.pos().y())
			r = 10
			line_s = 12		

			Yn = self.ymatrix[int(y_*self.rate_y), int(x_*self.rate_x)]
			Yn = "{:3.1f}".format(Yn)
			painter.drawEllipse(event.pos(),r,r)
			painter.drawText(event.pos().x() + r+3, event.pos().y(), "Point {}".format(str(self.pcount)) )
			painter.drawText(event.pos().x() + r+3, event.pos().y()+line_s, "Y {}".format(str(Yn)) )
			
			# self.list_coor.append([event.pos().x(), event.pos().y(), self.pcount])			
			# for coord_ in self.list_coor:
			# Yn = self.ymatrix[int(coord_[1]*self.rate_y), int(coord_[0]*self.rate_x)]
			
			# painter.drawEllipse(coord_[0],coord_[1],r,r)
			# painter.drawText(coord_[0] + r+3, coord_[1], "Y {}".format(str(Yn)) )

			self.drawing = True
			self.lastPoint = event.pos()
			self.points.append([event.pos().x(), event.pos().y(), self.pcount])			

			x = event.pos().x()
			y = event.pos().y()
			c = self.image.toImage().pixel(x,y)
			colors = QColor(c).getRgbF()
			self.pixels.append(colors)
			self.pcount = self.pcount +1
			self.update()

	def mouseMoveEvent(self, event):
		if event.buttons() and QtCore.Qt.LeftButton and self.drawing:
			painter = QtGui.QPainter(self.image)
			painter.setPen(QtGui.QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))            
			self.lastPoint = event.pos()
			self.update()

	def mouseReleaseEvent(self, event):
		if event.button == QtCore.Qt.LeftButton:
			self.drawing = False

	def sizeHint(self):
		return self.image.size()
