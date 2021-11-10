import sys
from PyQt5.QtWidgets import *
import time

from PyQt5.QtWidgets import (

    QApplication,

    QCheckBox,

    QFormLayout,

    QLineEdit,

    QVBoxLayout,

    QWidget,

    QHBoxLayout,
    QGridLayout,

)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from utils2 import *
from labels import *
from generate_images import *
import cv2 as cv2
from skimage import data
from skimage.color import rgb2xyz, xyz2rgb
from xyz_adjust import *
from numpy import savetxt
from XYZtoRGB import txt2XYZ
from read_img_from_speos import read_txtSpeos
from RebuildSpeosTxt2 import RebuildSpeosTxt

def cv2_to_pixmap(img_np):
    img_pqt = Image.fromarray(img_np, mode='RGB')
    qt_img = ImageQt.ImageQt(img_pqt)
    pixmap_r = QtGui.QPixmap.fromImage(qt_img)
    return pixmap_r

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

def crop_image(imgpath, p1x, p1y, p2x, p2y):
    print("Input path", imgpath)
    path_files = '/'.join(imgpath.split('/')[0:-1])
    # w, h = p2x-p1x, p2y-p1y
    img = cv2.imread(imgpath)
    crop_img = img[p1y:p2y, p1x:p2x]
    path_out = path_files+'/crop.png'
    cv2.imwrite(path_out,crop_img)
    return path_out

class Window(QWidget):

    make_crop = QtCore.pyqtSignal()
    # make_crop.connect(make_imgcrop)
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Color correction and Color bar")
        self.white_flag = True
        self.kernel_fil = False
        self.flag_sqr1 = False
        self.flag_sqr2 = False        

        self.outerLayout = QGridLayout()        
        self.topLayout = QGridLayout()
        self.nextLayout = QGridLayout()
        self.nextLayout2 = QGridLayout()        
        self.side_button0 = QGridLayout()
        self.side_button = QGridLayout()

        self.comboBox1 = QtWidgets.QComboBox()
        self.comboBox1.setObjectName(("comboBox"))
        self.comboBox1.addItem("Read txt files")
        self.comboBox1.addItem("Color correction")
        self.comboBox1.addItem("Color mode")
        self.comboBox1.addItem("XYZ MATCH")
        self.comboBox1.addItem("XYZ MATCH2")
        self.comboBox1.addItem("Speos txt format")
        
        self.comboBox2= QtWidgets.QComboBox()        
        self.comboBox2.addItem("COLORMAP_AUTUMN")
        self.comboBox2.addItem("COLORMAP_BONE")
        self.comboBox2.addItem("COLORMAP_JET")
        self.comboBox2.addItem("COLORMAP_WINTER")
        self.comboBox2.addItem("COLORMAP_RAINBOW")
        self.comboBox2.addItem("COLORMAP_OCEAN")
        self.comboBox2.addItem("COLORMAP_SUMMER")
        self.comboBox2.addItem("COLORMAP_SPRING")
        self.comboBox2.addItem("COLORMAP_COOL")
        self.comboBox2.addItem("COLORMAP_HSV")
        self.comboBox2.addItem("COLORMAP_PINK")
        self.comboBox2.addItem("COLORMAP_HOT")
        self.comboBox2.addItem("COLORMAP_PARULA")
        self.comboBox2.addItem("COLORMAP_MAGMA")
        self.comboBox2.addItem("COLORMAP_INFERNO")
        self.comboBox2.addItem("COLORMAP_PLASMA")
        self.comboBox2.addItem("COLORMAP_VIRIDIS")
        self.comboBox2.addItem("COLORMAP_CIVIDIS")
        self.comboBox2.addItem("COLORMAP_TWILIGHT")
        self.comboBox2.addItem("COLORMAP_TWILIGHT_SHIFTED")
        self.comboBox2.addItem("COLORMAP_TURBO")
        self.comboBox2.setCurrentIndex(20)
        # self.comboBox2.addItem("COLORMAP_DEEPGREEN")

        self.getImageButton0 = QtWidgets.QPushButton('Start')
        self.getImageButton1 = QtWidgets.QPushButton('Load IMG')
        self.getImageButton2 = QtWidgets.QPushButton('Load XYZ/SPEOS')
        self.getImageButton3 = QtWidgets.QPushButton('Brightness')

        self.getImageButton4 = QtWidgets.QPushButton('Color Correction')
        self.getImageButton5 = QtWidgets.QPushButton('Clear points')

        self.getImageButton6 = QtWidgets.QPushButton('White balance')
        # togglePushButton = QPushButton("Toggle Push Button")
        self.getImageButton6.setCheckable(True)
        self.getImageButton6.setChecked(False)

        self.radius_val =  QtWidgets.QLineEdit()
        self.radius_val.setText("15")
        self.kernel_size =  QtWidgets.QLineEdit()
        self.kernel_size.setText("3")
        self.getImageButton7 = QtWidgets.QPushButton('Radius')
        self.getImageButton8 = QtWidgets.QPushButton('Save')
        self.getImageButton9 = QtWidgets.QPushButton('Close')
        self.getImageButton10 = QtWidgets.QPushButton('Reset')
        self.getImageButton11 = QtWidgets.QPushButton('Save Image')
        self.getImageButton12 = QtWidgets.QPushButton('Kernel filter')
        self.getImageButton20 = QtWidgets.QPushButton('Save Points-M')
        self.getImageButton21 = QtWidgets.QPushButton('Save Points-S')
        self.getImageButton22 = QtWidgets.QPushButton('Save Adjust Img')
        self.getImagebrightMode = QtWidgets.QPushButton('Brightness Mode')
        self.getImageBack = QtWidgets.QPushButton('Back')
        self.clear_points1 = QtWidgets.QPushButton('Clear Points 1')
        self.clear_points2 = QtWidgets.QPushButton('Clear Points 2')

        self.saveY1 = QtWidgets.QPushButton('Save Y1')
        self.saveY2 = QtWidgets.QPushButton('Save Y2')

        self.getImageButton12.setCheckable(True)
        self.getImageButton12.setChecked(False)
        
        self.spinBox1 = QtWidgets.QSpinBox()
        self.spinBox2 = QtWidgets.QSpinBox()
        self.spinBox1.setRange(0, 255)
        self.spinBox2.setRange(0, 255)
        self.spinBox1.setValue(150)
        self.spinBox2.setValue(0)
        self.max_val = 150
        self.min_val = 0

        self.getImageButton4.clicked.connect(self.color_correction)
        self.getImageButton5.clicked.connect(self.clear_points)
        self.getImageButton6.clicked.connect(self.apply_white_balance)
        self.getImageButton7.clicked.connect(self.upload_radius)
        self.getImageButton8.clicked.connect(self.file_save)
        self.getImageButton10.clicked.connect(self.reset_window)
        self.getImageButton11.clicked.connect(self.save_image)
        self.getImageButton12.clicked.connect(self.kernel_filter)

        self.comboBox1.setFixedSize(140,40)
        self.comboBox2.setFixedSize(140,40)
        self.getImageButton0.setFixedSize(120,40)
        self.getImageButton1.setFixedSize(120,40)
        self.getImageButton2.setFixedSize(120,40)
        self.getImageButton3.setFixedSize(120,40)
        self.getImageButton4.setFixedSize(120,40)
        self.getImageButton5.setFixedSize(120,40)
        self.getImageButton6.setFixedSize(110,40)
        self.getImageButton7.setFixedSize(110,40)
        self.radius_val.setFixedSize(40,40)

        self.getImageButton8.setFixedSize(110,40)
        self.getImageButton9.setFixedSize(120,40)
        self.getImageButton10.setFixedSize(120,40)
        self.getImageButton11.setFixedSize(110,40)
        self.getImageButton12.setFixedSize(110,40)
        self.kernel_size.setFixedSize(40,40)

        self.getImageButton21.setFixedSize(150,40)
        self.getImageButton20.setFixedSize(150,40)
        self.getImageButton22.setFixedSize(110,40)
        self.getImagebrightMode.setFixedSize(110,40)
        self.getImageBack.setFixedSize(110,40)
        self.clear_points1.setFixedSize(110,40)
        self.clear_points2.setFixedSize(110,40)       
        self.saveY1.setFixedSize(110,40)
        self.saveY2.setFixedSize(110,40)        

        self.getImageButton0.clicked.connect(self.select_mode)
        self.getImageButton1.clicked.connect(self.load_image1)
        self.getImageButton2.clicked.connect(self.load_image2)
        self.getImageButton3.clicked.connect(self.upload_brightness)
        self.getImageButton20.clicked.connect(self.save_point_measured)
        self.getImageButton21.clicked.connect(self.save_point_simulated)
        self.getImageButton22.clicked.connect(self.save_adjust_img)    
        self.getImagebrightMode.clicked.connect(self.brightnessmode)
        # self.getImageBack.clicked.connect(self.brightnessmode)

        self.verticalSlider = QtWidgets.QSlider()
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setMinimum(0) # set min limit
        self.verticalSlider.setMaximum(100) # set max limit
        self.verticalSlider.setValue(0) # initiate value
        self.brightness_level = 0
        self.verticalSlider.setTickInterval(10)
        self.verticalSlider.valueChanged.connect(self.change_brightness)

        # vertical slider 1
        self.verticalSlider1 = QtWidgets.QSlider()
        self.verticalSlider1.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider1.setMinimum(0) # set min limit
        self.verticalSlider1.setMaximum(100) # set max limit
        self.verticalSlider1.setValue(0) # initiate value
        self.brightness_level1 = 0
        self.brightness_level1_old = 0
        self.verticalSlider1.setTickInterval(10)
        self.verticalSlider1.sliderReleased.connect(self.brightness1)
        # self.verticalSlider1.valueChanged.connect(self.brightness1)

        # vertical slider 2
        self.verticalSlider2 = QtWidgets.QSlider()
        self.verticalSlider2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider2.setMinimum(0) # set min limit
        self.verticalSlider2.setMaximum(100) # set max limit
        self.verticalSlider2.setValue(0) # initiate value
        self.brightness_level2 = 0
        self.brightness_level2_old = 0
        self.verticalSlider2.setTickInterval(10)
        self.verticalSlider2.sliderReleased.connect(self.brightness2)
        self.clear_points1.clicked.connect(self.clearimg1)
        self.clear_points2.clicked.connect(self.clearimg2)

        self.saveY1.clicked.connect(self.saveY1_ref)
        self.saveY2.clicked.connect(self.saveY2_sim)

        self.spinBox1.valueChanged.connect(self.change_max)
        self.spinBox2.valueChanged.connect(self.change_min)

        self.topLayout.addWidget(self.comboBox1, 0,0)
        self.topLayout.addWidget(self.getImageButton0, 0,1)

        self.outerLayout.addLayout(self.topLayout,0,0)
        self.outerLayout.addLayout(self.nextLayout,1,0)        
        self.outerLayout.addLayout(self.side_button0,2,0)
        self.outerLayout.addLayout(self.side_button,3,0)
        self.outerLayout.addLayout(self.nextLayout2,4,0,2,1)
        self.setLayout(self.outerLayout)
        self.showMaximized()      
    
    def saveY1_ref(self):
        name = QFileDialog.getSaveFileName(filter="(*.png*)")
        if name[0] !='':

            print("self.pathY1", self.pathY1)

            img_ref = cv2.imread(self.pathY1)
            gray = cv2.cvtColor(img_ref,cv2.COLOR_RGB2GRAY)

            print("self.pathY1", img_ref.shape)

            cv2.imwrite(name[0], img_ref)
            print("save matrix Y")
            base = os.path.splitext(name[0])[0]
            file2 = (name[0], base + '.npy')[1]
            file3 = (name[0], base + '.csv')[1]
            print("file3",file3)

            with open(file2, 'wb') as f:
                np.save(f, self.ref.Y)

            # save csv file
            print("gray.shape", gray.shape)
            print("gray pix#", gray[10,10], type(gray[10,10]))


            savetxt(file3, gray, delimiter=',')

            alert = QMessageBox()
            alert.setText('Saved Succesfully!')
            alert.exec()

    def saveY2_sim(self):
        print("Save image 2")
        name = QFileDialog.getSaveFileName(filter="(*.png*)")
        if name[0] !='':
            print("Name", name[0])
            print("self.pathY2", self.pathY2)

            img_ref = cv2.imread(self.pathY2)
            gray = cv2.cvtColor(img_ref,cv2.COLOR_RGB2GRAY)
            cv2.imwrite(name[0], img_ref)
            
            base = os.path.splitext(name[0])[0]
            file2 = (name[0], base + '.npy')[1]
            file3 = (name[0], base + '.csv')[1]
            print("save matrix Y2", file2)

            with open(file2, 'wb') as f:
                np.save(f, self.imgCorr)

            # save csv file
            savetxt(file3, gray, delimiter=',')

            alert = QMessageBox()
            alert.setText('Saved Succesfully!')
            alert.exec()        

    def change_val1(self, value):        
        self.brightness_level1 = value

    def change_val2(self, value):        
        self.brightness_level2 = value

    def clearimg1(self):        
        self.pathY1 = get_image_set2("images/brightmode/Y_matrix.png",self.brightness_level1, 20, 0, 0)
        self.y_ref = Label_ymatrix(self.pathY1, self.ref.Y, list_coor = [])
        self.nextLayout.addWidget(self.y_ref,0,0,6,8)
        print("clear img1 self.y_ref.points", self.y_ref.points)
        self.pointsY1 = []
        
    def clearimg2(self):
        self.pathY2 = get_image_set2("images/brightmode/Y_matrix_correct.png", self.brightness_level2, 20, 0, 0)
        self.y_corr = Label_ymatrix(self.pathY2, self.Ysnew,list_coor = [])        
        self.nextLayout.addWidget(self.y_corr,0,10,6,8)
        print("clear img 2self.y_corr.points", self.y_corr.points)
        self.pointsY2 = []
        
    def brightness1(self):
        self.verticalSlider1.valueChanged.connect(self.change_val1)
        self.pathY1 = get_image_set2("images/brightmode/Y_matrix.png",self.brightness_level1, 20, 0, 0)                
        self.y_ref = Label_ymatrix(self.pathY1, self.ref.Y)
        self.nextLayout.addWidget(self.y_ref,0,0,6,8)       
        
    def brightness2(self):
        self.verticalSlider2.valueChanged.connect(self.change_val2)
        self.pathY2 = get_image_set2("images/brightmode/Y_matrix_correct.png", self.brightness_level2, 20, 0, 0)
        
        self.y_corr = Label_ymatrix(self.pathY2, self.Ysnew)
        self.nextLayout.addWidget(self.y_corr,0,10,6,8)

    def brightnessmode(self):
        # add button in top layout
        for i in reversed(range(self.topLayout.count())): 
            self.topLayout.itemAt(i).widget().setParent(None)
        
        self.topLayout.addWidget(self.getImageBack, 0, 0)
        self.topLayout.addWidget(self.clear_points1, 0, 1)
        self.topLayout.addWidget(self.saveY1, 0, 2)
        self.topLayout.addWidget(self.clear_points2, 0, 3)
        self.topLayout.addWidget(self.saveY2, 0, 4)

        # if self.comboBox1.currentIndex() == 3|4:
        if self.comboBox1.currentIndex() == 3 or self.comboBox1.currentIndex() == 4:
            for i in reversed(range(self.nextLayout.count())): 
                try:
                    self.nextLayout.itemAt(i).widget().setParent(None)
                except:
                    print ("Widget not null")

            for i in reversed(range(self.nextLayout2.count())): 
                try:
                    self.nextLayout2.itemAt(i).widget().setParent(None)
                except:
                    print ("Widget not null")

            for i in reversed(range(self.side_button.count())):
                self.side_button.itemAt(i).widget().setParent(None)

            for i in reversed(range(self.side_button0.count())):
                self.side_button0.itemAt(i).widget().setParent(None)

            for i in reversed(range(self.nextLayout2.count())): 
                try:
                    self.nextLayout2.itemAt(i).widget().setParent(None)
                except:
                    print("Error Top layout")

        self.imgCorr = self.Ysnew[self.p1y:self.p2y , self.p1x:self.p2x]

        with open('images/brightmode/Y_ref.npy', 'wb') as f:
              np.save(f, self.ref.Y)

        with open('images/brightmode/Y_corr.npy', 'wb') as f:
              np.save(f, self.imgCorr)        
        
        cv2.imwrite("images/brightmode/Y_matrix.png" , self.ref.Y)
        cv2.imwrite("images/brightmode/Y_matrix_correct.png", self.imgCorr)
        
        self.filtering = True
        if self.filtering:
            self.imgCorr = cv2.medianBlur(self.imgCorr, int(self.filter_size.text()) )

        self.pathY1 = get_image_set2("images/brightmode/Y_matrix.png",0, 20, 0, 0)        
        self.y_ref = Label_ymatrix(self.pathY1, self.ref.Y)        
        self.nextLayout.addWidget(self.ref,0,0,4,6)        

        cv2.imwrite("images/brightmode/Y_matrix_correct.png", self.imgCorr)        
        self.pathY2 = get_image_set2("images/brightmode/Y_matrix_correct.png", self.brightness_level2, 20, 0, 0)
        self.y_corr = Label_ymatrix(self.pathY2, self.imgCorr)       

        self.nextLayout.addWidget(self.y_ref,0,0,6,8)
        self.nextLayout.addWidget(self.y_corr,0,10,6,8)
        self.nextLayout.addWidget(self.verticalSlider1,0,9,6,1)
        self.nextLayout.addWidget(self.verticalSlider2,0,19,6,1)        

        self.real_vals = QtWidgets.QPushButton('Real vals')
        self.real_vals.setFixedSize(110,40)
        self.real_vals.clicked.connect(self.apply_real_vals)        
        self.side_button0.addWidget(self.real_vals, 0, 1)

        ##############################################################
        self.real_vals = QtWidgets.QPushButton('Save Y sim')
        self.real_vals.setFixedSize(110,40)
        self.real_vals.clicked.connect(self.save_real_vals1)        
        self.side_button0.addWidget(self.real_vals, 0, 2)

        self.real_vals = QtWidgets.QPushButton('Save Y speos')
        self.real_vals.setFixedSize(110,40)
        self.real_vals.clicked.connect(self.save_real_vals2)
        self.side_button0.addWidget(self.real_vals, 0, 3)

    def save_real_vals1(self):        

        name = QFileDialog.getSaveFileName(filter="(*.png*)")
        if name[0] !='':            
            img_ref = cv2.imread("images/brightmode/Y_Mreal.png")            
            gray = cv2.cvtColor(img_ref,cv2.COLOR_RGB2GRAY)            
            
            base = os.path.splitext(name[0])[0]            
            filename = (name[0], base + '.csv')[1]           

            # save csv file
            savetxt(filename, gray, delimiter=',')

            alert = QMessageBox()
            alert.setText('Saved Succesfully!')
            alert.exec()

    def save_real_vals2(self):        

        name = QFileDialog.getSaveFileName(filter="(*.png*)")
        if name[0] !='':                        
            img_ref = cv2.imread("images/brightmode/Y_Sreal.png")
            gray = cv2.cvtColor(img_ref,cv2.COLOR_RGB2GRAY)            
            
            base = os.path.splitext(name[0])[0]            
            filename = (name[0], base + '.csv')[1]           

            # save csv file
            savetxt(filename, gray, delimiter=',')

            alert = QMessageBox()
            alert.setText('Saved Succesfully!')
            alert.exec()

    def apply_real_vals(self):

        print("Real values")

        self.yMreal = (self.ref.Y/255)*int(self.max_valM.text())
        self.ySreal = (self.imgCorr/255)*int(self.max_valS.text())
        
        cv2.imwrite("images/brightmode/Y_Mreal.png", self.yMreal)
        cv2.imwrite("images/brightmode/Y_Sreal.png", self.ySreal)

        colormapath = get_image_set2("images/brightmode/Y_matrix_correct.png", self.brightness_level2, 20, 0, 0)

        self.yrealM = Label_ymatrix(colormapath, self.yMreal)
        self.nextLayout2.addWidget(self.yrealM,0,0,6,8)

        colormapath2 = get_image_set2("images/brightmode/Y_Sreal.png", self.brightness_level2, 20, 0, 0)
        self.yrealS = Label_ymatrix(colormapath2, self.ySreal)
        self.nextLayout2.addWidget(self.yrealS,0,10,6,8)

    def apply_filter(self):
        if self.Filter_flag:            
            img = cv2.imread(self.path_color)
            filtered = cv2.medianBlur(img,int(self.filter_size.text()))
            cv2.imwrite(self.filt_color, filtered)
            path_updated = self.filt_color
        else:
            path_updated = self.path_color

        self.xyz_correct = Label3(path_updated, new_width = 800, new_height = 450)
        self.xyz_correct.flag_sqr = False
        self.xyz_correct.make_crop = False
        self.text = self.xyz_correct.textEdit            
        self.nextLayout2.addWidget(self.xyz_correct, 0, 1)        
        self.Filter_flag = not self.Filter_flag
        self.Filter.setCheckable(False)
        # self.Filter.setChecked(True)
      
    def adjust_img(self):        

        xr, yr, xg, yg, xb, yb, fl1 = pitxels_array2(self.ref.pixels, self.sim.pixels)       

        # xr = np.array([0,125,255])
        # yr = np.array([0,200,255])

        # xg = np.array([0,125,255])
        # yg = np.array([0,200,255])  

        # xb = np.array([0,125,255])
        # yb = np.array([0,200,255])
        
        if fl1:

            file_ = '/'.join(self.filename_sim.split('/')[0:-1])+"/results/xyz_adjust.png"
            file_white = '/'.join(self.filename_sim.split('/')[0:-1])+"/results/xyz_adjust_white.png"

            X,  Y,  Z, cxref, cyref = self.ref.X, self.ref.Y, self.ref.Z, self.ref.cx, self.ref.cy

            Xs, Ys, Zs, cx, cy  = self.sim.X, self.sim.Y, self.sim.Z, self.sim.cx, self.sim.cy

            # X, Y, Z, Xsn, Ysn,Zsn = cleanxyz(X, Y, Z, Xs, Ys, Zs)

            #second approach clan pixels in image reference image
            # lim_minM, lim_maxM = 0, 10
            # lim_minS, lim_maxS = 0, 70
            # valnorm = 255
            # X,Y,Z = cleanxyzV2(X,Y,Z, lim_minM, lim_maxM)
            # Xs,Ys,Zs = cleanxyzV2(Xs,Ys,Zs, lim_minS, lim_maxS)
            # X,Y,Z = normalization(X,Y,Z, valnorm)
            # Xs,Ys,Zs = normalization(Xs,Ys,Zs, valnorm)
            # normalization after delete outliers
            xyz_path = '/'.join(self.filename_sim.split('/')[0:-1])+"/xyz_rebuild.png"            

            degree = 2
            # print("before correction Ysnew>>>>>>>>>>>>>>>", np.mean(Ys), np.std(Ys))
            Xsnew, Ysnew, Zsnew = curve_colorY(Xs, Ys, Zs, xr, yr, xg, yg, xb, yb, degree)

            Xsnew = cv2.medianBlur(Xsnew,5)
            Ysnew = cv2.medianBlur(Ysnew,5)
            Zsnew = cv2.medianBlur(Zsnew,5)
            # print("After correction Ysnew>>>>>>>>>>>>>>>>", np.mean(Ysnew), np.std(Ysnew))
                        
            self.Xsnew, self.Ysnew, self.Zsnew =  Xsnew, Ysnew, Zsnew            
            folder_ref = '/'.join(self.filename_meas.split('/')[0:-1])
            folder_sim = '/'.join(self.filename_sim.split('/')[0:-1])
            Xsnew, Ysnew, Zsnew = crop_savematrix(Xsnew, Ysnew, Zsnew, cx, cy, self.p1y, self.p2y, self.p1x,self.p2x, folder_sim)
            X,  Y,  Z = crop_savematrix(X,  Y,  Z, cxref, cyref, self.ref.p1y, self.ref.p2y, self.ref.p1x, self.ref.p2x, folder_ref)
           
            maxx = np.max([np.max(Xsnew), np.max(Ysnew), np.max(Zsnew)])
            img1 = cv2.merge((Xsnew/maxx, Ysnew/maxx, Zsnew/maxx))
            # img1 = cv2.merge((Xsnew, Ysnew, Zsnew))

            img2 = xyz2rgb(img1)
            R, G , B = cv2.split(img2)
            img = cv2.merge((B,G,R))*255
            # img = img[y_ini, y_end: x_ini, x_end]
            # make crop of image.            
            # img = img#[self.p1y:self.p2y , self.p1x:self.p2x]
            cv2.imwrite(file_, img)

            lim_ = 0.0005
            img_corrected_white_xyz = white_balance(img, lim_)
            self.path_color = file_
            self.path_white = file_white
            cv2.imwrite(self.path_color, img)
            cv2.imwrite(self.path_white, img_corrected_white_xyz)

            cv2.imwrite(file_, img)            
            self.xyz_correct = Label3(self.path_white, new_width = 800, new_height = 450)
            self.xyz_correct.flag_sqr = False
            self.xyz_correct.make_crop = False
            self.text = self.xyz_correct.textEdit            
            self.nextLayout2.addWidget(self.xyz_correct, 0, 1)
            self.side_button.addWidget(self.getImageButton6, 0,1)
            self.side_button.addWidget(self.getImageButton5, 0,2) # clear points          
            self.side_button.addWidget(self.getImageButton7, 0,3)
            self.side_button.addWidget(self.radius_val     , 0,4)
            self.side_button.addWidget(self.getImageButton8, 0,5)
            self.side_button.addWidget(self.getImageButton22, 0,6)
            self.side_button.addWidget(self.getImageButton10, 0,7)
            self.side_button.addWidget(self.getImagebrightMode, 0,8)         

            # Filter image button definitions ########
            self.Filter_flag = True
            self.Filter = QtWidgets.QPushButton('Filter')
            self.Filter.setFixedSize(110,40)
            self.Filter.clicked.connect(self.apply_filter)
            self.filter_size =  QtWidgets.QLineEdit()
            self.filter_size.setText("3")
            self.filter_size.setFixedSize(110,40)
            # filter_size = int(self.filter_size.text())
            self.filter_size.text()

            self.side_button.addWidget(self.Filter, 0,9)
            self.side_button.addWidget(self.filter_size, 0,10)
            # save image with filter aplied
            img = img.astype(np.uint8)
            img_filtered = cv2.medianBlur(img,int(self.filter_size.text()))
            
            white_img = cv2.imread(self.path_white)
            white_filtered = cv2.medianBlur(white_img,int(self.filter_size.text()))            
            self.filt_color = '/'.join(self.filename_sim.split('/')[0:-1])+"/results/img_filter.png"
            self.filt_white = '/'.join(self.filename_sim.split('/')[0:-1])+"/results/white_filter.png"
            cv2.imwrite(self.filt_color, img_filtered)
            cv2.imwrite(self.filt_white, white_filtered)

        else:
            alert = QMessageBox()
            alert.setText('You have to select the same number of points!')
            alert.exec()       
        
        print ("Adjust XYZ done!!!")

    def test_connect(self):
        print ("Connect test")

    def connect_crop(self):
        self.make_crop.connect(self.make_imgcrop)
        # print ("Make crop connection")    

    def apply_white_balance(self):

        self.white_flag = not self.white_flag

        if self.comboBox1.currentIndex() == 0:

            if self.white_flag:
                self.luminance = Label2(self.path_white)
                self.nextLayout2.addWidget(self.luminance,0,0)
                self.luminance.RADIUS = int(self.radius_val.text())
              
            else:
                self.luminance = Label2(self.path_color)
                self.nextLayout2.addWidget(self.luminance,0,0)
                self.luminance.RADIUS = int(self.radius_val.text())
            print ("white_flag", self.white_flag)

        if self.comboBox1.currentIndex() == 2:

            if self.white_flag:
                self.xyz_correct = Label3(self.path_white, new_width = 800, new_height = 450)
                self.nextLayout2.addWidget(self.xyz_correct, 0, 1)              
                self.xyz_correct.flag_sqr = False
                self.xyz_correct.make_crop = False
            else:
                self.xyz_correct = Label3(self.path_color, new_width = 800, new_height = 450)
                self.nextLayout2.addWidget(self.xyz_correct, 0, 1)
                self.xyz_correct.flag_sqr = False
                self.xyz_correct.make_crop = False

    def LoadTxt_x(self):
        self.filenamex = QFileDialog.getOpenFileName(filter="(*.*)")[0]
        print("self.filenamex", self.filenamex)

    def LoadTxt_y(self):
        self.filenamey = QFileDialog.getOpenFileName(filter="(*.*)")[0]

    def LoadTxt_z(self):
        self.filenamez = QFileDialog.getOpenFileName(filter="(*.*)")[0]

    def LoadTxtSpeos(self):
        self.fileSpeos = QFileDialog.getOpenFileName(filter="(*.*)")[0]

    def PathOutFunc(self):
        self.pathout = QFileDialog.getExistingDirectory()
        print("self.pathout",self.pathout)

    def PathOutSpeos(self):
        self.pathoutSpeos = QFileDialog.getExistingDirectory()
        print("self.pathout",self.pathoutSpeos)

    def LoadNpy_x(self):
        self.PathNpyx = QFileDialog.getOpenFileName(filter="(*.*)")[0]

    def LoadNpy_y(self):
        self.PathNpyy = QFileDialog.getOpenFileName(filter="(*.*)")[0]

    def LoadNpy_z(self):
        self.PathNpyz = QFileDialog.getOpenFileName(filter="(*.*)")[0]

    def PathOutSpeosTxt(self):
        name = QFileDialog.getSaveFileName(filter="txt (*.txt*)")
        print("name", name)
        if name[0] !='':
            # self.PathOutSpeosTxt = QFileDialog.getOpenFileName(filter="(*.*)")
            print("self.PathOutSpeosTxt", self.PathOutSpeosTxt)

        else:   
            print ("Path empty")
            

    def SpeosTxtFormat(self):
        print("txt_format")
        RebuildSpeosTxt(self.PathNpyx, self.PathNpyx, self.PathNpyx, self.PathOutSpeosTxt)

    def BuildImage(self):
        print("Build Image")
        filename = 'RGB.png'
        height, width = 3264, 4896
        self.rgbpathout = txt2XYZ(self.filenamex, self.filenamey, self.filenamez, self.pathout, filename, height,width)
        print("self.rgbpathout", self.rgbpathout)
        self.XYZ = Label(self.rgbpathout)
        self.nextLayout.addWidget(self.XYZ,0,0,4,6)        

    def BuildImageFromSpeos(self):
        print("Build Image from Speos")
        filename = "speos.png"
        self.pathoutSpeos = read_txtSpeos(self.fileSpeos, self.pathoutSpeos, filename)
        self.SPEOS = Label(self.pathoutSpeos)
        self.nextLayout.addWidget(self.SPEOS,0,9,4,6)

    def load_Mimage(self):
        self.filename_meas = QFileDialog.getOpenFileName(filter="(*.*)")[0]
        # lim_min, lim_max = 0, 10
        self.ref = Label3(self.filename_meas, fl_crop = False)

        self.ref.flag_sqr = False
        self.ref.make_crop = False
        self.text = self.ref.textEdit
        self.nextLayout.addWidget(self.ref,0,0,4,6)
        self.topLayout.addWidget(self.getImageButton16, 0, 1)
        self.topLayout.addWidget(self.getImageButton18, 0, 2)
        self.topLayout.addWidget(self.getImageButton20, 0, 3)
        self.topLayout.addWidget(self.getImageButton14, 0, 4)
        
        #add statistical feactures
        if self.comboBox1.currentIndex() == 4:

            self.resultM =  QtWidgets.QPlainTextEdit()            
            self.resultM.insertPlainText("Statistics1:\n")

            self.resultM.setFixedSize(200, 200) 
            self.nextLayout.addWidget(self.resultM,0,7, 4, 2)

            self.min_valM =  QtWidgets.QLineEdit()
            self.min_valM.setText("0")
            self.min_valM.setFixedSize(40,40)

            self.max_valM =  QtWidgets.QLineEdit()
            self.max_valM.setText("255")
            self.max_valM.setFixedSize(40,40)                

            self.cleanM = QtWidgets.QPushButton('clean M')
            self.cleanM.setFixedSize(110,40)
            self.cleanM.clicked.connect(self.clean_M)

            self.side_button0.addWidget(self.min_valM, 0, 1)
            self.side_button0.addWidget(self.max_valM, 0, 2)
            self.side_button0.addWidget(self.cleanM, 0, 3)

    def load_Simage(self):
        self.filename_sim = QFileDialog.getOpenFileName(filter="(*.*)")[0]
        # lim_min, lim_max = 0, 70
        self.sim = Label3(self.filename_sim, fl_crop = False)        
        self.sim.flag_sqr = False
        self.sim.make_crop = False
        self.text = self.sim.textEdit        
        self.nextLayout.addWidget(self.sim,0,9,4,6)
        self.topLayout.addWidget(self.getImageButton17, 0, 5)
        self.topLayout.addWidget(self.getImageButton19, 0, 6)
        self.topLayout.addWidget(self.getImageButton21, 0, 7)
        self.topLayout.addWidget(self.getImageButton15, 0, 8)

        if self.comboBox1.currentIndex() == 4:

            # self.resultS =  QtWidgets.QLineEdit()
            self.resultS =  QtWidgets.QPlainTextEdit()
            self.resultS.insertPlainText("Statistics2:\n")
            self.resultS.setFixedSize(200,200)            
            self.nextLayout.addWidget(self.resultS,0,15, 4, 2)

            self.min_valS =  QtWidgets.QLineEdit()
            self.min_valS.setText("0")
            self.min_valS.setFixedSize(40,40)

            self.max_valS =  QtWidgets.QLineEdit()
            self.max_valS.setText("255")
            self.max_valS.setFixedSize(40,40)            

            self.cleanS = QtWidgets.QPushButton('clean M')
            self.cleanS.setFixedSize(110,40)
            self.cleanS.clicked.connect(self.clean_S)

            self.side_button0.addWidget(self.min_valS, 0, 4)
            self.side_button0.addWidget(self.max_valS, 0, 5)
            self.side_button0.addWidget(self.cleanS, 0, 6)

    def clean_M(self):         
        self.ref = Label3(self.path_crop1,lim_min=int(self.min_valM.text()), lim_max=int(self.max_valM.text()),fl_crop = True, pts = [self.p1xS, self.p1yS, self.p2xS, self.p2yS])
        self.ref.pt1 = False
        self.ref.flag_sqr = False
        self.ref.make_crop = False        
        self.getImageButton16.setCheckable(True)
        self.getImageButton16.setChecked(False)
        self.ref.flag_sqr = False        
        self.nextLayout.addWidget(self.ref,0,0,4,6)

        numpix, pixout = self.ref.all_pix, self.ref.pixelsout        
        pixout = (pixout/numpix)*100
        pixout = "{:3.2f}".format(pixout) 
        Ymean = "{:3.2f}".format(np.mean(self.ref.Y))
        Ystd = "{:3.2f}".format(np.std(self.ref.Y))

        self.resultM.insertPlainText("Total pixels\n")
        self.resultM.insertPlainText(str(numpix)+"\n")
        self.resultM.insertPlainText("% pixels clean \n")
        self.resultM.insertPlainText(str(pixout)+"\n")
        self.resultM.insertPlainText("Mean value\n")
        self.resultM.insertPlainText(Ymean+"\n")       
        self.resultM.insertPlainText("std value\n")        
        self.resultM.insertPlainText(Ystd+"\n")       

    def clean_S(self):
        print("Pts S", self.p1x, self.p1y, self.p2x, self.p2y)
        self.sim = Label3(self.path_crop2,lim_min=int(self.min_valS.text()), lim_max=int(self.max_valS.text()), fl_crop = True, pts = [self.p1x, self.p1y, self.p2x, self.p2y])
        self.sim.pt1 = False
        self.sim.flag_sqr = False
        self.sim.make_crop = False        
        self.getImageButton17.setCheckable(True)
        self.getImageButton17.setChecked(False)
        self.sim.flag_sqr = False        
        self.nextLayout.addWidget(self.sim,0,9,4,6)        
        numpix, pixout = self.sim.all_pix, self.sim.pixelsout
        Ymean = "{:3.1f}".format(np.mean(self.sim.Y))

        numpix, pixout = self.sim.all_pix, self.sim.pixelsout        
        pixout = (pixout/numpix)*100
        
        pixout = "{:3.2f}".format(pixout) 
        Ymean = "{:3.2f}".format(np.mean(self.sim.Y))
        Ystd = "{:3.2f}".format(np.std(self.sim.Y))

        self.resultS.insertPlainText("Total pixels\n")
        self.resultS.insertPlainText(str(numpix)+"\n")
        self.resultS.insertPlainText("% pixels clean \n")
        self.resultS.insertPlainText(str(pixout)+"\n")
        self.resultS.insertPlainText("Mean value\n")
        self.resultS.insertPlainText(Ymean+"\n")       
        self.resultS.insertPlainText("std value\n")
        self.resultS.insertPlainText(Ystd+"\n")        
        
    def save_point_measured(self):
        name = QFileDialog.getSaveFileName(filter="csv (*.csv*)")
        if name[0] !='':
            print ("name file", name)
            file = open(name[0],'w')
            self.text = self.ref.textEdit
            file.write(self.text)
            file.close()
        else:   
            print ("Path empty")

    def save_point_simulated(self):
        name = QFileDialog.getSaveFileName(filter="csv (*.csv*)")
        if name[0] !='':
            print ("name file", name)
            file = open(name[0],'w')
            self.text = self.sim.textEdit
            file.write(self.text)
            file.close()
        else:   
            print ("Path empty")

    def save_adjust_img(self):
        print ("Saving image <<<<<<<<<<<<<<<<<<<<<")
        name = QFileDialog.getSaveFileName(filter="(*.png*)")
        if name[0] !='':

            xr, yr, xg, yg, xb, yb, fl1 = pitxels_array2(self.ref.pixels, self.sim.pixels)
            X,  Y,  Z  = self.ref.X, self.ref.Y, self.ref.Z
            Xs, Ys, Zs = self.sim.X, self.sim.Y, self.sim.Z
            Xsn, Ysn, Zsn = cleanxyz(X, Y, Z, Xs, Ys, Zs)
            Xsnew, Ysnew, Zsnew = curve_color2(Xsn, Ysn, Zsn, xr, yr, xg, yg, xb, yb, 2)
            Xsnew = Xsnew[self.p1y:self.p2y , self.p1x:self.p2x]
            Ysnew = Ysnew[self.p1y:self.p2y , self.p1x:self.p2x]
            Zsnew = Zsnew[self.p1y:self.p2y , self.p1x:self.p2x]
            print(name[0])
            cv2.imwrite(name[0], Ysnew)
            cv2.imwrite('a_speos_data/Original_Y.png', Y)

        # with open(folder_sim + '/results/Y_adjust_2.npy', 'wb') as f:
        #   np.save(f, Ysnew)

        # with open(folder_sim + '/results/Z_adjust_3.npy', 'wb') as f:
        #     np.save(f, Zsnew)

            alert = QMessageBox()
            alert.setText('Saved Succesfully!')
            alert.exec()

    def make_crop1(self):
        self.p1xS, self.p1yS, self.p2xS, self.p2yS = self.ref.p1x, self.ref.p1y, self.ref.p2x, self.ref.p2y        
        self.path_crop1 = crop_image(self.filename_meas, self.p1xS, self.p1yS, self.p2xS, self.p2yS)        
        self.ref = Label3(self.path_crop1,lim_min=int(self.min_valM.text()), lim_max=int(self.max_valM.text()),fl_crop = True, pts = [self.p1xS, self.p1yS, self.p2xS, self.p2yS])
        self.ref.pt1 = False
        self.ref.flag_sqr = False
        self.ref.make_crop = False        
        self.getImageButton16.setCheckable(True)
        self.getImageButton16.setChecked(False)
        self.ref.flag_sqr = False        
        self.nextLayout.addWidget(self.ref,0,0,4,6)
            
    def make_crop2(self):    
        self.p1x, self.p1y, self.p2x, self.p2y = self.sim.p1x, self.sim.p1y, self.sim.p2x, self.sim.p2y
        self.path_crop2 = crop_image(self.filename_sim, self.p1x, self.p1y, self.p2x, self.p2y)        
        self.sim = Label3(self.path_crop2,lim_min=int(self.min_valS.text()), lim_max=int(self.max_valS.text()), fl_crop = True, pts = [self.p1x, self.p1y, self.p2x, self.p2y])
        self.sim.pt1 = False
        self.sim.flag_sqr = False
        self.sim.make_crop = False        
        self.getImageButton17.setCheckable(True)
        self.getImageButton17.setChecked(False)
        self.sim.flag_sqr = False        
        self.nextLayout.addWidget(self.sim,0,9,4,6)

    def crop_img1(self):
        self.ref.flag_sqr = not self.ref.flag_sqr

    def crop_img2(self):
        self.sim.flag_sqr = not self.sim.flag_sqr

    def kernel_filter(self):
        self.kernel_fil = not self.kernel_fil
        print ("Kernel value")
        kernel_val = int(self.kernel_size.text())
        print (kernel_val)
        img1 = cv2.imread(self.path_color)
        img2 = cv2.imread(self.path_white)

        if self.kernel_fil:

            if self.white_flag:
                fil_img1 = cv2.medianBlur(img2, kernel_val)
                # fil_img1= cv2.GaussianBlur(img1,(kernel_val,kernel_val),0)
                cv2.imwrite("images/filtered.png", fil_img1)
                self.luminance = Label2("images/filtered.png")
                self.nextLayout2.addWidget(self.luminance,0,0)

            else:
                # fil_img2= cv2.GaussianBlur(img1,(kernel_val,kernel_val),0)
                fil_img2 = cv2.medianBlur(img1, kernel_val)
                cv2.imwrite("images/filtered.png", fil_img2)
                self.luminance = Label2("images/filtered.png")
                self.nextLayout2.addWidget(self.luminance,0,0)
        else:
            if self.white_flag:                
                
                self.luminance = Label2(self.path_white)
                self.nextLayout2.addWidget(self.luminance,0,0)

            else:                
                self.luminance = Label2(self.path_color)
                self.nextLayout2.addWidget(self.luminance,0,0)

    def save_image(self):
        print ("Saving image <<<<<<<<<<<<<<<<<<<<<")
        name = QFileDialog.getSaveFileName(filter="(*.png*)")
        if name[0] !='':
            if self.white_flag:
                img = cv2.imread(self.path_white)
                cv2.imwrite(name[0], img)
            else:                
                img = cv2.imread(self.path_color)
                cv2.imwrite(name[0],img)
        else:
            print ("Path empty")

    def reset_window(self):
        for i in reversed(range(self.topLayout.count())): 
            self.topLayout.itemAt(i).widget().setParent(None)        

        if self.comboBox1.currentIndex() == 1:

            for i in reversed(range(self.side_button.count())): 
                self.side_button.itemAt(i).widget().setParent(None)

            for i in reversed(range(self.nextLayout.count())): 
                try:
                    self.nextLayout.itemAt(i).widget().setParent(None)
                except:
                    print("Error Top layout")

            for i in reversed(range(self.nextLayout2.count())): 
                try:
                    self.nextLayout2.itemAt(i).widget().setParent(None)
                except:
                    print("Error Top layout")           


        if (self.comboBox1.currentIndex() == 2):
            for i in reversed(range(self.nextLayout.count())):
                try:
                    self.nextLayout.itemAt(i).widget().setParent(None)
                except:
                    print ("Widget not null")

            for i in reversed(range(self.nextLayout2.count())): 
                try:
                    self.nextLayout2.itemAt(i).widget().setParent(None)
                except:
                    print("Error Top layout")

            for i in reversed(range(self.spinboxeside.count())):
                self.spinboxeside.itemAt(i).widget().setParent(None)

            # for i in reversed(range(self.nextLayout.count())):
            #     print (self.nextLayout.itemAt(i))
            # self.nextLayout.itemAt(3).widget().setParent(None)

        if self.comboBox1.currentIndex() == 3:
            for i in reversed(range(self.nextLayout.count())): 
                try:
                    self.nextLayout.itemAt(i).widget().setParent(None)
                    self.nextLayout2.itemAt(i).widget().setParent(None)
                    
                except:
                    print ("Widget not null")
            for i in reversed(range(self.side_button.count())):
                self.side_button.itemAt(i).widget().setParent(None)

            for i in reversed(range(self.nextLayout2.count())): 
                try:
                    self.nextLayout2.itemAt(i).widget().setParent(None)
                except:
                    print("Error Top layout")

        self.topLayout.addWidget(self.comboBox1, 0,0)
        self.topLayout.addWidget(self.getImageButton0, 0,1)
        print ("Clear done")

    def select_mode(self):
        if self.comboBox1.currentIndex() == 0:
            print("Mode 0")
            self.load_txt_x = QtWidgets.QPushButton('Load txt X')
            self.load_txt_y = QtWidgets.QPushButton('Load txt Y')
            self.load_txt_z = QtWidgets.QPushButton('Load txt Z')
            self.path_out = QtWidgets.QPushButton('Path out')
            self.build_image = QtWidgets.QPushButton('Build Image')
            ####    SPEOS TXT ####

            self.load_txt_speos = QtWidgets.QPushButton('Load Speos txt')
            self.path_out_speos = QtWidgets.QPushButton('Path out Speos')
            self.build_image_speos = QtWidgets.QPushButton('Build Img Speos')


            self.load_txt_x.setFixedSize(120,40)            
            self.load_txt_y.setFixedSize(120,40)
            self.load_txt_z.setFixedSize(120,40)
            self.build_image.setFixedSize(120,40)
            self.path_out.setFixedSize(120,40)
            self.load_txt_speos.setFixedSize(120,40)
            self.path_out_speos.setFixedSize(120,40)
            self.build_image_speos.setFixedSize(120,40)
            

            self.load_txt_x.clicked.connect(self.LoadTxt_x)
            self.load_txt_y.clicked.connect(self.LoadTxt_y)
            self.load_txt_z.clicked.connect(self.LoadTxt_z)
            self.build_image.clicked.connect(self.BuildImage)
            self.path_out.clicked.connect(self.PathOutFunc)

            self.load_txt_speos.clicked.connect(self.LoadTxtSpeos)
            self.path_out_speos.clicked.connect(self.PathOutSpeos)
            self.build_image_speos.clicked.connect(self.BuildImageFromSpeos)

            for i in reversed(range(self.topLayout.count())): 
                self.topLayout.itemAt(i).widget().setParent(None)
            # self.topLayout.addWidget(self.comboBox1, 0, 0)
            self.topLayout.addWidget(self.load_txt_x, 0, 0)
            self.topLayout.addWidget(self.load_txt_y, 0, 1)
            self.topLayout.addWidget(self.load_txt_z, 0, 2)            
            self.topLayout.addWidget(self.path_out, 0, 3)
            self.topLayout.addWidget(self.build_image, 0, 4)
            self.topLayout.addWidget(self.load_txt_speos, 0, 5)
            self.topLayout.addWidget(self.path_out_speos, 0, 6)
            self.topLayout.addWidget(self.build_image_speos, 0, 7)


        if self.comboBox1.currentIndex() == 1:
            for i in reversed(range(self.topLayout.count())): 
                self.topLayout.itemAt(i).widget().setParent(None)

            self.topLayout.addWidget(self.getImageButton10, 0, 0)
            self.topLayout.addWidget(self.getImageButton1, 0, 1)
            self.topLayout.addWidget(self.getImageButton2, 0, 2)            
        
        if self.comboBox1.currentIndex() == 2:
            for i in reversed(range(self.topLayout.count())): 
                self.topLayout.itemAt(i).widget().setParent(None)

            # self.topLayout.addWidget(self.comboBox1, 0, 0)
            self.topLayout.addWidget(self.getImageButton10, 0, 0)
            # self.topLayout.addWidget(self.getImageButton0, 0,1)
            self.topLayout.addWidget(self.getImageButton1, 0,1)            
            # self.topLayout.addWidget(self.getImageButton8, 0,3)
            self.topLayout.addWidget(self.comboBox2, 0, 2)

        if self.comboBox1.currentIndex() == 3:

            self.getImageButton13 = QtWidgets.QPushButton('Measured Image')
            self.getImageButton14 = QtWidgets.QPushButton('Simulated Image')            

            self.getImageButton15 = QtWidgets.QPushButton('Adjust image')
            self.getImageButton16 = QtWidgets.QPushButton('Crop Img')
            self.getImageButton17 = QtWidgets.QPushButton('Crop Img2')
            self.getImageButton18 = QtWidgets.QPushButton('Make crop')
            self.getImageButton19 = QtWidgets.QPushButton('Make crop2')

            self.getImageButton16.setCheckable(True)
            self.getImageButton16.setChecked(False)
            self.getImageButton17.setCheckable(True)
            self.getImageButton17.setChecked(False)

            self.getImageButton13.setFixedSize(120,40)
            self.getImageButton14.setFixedSize(120,40)
            self.getImageButton15.setFixedSize(120,40)
            self.getImageButton16.setFixedSize(120,40)
            self.getImageButton17.setFixedSize(120,40)
            self.getImageButton18.setFixedSize(120,40)
            self.getImageButton19.setFixedSize(120,40)

            self.getImageButton13.clicked.connect(self.load_Mimage)
            self.getImageButton14.clicked.connect(self.load_Simage)
            self.getImageButton15.clicked.connect(self.adjust_img)
            self.getImageButton16.clicked.connect(self.crop_img1)
            self.getImageButton17.clicked.connect(self.crop_img2)
            self.getImageButton18.clicked.connect(self.make_crop1)
            self.getImageButton19.clicked.connect(self.make_crop2)

            for i in reversed(range(self.topLayout.count())): 
                self.topLayout.itemAt(i).widget().setParent(None)
            # self.topLayout.addWidget(self.comboBox1, 0, 0)
            self.topLayout.addWidget(self.getImageButton13, 0, 0)
            self.topLayout.addWidget(self.getImageButton14, 0, 1)

        if self.comboBox1.currentIndex() == 4:

            self.getImageButton13 = QtWidgets.QPushButton('Measured Image')
            self.getImageButton14 = QtWidgets.QPushButton('Simulated Image')            

            self.getImageButton15 = QtWidgets.QPushButton('Adjust image')
            self.getImageButton16 = QtWidgets.QPushButton('Crop Img')
            self.getImageButton17 = QtWidgets.QPushButton('Crop Img2')
            self.getImageButton18 = QtWidgets.QPushButton('Make crop')
            self.getImageButton19 = QtWidgets.QPushButton('Make crop2')

            self.getImageButton16.setCheckable(True)
            self.getImageButton16.setChecked(False)
            self.getImageButton17.setCheckable(True)
            self.getImageButton17.setChecked(False)

            self.getImageButton13.setFixedSize(120,40)
            self.getImageButton14.setFixedSize(120,40)
            self.getImageButton15.setFixedSize(120,40)
            self.getImageButton16.setFixedSize(120,40)
            self.getImageButton17.setFixedSize(120,40)
            self.getImageButton18.setFixedSize(120,40)
            self.getImageButton19.setFixedSize(120,40)

            self.getImageButton13.clicked.connect(self.load_Mimage)
            self.getImageButton14.clicked.connect(self.load_Simage)
            self.getImageButton15.clicked.connect(self.adjust_img)
            self.getImageButton16.clicked.connect(self.crop_img1)
            self.getImageButton17.clicked.connect(self.crop_img2)
            self.getImageButton18.clicked.connect(self.make_crop1)
            self.getImageButton19.clicked.connect(self.make_crop2)

            for i in reversed(range(self.topLayout.count())): 
                self.topLayout.itemAt(i).widget().setParent(None)
            # self.topLayout.addWidget(self.comboBox1, 0, 0)
            self.topLayout.addWidget(self.getImageButton13, 0, 0)
            self.topLayout.addWidget(self.getImageButton14, 0, 1)

        if self.comboBox1.currentIndex() == 5:
            print("Mode 5")
            self.load_txt_x = QtWidgets.QPushButton('Load X.npy')
            self.load_txt_y = QtWidgets.QPushButton('Load Y.npy')
            self.load_txt_z = QtWidgets.QPushButton('Load Z.npy')
            self.path_out = QtWidgets.QPushButton('Path out')
            self.build_image = QtWidgets.QPushButton('Build txt Speos')
            ####    SPEOS TXT ####

            # self.load_txt_speos = QtWidgets.QPushButton('Load Speos txt')
            # self.path_out_speos = QtWidgets.QPushButton('Path out Speos')
            # self.build_image_speos = QtWidgets.QPushButton('Build Img Speos')


            self.load_txt_x.setFixedSize(120,40)            
            self.load_txt_y.setFixedSize(120,40)
            self.load_txt_z.setFixedSize(120,40)            
            self.path_out.setFixedSize(120,40)
            self.build_image.setFixedSize(120,40)
            # self.load_txt_speos.setFixedSize(120,40)
            # self.path_out_speos.setFixedSize(120,40)
            # self.build_image_speos.setFixedSize(120,40)            

            self.load_txt_x.clicked.connect(self.LoadNpy_x)
            self.load_txt_y.clicked.connect(self.LoadNpy_y)
            self.load_txt_z.clicked.connect(self.LoadNpy_z)            
            self.path_out.clicked.connect(self.PathOutSpeosTxt)
            self.build_image.clicked.connect(self.SpeosTxtFormat)

            # self.load_txt_speos.clicked.connect(self.LoadTxtSpeos)
            # self.path_out_speos.clicked.connect(self.PathOutSpeos)
            # self.build_image_speos.clicked.connect(self.BuildImageFromSpeos)
            for i in reversed(range(self.topLayout.count())): 
                self.topLayout.itemAt(i).widget().setParent(None)
            # self.topLayout.addWidget(self.comboBox1, 0, 0)
            self.topLayout.addWidget(self.load_txt_x, 0, 0)
            self.topLayout.addWidget(self.load_txt_y, 0, 1)
            self.topLayout.addWidget(self.load_txt_z, 0, 2)            
            self.topLayout.addWidget(self.path_out, 0, 3)
            self.topLayout.addWidget(self.build_image, 0, 4)
            # self.topLayout.addWidget(self.load_txt_speos, 0, 5)
            # self.topLayout.addWidget(self.path_out_speos, 0, 6)
            # self.topLayout.addWidget(self.build_image_speos, 0, 7)



    def load_image1(self):
        self.filename_ref = QFileDialog.getOpenFileName(filter="xyz (*.*)")[0]        
        if self.comboBox1.currentIndex() == 0:            

            self.ref = Label(self.filename_ref)
            self.nextLayout.addWidget(self.ref, 0, 0) 

        if self.comboBox1.currentIndex() == 1:
            # for i in reversed(range(self.nextLayout.count())): 
            #     self.nextLayout.itemAt(i).widget().setParent(None)
            
            get_image_set(self.filename_ref, self.brightness_level,
             self.comboBox2.currentIndex(), self.max_val, self.min_val)

            self.topLayout.addWidget(self.getImageButton3, 0,4)
            self.ref = QtGui.QPixmap(self.filename_ref)
            self.ref = self.ref.scaledToHeight(350)
            self.label = QtWidgets.QLabel()
            self.label.setPixmap(self.ref)
            self.nextLayout.addWidget(self.label,0,0,4,6)
            ####################################################################
            self.nextLayout.addWidget(self.verticalSlider,0,7,4,1)
            #####################################################################
            self.adj = QtGui.QPixmap("images/A2.png")
            print ("Image2")
            self.adj = self.adj.scaledToHeight(350)
            self.label2 = QtWidgets.QLabel()
            self.label2.setPixmap(self.adj)
            self.nextLayout.addWidget(self.label2,0,8,4,6)
            #######################################################################
            self.bar = QtGui.QPixmap("images/A1_bar.png")
            print ("Load bar")
            self.bar = self.bar.scaledToHeight(350)
            self.label3 = QtWidgets.QLabel()
            self.label3.setPixmap(self.bar)
            self.nextLayout.addWidget(self.label3,0,16,4,4)
            self.spinboxeside = QGridLayout()
            self.spinboxeside.addWidget(self.spinBox1,0,0,1,1)
            self.spinboxeside.setRowStretch(4,4)
            self.spinboxeside.addWidget(self.spinBox2,8,0,1,1)
            self.nextLayout.addLayout(self.spinboxeside,0,21,4,1)

    def load_image2(self):
        self.filename_xyz = QFileDialog.getOpenFileName(filter="rgb (*.*)")[0]
        self.xyz = Label(self.filename_xyz)
        self.topLayout.addWidget(self.getImageButton5, 0, 4)
        self.topLayout.addWidget(self.getImageButton4, 0, 5)      
        # self.topLayout.addWidget(self.getImageButton10, 0, 5)

        self.nextLayout.addWidget(self.xyz, 0, 1)

        if self.comboBox1.currentIndex() == 1:            
            self.topLayout.addWidget(self.getImageButton1, 0,2)
            self.topLayout.addWidget(self.getImageButton3, 0,3)
            self.topLayout.addWidget(self.comboBox2, 0,4)      
            
    def upload_brightness(self):
        get_image_set(self.filename_ref, self.brightness_level,
         self.comboBox2.currentIndex(), self.max_val, self.min_val)

        for i in reversed(range(self.nextLayout.count())):
            try:
                self.nextLayout.itemAt(i).widget().setParent(None)
            except:
                print ("None")

        self.ref = QtGui.QPixmap("images/A1.png")
        self.ref = self.ref.scaledToHeight(350)
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(self.ref)
        self.nextLayout.addWidget(self.label, 0, 0, 4, 6)
        ####################################################################
        self.nextLayout.addWidget(self.verticalSlider,0,7,4,1)
        #####################################################################
        self.adj = QtGui.QPixmap("images/A2.png")
        self.adj = self.adj.scaledToHeight(350)
        self.label2 = QtWidgets.QLabel()
        self.label2.setPixmap(self.adj)
        self.nextLayout.addWidget(self.label2, 0, 8, 4, 6)
        #####################################################################
        self.bar = QtGui.QPixmap("images/A1_bar.png")
        self.bar = self.bar.scaledToHeight(350)
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(self.bar)
        self.nextLayout.addWidget(self.label, 0, 16, 4, 4)
        self.nextLayout.addLayout(self.spinboxeside,0,21,4,1)

        valmax = self.max_val
        print (valmax)
        valmin = self.min_val
        print (valmin)      
    
    def color_correction(self):
        print ("    >>>>>>>     Color correction ######")

        xr, yr, xg, yg, xb, yb, fl1 = pitxels_array(self.xyz.pixels, self.ref.pixels)
        print ("Xr >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", xr.shape)
        degree = 2
        if fl1:
            img_corrected = curve_color(self.filename_xyz, xr, yr, xg, yg, xb, yb, degree)
            pixmap_corre = cv2_to_pixmap(img_corrected)

            lim_ = 0.0005
            img_corrected_white = white_balance(img_corrected, lim_)

            img_corrected2 = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB).copy()
            img_corrected_white2 = cv2.cvtColor(img_corrected_white, cv2.COLOR_BGR2RGB).copy()
            
            cv2.imwrite("images/color_correction.png", img_corrected2)
            cv2.imwrite("images/white_balance.png", img_corrected_white2)
            self.path_color = "images/color_correction.png"
            self.path_white = "images/white_balance.png"
        else:
            alert = QMessageBox()
            alert.setText('You have to select the same number of points!')
            alert.exec()
        
        self.luminance = Label2(self.path_white)
        self.luminance.RADIUS = int(self.radius_val.text())
        # self.text = self.luminance.textEdit        
        self.nextLayout2.addWidget(self.luminance,0,0)


        self.side_button.addWidget(self.getImageButton6, 0,1)        
        self.side_button.addWidget(self.getImageButton7, 0,2)
        self.side_button.addWidget(self.radius_val     , 0,3)
        self.side_button.addWidget(self.getImageButton8, 0,4) 
        self.side_button.addWidget(self.getImageButton11,0,5)
        self.side_button.addWidget(self.getImageButton10,0,6)
        self.side_button.addWidget(self.getImageButton12,0,7)
        self.side_button.addWidget(self.kernel_size,     0,8)        
        self.show()

    def file_save(self):
        name = QFileDialog.getSaveFileName(filter="csv (*.csv*)")
        if name[0] !='':
            print ("name file", name)
            file = open(name[0],'w')

            if self.comboBox1.currentIndex() == 0:
                self.text = self.luminance.textEdit 

            if self.comboBox1.currentIndex() == 2:           
                self.text = self.xyz_correct.textEdit

            file.write(self.text)
            file.close()
        else:   
            print ("Path empty")

    def on_button_clicked():
        alert = QMessageBox()
        alert.setText('You clicked the button!')
        alert.exec()    

    def upload_radius(self):
        self.luminance.RADIUS = int(self.radius_val.text())

    def clear_points(self):

        if self.comboBox1.currentIndex() == 0:

            for i in reversed(range(self.nextLayout.count())): 
                try:
                    self.nextLayout.itemAt(i).widget().setParent(None)
                except Exception as e: print(e)

            self.ref = Label(self.filename_ref)
            self.nextLayout.addWidget(self.ref)
            self.xyz = Label(self.filename_xyz)
            self.nextLayout.addWidget(self.xyz)

        if self.comboBox1.currentIndex() == 2:
            for i in reversed(range(self.nextLayout.count())): 
                try:
                    self.nextLayout.itemAt(i).widget().setParent(None)
                except Exception as e: print(e)
            print('Cleaning in table 3')
            self.ref = Label3(self.filename_meas, fl_crop = False)
            self.ref.flag_sqr = False
            self.ref.make_crop = False
            self.text = self.ref.textEdit    
            self.nextLayout.addWidget(self.ref,0,0,4,6)
            self.topLayout.addWidget(self.getImageButton16, 0, 1)
            self.topLayout.addWidget(self.getImageButton18, 0, 2)
            self.topLayout.addWidget(self.getImageButton20, 0, 3)
            self.topLayout.addWidget(self.getImageButton14, 0, 4)
            #make crop 2
            self.sim = Label3(self.filename_sim, fl_crop = False)        
            self.p1x, self.p1y, self.p2x, self.p2y = self.sim.p1x, self.sim.p1y, self.sim.p2x, self.sim.p2y
            self.sim.flag_sqr = False
            self.sim.make_crop = False
            self.text = self.sim.textEdit
            # self.p1x, self.p1y, self.p2x, self.p2y = self.sim.p1x, self.sim.p1y, self.sim.p2x, self.sim.p2y
            # self.nextLayout.addWidget(self.sim, 0, 1)
            self.nextLayout.addWidget(self.sim,0,8,4,6)
            self.topLayout.addWidget(self.getImageButton17, 0, 5)
            self.topLayout.addWidget(self.getImageButton19, 0, 6)
            self.topLayout.addWidget(self.getImageButton21, 0, 7)
            self.topLayout.addWidget(self.getImageButton15, 0, 8)

    def change_brightness(self, value):
        
        self.brightness_level = value

    def change_max(self, value):
        
        self.max_val = value

    def change_min(self, value):
        
        self.min_val = value

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = Window()

    window.show()

    sys.exit(app.exec_())

