import sys
import cv2
from cv2 import Stitcher
import numpy as np
import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout, QFileDialog
from imageProcessingDlg import Ui_Dialog

class ImageFusion(QMainWindow, Ui_Dialog):
    def __init__(self):
        super(ImageFusion, self).__init__()
        self.setupUi(self)

    def graytograyimg(img):#灰色图像转灰度图
        grayimg=img*1
        weight=img.shape[0]
        height=img.shape[1]
        for i in range(weight):
            for j in range(height):
                grayimg[i,j]=0.299*img[i,j,0]+0.587*img[i,j,1]+0.114*img[i,j,2]
        return(grayimg)

    def graytoHSgry(grayimg,HSVimg):#将灰度图替换HSV中的V
        H,S,V=cv2.split(HSVimg)
        rows,cols=V.shape
        for i in range(rows):
            for j in range(cols):
                V[i,j]=grayimg[i][j][0]
        newimg=cv2.merge([H,S,V])
        newimg=np.uint8(newimg)
        return newimg

    def Fusion(self):
        path1=self.lineEdit1_1.text()
        path2=self.lineEdit1_2.text()
        path3=self.lineEdit1_3.text()
        img = cv2.imread(path1)
        gray = cv2.imread(path2)
        grayimg = ImageFusion.graytograyimg(gray)
        HSVimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        HSgray = ImageFusion.graytoHSgry(grayimg, HSVimg)
        RGBimg = cv2.cvtColor(HSgray, cv2.COLOR_HSV2BGR)
        cv2.imshow("image", img)
        cv2.imshow("Grayimage", grayimg)
        cv2.imshow("RGBimage", RGBimg)
        cv2.imwrite(path3, RGBimg)
        # 利用qlabel显示结果图片
        png = QtGui.QPixmap(path3).scaled(self.label1_img.width(), self.label1_img.height())
        self.label1_img.setPixmap(png)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        QMessageBox.information(self,"提示", "处理已完成")

    def openimage1_1(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self,"打开彩色图片",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit1_1.setText(imgName)
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label1_1.width(), self.label1_1.height())
        self.label1_1.setPixmap(png)

    def openimage1_2(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self,"打开灰度图片",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit1_2.setText(imgName)
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label1_2.width(), self.label1_2.height())
        self.label1_2.setPixmap(png)

    def saveimage1(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getSaveFileName(self,"输出融合图像",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit1_3.setText(imgName)

    def openimage2_1(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self,"打开图片",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit2_1.setText(imgName)
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label2_1.width(), self.label2_1.height())
        self.label2_1.setPixmap(png)

    def openimage2_2(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self,"打开图片",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit2_2.setText(imgName)
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label2_2.width(), self.label2_2.height())
        self.label2_2.setPixmap(png)

    def saveimage2(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getSaveFileName(self,"输出镶嵌图像",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit2_3.setText(imgName)

    def Mosaic(self):
        path1 = self.lineEdit2_1.text()
        path2 = self.lineEdit2_2.text()
        path3 = self.lineEdit2_3.text()
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        (_result, pano) = stitcher.stitch((img1, img2))
        cv2.imshow('pano', pano)
        cv2.imwrite(path3, pano)
        # 利用qlabel显示结果图片
        png = QtGui.QPixmap(path3).scaled(self.label2_img.width(), self.label2_img.height())
        self.label2_img.setPixmap(png)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        QMessageBox.information(self, "提示", "处理已完成")

    def openimage3(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self,"打开图片",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit3_1.setText(imgName)
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label3_1.width(), self.label3_1.height())
        self.label3_1.setPixmap(png)

    def saveimage3(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getSaveFileName(self,"输出边缘检测图像",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit3_2.setText(imgName)

    def Canny(self):
        path = self.lineEdit3_1.text()
        path2 = self.lineEdit3_2.text()
        img = cv2.imread(path)
        def CannyThreshold():
            lowThreshold = self.horizontalSlider.value()
            img_edges = cv2.GaussianBlur(img_gray, (3, 3), 0)
            img_edges = cv2.Canny(img_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
            cv2.imwrite(path2, img_edges)
            # 利用qlabel显示结果图片
            png = QtGui.QPixmap(path2).scaled(self.label3_2.width(), self.label3_2.height())
            self.label3_2.setPixmap(png)
        x, y = img.shape[0:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ratio = 3
        kernel_size = 3
        CannyThreshold()

    def openimage4(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self,"打开图片",""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit4_1.setText(imgName)
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label4_1.width(), self.label4_1.height())
        self.label4_1.setPixmap(png)

    def saveimage4(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getSaveFileName(self, "输出阴影检测图像", ""," *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        self.lineEdit4_2.setText(imgName)

    def Getshadow(self):
        path = self.lineEdit4_1.text()
        path2 = self.lineEdit4_2.text()
        img = cv2.imread(path)
        b, g, r = np.double(cv2.split(img))
        B = b.ravel()
        G = g.ravel()
        R = r.ravel()
        X = np.vstack((B, G, R))
        Z = np.cov(X)
        eigen_vals, eigen_vecs = np.linalg.eig(Z)
        kl = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for n in range(0, img.shape[0] * img.shape[1]):
            i = n % img.shape[0]
            j = int(n / img.shape[0])
            kl[i, j] = eigen_vecs[0].transpose().dot(np.double(img[i, j, :]))
        for n in range(0, img.shape[0] * img.shape[1]):
            i = n % img.shape[0]
            j = int(n / img.shape[0])
            if kl[i, j] > 160:
                kl[i, j] = 255
            else:
                kl[i, j] = 0

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.destroyAllWindows()
        shadow_ratio = (4 / np.pi) * np.arctan2((b - g), (b + g))  # mutiply 4/pi is to ensure value[0,1]
        shadow_mask = shadow_ratio > 0
        ndvi = np.uint8((np.array(shadow_mask) == False) * 255)

        shadow = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for n in range(0, img.shape[0] * img.shape[1]):
            i = n % img.shape[0]
            j = int(n / img.shape[0])
            if kl[i, j] == 0 and ndvi[i, j] == 0 and img_gray[i,j]<128:
                shadow[i, j] = 0
            else:
                shadow[i, j] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel[1, 0] = 0
        kernel[3, 0] = 0
        kernel[1, 4] = 0
        kernel[3, 4] = 0
        f = cv2.erode(shadow, kernel)
        f = cv2.erode(f, kernel)
        f = cv2.dilate(f, kernel)
        f = cv2.dilate(f, kernel)
        for n in range(0, img.shape[0] * img.shape[1]):
            i = n % img.shape[0]
            j = int(n / img.shape[0])
            if f[i, j] == 0:
                img[i, j] = [0, 0, 255]
        cv2.imwrite(path2, img)
        # 利用qlabel显示结果图片
        png = QtGui.QPixmap(path2).scaled(self.label4_2.width(), self.label4_2.height())
        self.label4_2.setPixmap(png)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  myWin = ImageFusion()
  myWin.show()
  sys.exit(app.exec_())


