# -*- coding: utf-8 -*-

 
# 1.导入库
import cv2 as cv
 
# 2.加载人脸模型
faceModel = cv.CascadeClassifier('./haarcascade_frontalface_alt.xml')
 
# 3.打开摄像头
capture = cv.VideoCapture(0)
 
# 4.获取摄像头的实时画面
while True:
    #4.1读取每一帧的画面
    ret,image = capture.read()
 
    #4.2灰度处理
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
 
    #4.3检查人脸
    # faces = faceModel.detectMultiScale(gray,1.1,3,0)
    faces = faceModel.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print("发现{0}个人脸!".format(len(faces)))
 
    #4.4标记人脸
    for(x,y,w,h) in faces:
        # 1.原始图片 2.人脸坐标原点 3.标记的高度 4，线的颜色 5，线宽
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
 
    #4.5显示图片
    cv.imshow('人脸识别摄像头',image)
 
    #4.6暂停窗口
    if cv.waitKey(5) & 0xFF == ord('q'):
        break
 
# 5.释放资源
capture.release()
 
# 6.销毁窗口
cv.destroyAllWindows()

