import cv2
import glob
import time
import sys
from datetime import datetime


cap = cv2.VideoCapture(0)

cascade_path = '/Users/kamimura/opt/anaconda3/envs/pose/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)

dir = "./images/"

num = 300
label = str(input("人を判別するを半角英数3文字で入力してください"))
file_number = len(glob.glob("./images/")) #現在のフォルダ内のファイル数
count = 0

#ラベルの文字数を確認
if not len(label) == 3:
    print("半角英数3文字で入力してください")
    sys.exit()

while True:
    if count < num:
        time.sleep(0.01)
        print(num)
        print(count)
        print("あと{0}枚です。".format(num-count))

        now = datetime.now()
        r, img = cap.read()

        #結果を保存するための変数
        img_result = img

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100))

        if len(faces) > 0:
            for face in faces:
                x = face[0]
                y = face[1]
                width = face[2]
                height = face[3]

                #50x50の大きさにリサイズ
                roi = cv2.resize(img[y:y+height, x:x+width],(50,50), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(dir+label+"__"+str(now)+'.jpg', roi)

        count = len(glob.glob("./images/")) - file_number
    else:
        break

#カメラをOFFにする
cap.release()


