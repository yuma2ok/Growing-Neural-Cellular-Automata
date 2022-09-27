import os
import cv2
import face_recognition

def save_face_located_image(filepath, face_locations, prefix):
    # 保存用の画像データを準備
    save_img = cv2.imread(filepath)
    
    # 検出した範囲を保存用画像に短形表示
    for face_location in face_locations:
        y1, x2, y2, x1 = face_location
        cv2.rectangle(save_img, (x1, y1), (x2, y2), color=(0,0,255), thickness=4)
        
    # 検出領域を示した画像を保存
    filename = os.path.basename(filepath)
    filedir = os.path.dirname(filepath)
    save_filename = prefix + '_' + filename
    save_imgpath = os.path.join(filedir, save_filename)
    cv2.imwrite(save_imgpath, save_img)

# 顔検出する画像のファイルリスト
filelist = ['img.jpg', 'img3.png']

imgdir = os.path.join('./img')
filepaths = [os.path.join(imgdir, file) for file in filelist]

# HOG特徴量ベースの顔検出
for filepath in filepaths:
    # 画像読み込み
    image = face_recognition.load_image_file(filepath)
    
    # HOGベースモデルの顔検出実行
    face_locations = face_recognition.face_locations(image)
    
    # 検出した領域を表示
    print("------------------------------")
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    for face_location in face_locations:
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        
    # 顔検出した領域を短形表示した画像を保存
    save_face_located_image(filepath, face_locations, 'cnn')