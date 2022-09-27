import os
import cv2
import dlib

# 顔検出器を取得
detector = dlib.get_frontal_face_detector()

# 読み込むファイルのリスト
filelist = ['img.jpg', 'img3.png']

for file in filelist:
    
    print("------------------------------")
    print("Processing file: {}".format(file))
    
    # 画像を読み込み(画像はホーム下のimgディレクトリに置く想定)
    imgdir = os.path.join('./img')
    imgpath = os.path.join(imgdir, file)
    img = dlib.load_rgb_image(imgpath)
    
    # 顔検出(第二引数はアップサンプリング(拡大)の回数)
    dets, scores, idx = detector.run(img, 1, -0.5)
    print("Number of faces detected: {}".format(len(dets)))
    
    # 検出した範囲を表示
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} score: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom(), scores[i]))
    
    # 検出した範囲を短形表示し、別画像として保存
    save_img = cv2.imread(imgpath)
    
    # 検出した範囲を保存用画像に短形表示
    for d in dets:
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        cv2.rectangle(save_img, (x1, y1), (x2, y2), color=(0,0,255), thickness=4)
    
    # 別名ファイルで保存
    save_file = 'dlib_' + file
    save_imgpath = os.path.join(imgdir, save_file)
    cv2.imwrite(save_imgpath, save_img)