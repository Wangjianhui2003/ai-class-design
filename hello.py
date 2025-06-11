import cv2

image_path = "./dataset/images/0000001_02999_d_0000005.jpg"
label_path = "./dataset/annotations/0000001_02999_d_0000005.txt"

img = cv2.imread(image_path)

with open(label_path, "r") as f:
    for line in f:
        x, y, w, h, score, category, truncation, occlusion = map(int, line.strip().split(','))
        if category in range(10):  # 只处理真实类别
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"ID:{category}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("Image", img)
cv2.waitKey(0)