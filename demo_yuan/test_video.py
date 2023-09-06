import cv2

# 打开视频文件
#path = "smoking.avi"
cap = cv2.VideoCapture("smoke.mp4")

# 读取视频直到结束
while cap.isOpened():
    print("ok")
    # 读取视频的下一帧
    ret, frame = cap.read()
    if ret:
        # 对图片做处理
        frame_processed = process_frame(frame)
        # 显示处理后的图片
        cv2.imshow('frame processed', frame_processed)
    else:
        break

    # 等待用户按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

