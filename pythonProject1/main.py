import cv2
import numpy as np
from matplotlib import pyplot as plt

# Kết nối với camera của hai điện thoại
cap1 = cv2.VideoCapture("http://172.172.1.155:4747/video")  # Camera 1
cap2 = cv2.VideoCapture("http://172.172.26.35:4747/video")  # Camera 2

# Hàm để chụp ảnh từ camera
def capture_image(cap):
    ret, frame = cap.read()
    if not ret:
        print("Không thể chụp ảnh từ camera.")
        return None
    return frame

while True:
    # Hiển thị khung hình từ camera
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Không thể lấy khung hình từ camera.")
        break
    
    # Chỉnh kích thước khung hình thành 720x480
    frame1 = cv2.resize(frame1, (720, 480))
    frame2 = cv2.resize(frame2, (720, 480))
    
    # Xoay đứng khung hình
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
    
    # Hiển thị khung hình xếp kề nhau
    combined_frame = np.hstack((frame1, frame2))
    cv2.imshow('Camera 1 và Camera 2', combined_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        img1_color = capture_image(cap1)
        img2_color = capture_image(cap2)
        break
    elif key == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Kiểm tra xem ảnh có được chụp thành công không
if img1_color is None or img2_color is None:
    print("Không thể chụp ảnh từ camera. Vui lòng thử lại.")
    exit()

# Xoay lại ảnh chụp để khớp với chiều ảnh khi được xử lý
img1_color = cv2.rotate(img1_color, cv2.ROTATE_90_CLOCKWISE)
img2_color = cv2.rotate(img2_color, cv2.ROTATE_90_CLOCKWISE)

# Chuyển ảnh sang dạng xám để tính toán SIFT (SIFT chỉ làm việc với ảnh xám)
img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# Khởi tạo SIFT
sift = cv2.SIFT_create()

# Tìm điểm đặc trưng và mô tả với SIFT trên ảnh xám
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# Sử dụng FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)  # Kiểm tra tối đa 50 lần

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lọc các điểm ghép tốt bằng tỷ lệ Lowe's Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Nếu số lượng điểm khớp đủ nhiều, thực hiện homography với RANSAC
if len(good_matches) > 10:
    # Lấy tọa độ của các điểm khớp tốt nhất
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Tính homography sử dụng RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Lọc ra các điểm khớp đúng dựa trên RANSAC
    ransac_good_matches = [good_matches[i] for i in range(len(matchesMask)) if matchesMask[i]]

    # Chỉ lấy khoảng 10 điểm khớp tốt nhất
    ransac_good_matches = sorted(ransac_good_matches, key=lambda x: x.distance)[:20]

    # Vẽ các điểm khớp tốt nhất trên ảnh màu với đường dày hơn và màu sắc rõ ràng
    img_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, ransac_good_matches, None,
                                  matchColor=(0, 255, 0),  # Màu xanh lá cho các đường khớp
                                  singlePointColor=None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Vẽ các đường với độ dày lớn hơn
    for match in ransac_good_matches:
        pt1 = np.int32(kp1[match.queryIdx].pt)
        pt2 = np.int32(kp2[match.trainIdx].pt) + np.array([img1_color.shape[1], 0])

        # Vẽ đường kết nối
        cv2.line(img_matches, pt1, pt2, (0, 255, 0), thickness=3)  # Độ dày tăng lên 3

        # Vẽ chấm tròn tại các điểm đặc trưng
        cv2.circle(img_matches, pt1, 10, (255, 0, 0), thickness=-1)  # Chấm tròn màu đỏ
        cv2.circle(img_matches, pt2, 10, (255, 0, 0), thickness=-1)  # Chấm tròn màu đỏ

    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Khoảng 10 điểm khớp tốt nhất với chấm tròn")
    plt.show()
else:
    print("Không đủ điểm khớp tốt để thực hiện ghép ảnh.")
