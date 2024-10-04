import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
# from cali import mtx, cv_file

# Calibrate the camera using images of a chessboard

cv_file = cv.FileStorage('calibration.yaml', cv.FILE_STORAGE_READ)
mtx = cv_file.getNode('K').mat()
cv_file.release()
# Stereo disparity calculation with calibration
def stereo_disparity_with_calibration():
    focal_length_matrix = mtx
    focal_length_x = focal_length_matrix[0, 0]
    focal_length_y = focal_length_matrix[1, 1]
    focal_length = (focal_length_x + focal_length_y) / 2
    baseline = 8  # cm

    img1_color = cv.imread('22.jpg')
    img2_color = cv.imread('23.jpg')

    if img1_color is None or img2_color is None:
        print("Không thể đọc được ảnh từ file. Vui lòng thử lại.")
        return

    img1_gray = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 5:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        ransac_good_matches = [good_matches[i] for i in range(len(matchesMask)) if matchesMask[i]]
        ransac_good_matches = sorted(ransac_good_matches, key=lambda x: x.distance)[:20]

        src_pt = np.int32(kp1[ransac_good_matches[0].queryIdx].pt)
        dst_pt = np.int32(kp2[ransac_good_matches[0].trainIdx].pt)

        disparity = abs(src_pt[0] - dst_pt[0])
        print(f"Disparity: {disparity} pixels")

        if disparity > 0:
            z = (baseline * focal_length) / disparity
            print(f"Độ sâu (z): {z} cm")

            # Highlight corresponding points
            cv.circle(img1_color, tuple(src_pt), 5, (0, 255, 0), -1)
            cv.circle(img2_color, tuple(dst_pt), 5, (0, 255, 0), -1)

            # Resize the images for easier viewing
            img1_color = cv.resize(img1_color, (img1_color.shape[1] // 4, img1_color.shape[0] // 4))
            img2_color = cv.resize(img2_color, (img2_color.shape[1] // 4, img2_color.shape[0] // 4))

            # Display the images side by side
            combined_frame_with_points = np.hstack((img1_color, img2_color))
            cv.imshow('correction_point', combined_frame_with_points)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Disparity quá nhỏ, không thể tính độ sâu.")
    else:
        print("Không đủ điểm khớp tốt để tính toán.")


# Call the stereo disparity calculation with calibration
stereo_disparity_with_calibration()
