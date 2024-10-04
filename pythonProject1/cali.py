import numpy as np
import cv2 as cv
import glob

from numpy.ma.core import resize

# termination criteria for the iterative algorithm to stop
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# Create a grid of points in 3D space
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane.

# Use glob to get the list of all jpg, png, jpeg images in the directory
images = glob.glob('imgs/*.jpg')

for fname in images:
    # Read the image
    img = cv.imread(fname)
    # Reduce the size of the image
    # img = cv.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners in the grayscale image
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points and image points (after refining them)
    if ret:
        objpoints.append(objp)

        # Refine the corner locations
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners on the original image
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        d = cv.resize(img, (720, 1080))
        cv.imshow('img', d)
        # cv.waitKey(1000)qq
        if cv.waitKey() & 0xFF == ord('q'):
            continue
# Calibrate the camera and get the intrinsic parameters
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the intrinsic camera matrix
print("Intrinsic Camera Matrix:\n", mtx)

# Extract and print the focal lengths
fx = mtx[0, 0]  # Focal length in the x direction
fy = mtx[1, 1]  # Focal length in the y direction
print(f"\nFocal length in x direction (fx): {fx}")
print(f"Focal length in y direction (fy): {fy}")

# Print the distortion coefficients
print("\nDistortion Coefficients:\n", dist)

# Print the rotation vectors
print("\nRotation Vectors:\n", rvecs)

# Print the translation vectors
print("\nTranslation Vectors:\n", tvecs)
cv_file = cv.FileStorage('calibration.yaml', cv.FILE_STORAGE_WRITE)
cv_file.write('K',mtx)
cv_file.write('D',dist)
# cv_file.write('R',rvecs)
# cv_file.write('T',tvecs)
cv_file.release()
# Calculate mean reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("\nMean Reprojection Error:", mean_error / len(objpoints))

# Close all OpenCV windows
cv.destroyAllWindows()
