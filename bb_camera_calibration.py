# Diese CameraCalibration funktioniert mittels einer moeglichkeit des
# downscalings der Bilder. Die Bilder werden Schrittweise verkleinert
# und dann jeweils nach dem Schachbrettmuster gesucht
# Anschließend werden daraus die nötigen Daten für die Calibrierung errechnet

import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import shelve


# helper function for using with jupyter set type = 1
def _display_image(img, type=0):
    if type == 0:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 800, 600)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        plt.figure(figsize=(30, 20))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('img')
        plt.show()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

BOARDSIZE_WIDTH = 9
BOARDSIZE_HEIGHT = 6
CAMERA = '350d'
PATTERN = 'chessboard'
ENABLE_PLT = 1  # '1' enables matplot, '0' shows images with cv2
SCALING = 0  # '0' will try to find chessboard in org img, '1' will scale

IMG_EXCLUDE = ['IMG_9966.JPG']  # to exclude images from the Calibration

# Enables image downscaling
# DOWNSCAL_ENABLE = True
DOWNSCAL_UPPER_BOUND = 11.6
DOWNSCAL_LOWER_BOUND = 4
DOWNSCAL_STEP = 0.1
# IMAGES_PATHS = 'images/741/'
IMAGES_PATHS = os.path.join('images', CAMERA)
SHELVE_NAME = CAMERA + '_' + PATTERN + '.shelve'

# creates  a list of images
img_paths_list = glob.glob(IMAGES_PATHS+'/*.JPG')

# excludes specified images
img_paths_list = [img for img in img_paths_list if os.path.basename(img) not in IMG_EXCLUDE]

# z coordinates are zero, because of planar pattern (like chessboard)
objp = np.zeros((BOARDSIZE_HEIGHT*BOARDSIZE_WIDTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARDSIZE_WIDTH, 0:BOARDSIZE_HEIGHT].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space,
# describes how should the pattern look like
imgpoints = []  # 2d points in image plane for each input image contrains
# coordinates of the important points (corners, circle centers)

img_paths = []  # stores all images, where a chessboard was found

# creates a folder for the images with the drawn chessboardpattern
os.mkdir(os.path.join(IMAGES_PATHS, PATTERN))

# counts the images with found PATTERN
count_find_chessboard = 0


least_upper_bound = DOWNSCAL_UPPER_BOUND
greatest_lower_bound = DOWNSCAL_LOWER_BOUND
for cur_img_path in img_paths_list:
    print(cur_img_path)
    img_org = cv2.imread(cur_img_path)
    img_org_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    if SCALING == 0:
        ret, corners = cv2.findChessboardCorners(
            img_org_gray, (BOARDSIZE_WIDTH, BOARDSIZE_HEIGHT))
    else:
        ret = 0

    if ret:
        img_paths.append(os.path.basename(cur_img_path))
        print('pattern found in original sized image')
        count_find_chessboard += 1

        # improve the scaled up corners
        corners_imp = corners.copy()
        cv2.cornerSubPix(
            image=img_org_gray, corners=corners_imp,
            winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)

        # adding 'desires value' of cornerpoints
        objpoints.append(objp)

        # corresponding image corners
        imgpoints.append(corners_imp)

        # draw in the find corners
        img_org_chess = img_org.copy()
        cv2.drawChessboardCorners(
            img_org_chess, (BOARDSIZE_WIDTH, BOARDSIZE_HEIGHT),
            corners_imp, ret)

        # shows/saves the downscaled image
        _display_image(img_org_chess, ENABLE_PLT)
        cv2.imwrite(os.path.join(IMAGES_PATHS, PATTERN, os.path.basename(
            cur_img_path)), img_org_chess)

    else:
        print('No pattern find in original sized image. Trying to downscale '
              'image')
        for RATIO_DOWNSCALING in np.arange(DOWNSCAL_LOWER_BOUND, DOWNSCAL_UPPER_BOUND, DOWNSCAL_STEP):

            # downscaling of the image by the factor RATIO_DOWNSCALING
            img_downscaled = cv2.resize(
                img_org, (0, 0), fx=1/RATIO_DOWNSCALING,
                fy=1/RATIO_DOWNSCALING, interpolation=cv2.INTER_AREA)

            # convert image to gray colorspace
            img_downscaled_gray = cv2.cvtColor(
                img_downscaled, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners in the downscaled image
            ret, corners_downscaled = cv2.findChessboardCorners(
                img_downscaled_gray, (BOARDSIZE_WIDTH, BOARDSIZE_HEIGHT))

            if ret:
                img_paths.append(os.path.basename(cur_img_path))
                if RATIO_DOWNSCALING < least_upper_bound:
                    least_upper_bound = RATIO_DOWNSCALING
                if RATIO_DOWNSCALING > greatest_lower_bound:
                    greatest_lower_bound = RATIO_DOWNSCALING
                print('pattern found')
                print('RATIO_DOWNSCALING: ' + str(RATIO_DOWNSCALING))
                count_find_chessboard += 1
                # Refines the corner locations in the downscaled image.
                corners_downscaled_impr = corners_downscaled.copy()
                cv2.cornerSubPix(
                    image=img_downscaled_gray, corners=corners_downscaled_impr,
                    winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)

                # Draw and display the corners in the downscaled image
                img_downscaled_chess = img_downscaled.copy()
                cv2.drawChessboardCorners(
                    img_downscaled_chess, (BOARDSIZE_WIDTH, BOARDSIZE_HEIGHT),
                    corners_downscaled_impr, ret)

                #  scale up the positions of the corners to the orignal Size
                # of the image
                corners = corners_downscaled_impr.copy()
                corners = corners*RATIO_DOWNSCALING

                # improve the scaled up corners
                img_org_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
                corners_imp = corners.copy()
                # cv2.cornerSubPix(
                #     image=img_org_gray, corners=corners_imp,
                #     winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)

                # adding 'desires value' of cornerpoints
                objpoints.append(objp)

                # corresponding image corners
                imgpoints.append(corners_imp)

                # draw in the find corners
                img_org_chess = img_org.copy()
                cv2.drawChessboardCorners(
                    img_org_chess, (BOARDSIZE_WIDTH, BOARDSIZE_HEIGHT),
                    corners_imp, ret)

                # shows/saves the downscaled image
                _display_image(img_org_chess, ENABLE_PLT)
                cv2.imwrite(
                    os.path.join(IMAGES_PATHS, PATTERN, os.path.basename(
                        cur_img_path)), img_org_chess)
                break

            else:
                # print(RATIO_DOWNSCALING)
                if abs(RATIO_DOWNSCALING - DOWNSCAL_UPPER_BOUND - DOWNSCAL_STEP) < 1e-10:
                    print('no ChessboardCorners were found in' + cur_img_path)
                    _display_image(img_org, ENABLE_PLT)

cv2.destroyAllWindows()
print(
    'recognized patterns :' + str(count_find_chessboard) + ' of ' +
    str(len(img_paths_list)))
print('kleinster Wert für das Scaling: ' + str(least_upper_bound))
print('größter Wert für das Scaling: ' + str(greatest_lower_bound))

print(img_org_gray.shape)

# Finds the camera intrinsic and extrinsic parameters
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_org_gray.shape[::-1], None, None)

# Storing of the parameters
calibrateCamera_save = shelve.open(
    os.path.join(IMAGES_PATHS, SHELVE_NAME), protocol=0, writeback=True)
calibrateCamera_save['retval'] = retval
calibrateCamera_save['cameraMatrix'] = cameraMatrix
calibrateCamera_save['distCoeffs'] = distCoeffs
calibrateCamera_save['rvecs'] = rvecs
calibrateCamera_save['tvecs'] = tvecs
calibrateCamera_save.sync()
calibrateCamera_save.close


# loads an image, which will be rectified
cur_img = cv2.imread(img_paths_list[0])
h, w = cur_img.shape[:2]
newCameraMatrix, validPixRoi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix, distCoeffs, (w, h), 1, (w, h), 0)

# transform 'cur_img' to compensate for lens distortion
dst = cv2.undistort(cur_img, cameraMatrix, distCoeffs, None, newCameraMatrix)

x, y, w, h = validPixRoi
# dst = dst[y:y+h, x:x+w]
print('calib w = ' + str(w))
print('calib h = ' + str(h))
cv2.imwrite(os.path.join(IMAGES_PATHS, 'calibresult.png'), dst)
_display_image(dst, ENABLE_PLT)

mean_error = 0
error_p = []
for i in range(len(objpoints)):

    # transform the object point to image point
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], newCameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    error_p.append(error)
    mean_error += error

print('total error: ', mean_error/len(objpoints))
print(img_paths)
print(error_p)

x_pos = np.arange(len(img_paths))
plt.axhline(
    mean_error/len(objpoints), linewidth=3, color='r', label='Total Error')
plt.bar(x_pos, error_p, align='center', alpha=0.9)
plt.xticks(x_pos, img_paths, rotation=90)
plt.xlabel('Images')
plt.ylabel('Error')
plt.legend()
plt.show()
