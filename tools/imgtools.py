"""Various tools for manipulating images."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re


class Image_info(object):
    """Describes an image by his path, camera and timestamp."""

    def __init__(self, path):
        """Create an image instance."""
        name = os.path.basename(path)
        cam, time = get_cam_n_time(name)
        self.path = path
        self.cam = cam
        self.time = time

    def __repr__(self):
        """Compute 'official' string reprentation."""
        return repr((self.path, self.cam, self.time))

    def time_diff(self, image):
        """Compute the time difference."""
        return abs(self.time - image.time)


def get_cam_n_time(filename):
    """Parse the cam and the time of the video from the filename."""
    split = re.split('_{1,2}|\.', filename, 4)
    cam = int(split[1])
    time = int(split[2]+split[3])
    return cam, time


def display(img, title='image', matplotlib=False):
    """Display images with opencv func or with the help of the matplotlib."""
    if matplotlib:
        fig = plt.figure(figsize=(20, 20))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(title)
        plt.show()
        plt.close(fig)
    else:
        height, width = img.shape[:2]
        height_s = 1000
        width_s = int(width * height_s / height)
        img_s = cv2.resize(img, (width_s, height_s))
        cv2.imshow('im', img_s)
        cv2.waitKey(100)


def find_closest_img(img_list, cam_left, cam_right):
    """Find pairs of images which are close in time."""
    cam_sorted = sorted(img_list, key=lambda img: img.time)
    cam_diff = []
    done = []
    img_pred = None
    time_diff_pred = None
    time_diff_succ = None
    for i in range(len(cam_sorted)):
        img_cur = cam_sorted[i]
        if img_cur.cam == cam_right:
            img_pred = img_cur
        elif img_cur.cam == cam_left:
            if img_pred is not None:
                time_diff_pred = img_cur.time_diff(img_pred)
            #finding right image successor
            for j in range(i+1, len(cam_sorted)):
                img_succ = cam_sorted[j]
                if img_succ.cam == cam_right:
                    time_diff_succ = img_cur.time_diff(img_succ)
                    break
            if time_diff_pred is not None and time_diff_succ is not None:
                if time_diff_pred > time_diff_succ:
                    cam_diff.append((
                        img_cur.path,
                        img_succ.path,
                        time_diff_succ))
                else:
                    cam_diff.append((
                        img_cur.path,
                        img_pred.path,
                        time_diff_pred))
            elif time_diff_pred is not None:
                cam_diff.append((
                    img_cur.path,
                    img_pred.path,
                    time_diff_pred))
            elif time_diff_succ is not None:
                cam_diff.append((
                    img_cur.path,
                    img_succ.path,
                    time_diff_succ))
    return sorted(cam_diff, key=lambda time: time[2])


def rotate_image(img, angle):
    """Rotate img around center point by given angle.

    The returned img will be large enough to hold the entire
    new img, with a black background.
    """
    # Get img size
    size = (img.shape[1], img.shape[0])
    center = tuple(np.array(size) / 2)
    (width_half, height_half) = center

    # Convert the 3x2 rotation matrix to 3x3
    map_matrix = np.vstack([cv2.getRotationMatrix2D(center, angle, 1.0),
                            [0, 0, 1]])

    # To get just the rotation
    rotation_matrix = np.matrix(map_matrix[:2, :2])

    # Declare the corners of the image in relation to the center
    corners = [
        [-width_half,  height_half],
        [width_half,  height_half],
        [-width_half, -height_half],
        [width_half, -height_half]
    ]

    # get the rotated corners
    corners_rotated = corners * rotation_matrix

    # Find the size of the new img
    x_max = max(corners_rotated[:4, 0])
    x_min = min(corners_rotated[:4, 0])
    y_max = max(corners_rotated[:4, 1])
    y_min = min(corners_rotated[:4, 1])

    # get the new size
    w_rotated = int(abs(x_max - x_min))
    h_rotated = int(abs(y_max - y_min))
    size_new = (w_rotated, h_rotated)

    # matrix to center the rotated image
    translation_matrix = np.matrix([
        [1, 0, int(w_rotated / 2 - width_half)],
        [0, 1, int(h_rotated / 2 - height_half)],
        [0, 0, 1]
    ])

    # get the affine Matrix
    affine_mat = (translation_matrix * map_matrix)[0:2]

    # transformation
    retval = cv2.warpAffine(
        img,
        affine_mat,
        size_new,
        flags=cv2.INTER_LINEAR
    )

    return retval


def rectify_img(img, IntrinsicMatrix, Distortion_Coeff_matlab):
    """Take an image and undistort it."""
    # TODO check size, cmp matl
    h, w = img.shape[:2]
    newCameraMatrix_m, validPixRoi_m = cv2.getOptimalNewCameraMatrix(
        IntrinsicMatrix, Distortion_Coeff_matlab, (w, h), 1, (w, h), 0)

    img_rectified = cv2.undistort(
        img, IntrinsicMatrix, Distortion_Coeff_matlab, None, newCameraMatrix_m)
    return img_rectified
