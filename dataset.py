import glob
import cv2.cv2 as cv2
import imutils
import numpy as np


# ear_img_path = './dataset/images/orejas/'
# ear_mask_path = './dataset/mask/orejas/'


def generate_dataset(ear_img_path: str, ear_mask_path: str, augm_size: int = 7):
    """
    Generates ears dataset
    :param ear_img_path: path with ear images
    :param ear_mask_path: path with ear masks
    :param augm_size: number of augmentations to perform
    :return: ndarray of images, ndarray of masks, ndarray of angles
    """
    COLOR_MIN = (0, 0, 50)
    COLOR_MAX = (0, 0, 256)

    ear_mask_paths = glob.glob(ear_mask_path + './**/*label.png', recursive=True)

    X, Y_m, Y_a = [], [], []
    for path_mask in ear_mask_paths:
        path_img = ear_img_path + path_mask.split('/')[6].split('.')[0] + '.jpg'
        image = cv2.imread(path_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_new = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        mask = cv2.imread(path_mask)
        # continue
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.inRange(mask, COLOR_MIN, COLOR_MAX, cv2.THRESH_BINARY)
        mask_new = np.dot(mask, 1 / 255)

        X.append(image_new)
        Y_m.append(mask_new)
        Y_a.append(0)

        aug_images, aug_masks, aug_angles = random_rotate(image_new, mask_new, number_of_rotations=augm_size)

        X = X + aug_images
        Y_m = Y_m + aug_masks
        Y_a = Y_a + aug_angles

    return np.array(X), np.array(Y_m), np.array(Y_a)


def random_rotate(img, mask, number_of_rotations: int = 7, min_angle: int = 0, max_angle: int = 365):
    """
    Generates number_of_rotations images and masks, by rotating originals
    randomly between min_angle and max_angle
    :param img: original image
    :param mask: original mask
    :param number_of_rotations: number of rotations to perform
    :param min_angle: minimum angle
    :param max_angle: maximum angle
    :return: number_of_rotations tuples with rotated image, rotated mask, angle of rotation
    """
    rotated_images, rotated_masks, angles = [], [], []

    rotations = np.random.randint(min_angle, max_angle, number_of_rotations)
    for i, angle in enumerate(rotations):
        rotated_image = imutils.rotate(image=img, angle=angle)
        rotated_mask = imutils.rotate(image=mask, angle=angle)

        rotated_images.append(rotated_image)
        rotated_masks.append(rotated_mask)
        angles.append(angle)

    return rotated_images, rotated_masks, angles
