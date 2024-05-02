import numpy as np
import cv2

def d2n(image):


    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)
    image_depth = image.copy()
    # image_depth = image_depth-np.min(image_depth)
    # image_depth =image_depth/np.max(image_depth)
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    bg_threhold = 0.0

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0
    z = np.ones_like(x) * np.pi * 2.0
    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    return image



if __name__ == '__main__':
    depth_img = cv2.imread('data/Metadata/Images/1.png', -1)

    normal = d2n(depth_img)

    cv2.imwrite("depth.png", depth_img)
    cv2.imwrite("normal.png", normal)