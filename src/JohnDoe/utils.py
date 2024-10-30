import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
def display_mat(images, title="Title_here"):
    if isinstance(images, list):
        num_images = len(images)
        plt.figure()
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                plt.subplot(1, num_images, i + 1)  # 1 row, num_images columns
                if len(img.shape) == 2:  # Grayscale
                    plt.imshow(img, cmap='gray')
                else:  # RGB or BGR
                    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for proper color display
                plt.axis('off')
                plt.title(f"{title}_{i}")
            else:
                print(f"Item {i} is not an image.")
    else:
        # Single image case
        plt.imshow(cv.cvtColor(images, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        
    plt.tight_layout()
    plt.show()

def rescale(imgs, scale=0.5):
    h, w = imgs.shape[:2]  # Get the height and width of the image
    new_size = (int(w * scale), int(h * scale))  # Calculate new size (width, height)
    return cv.resize(imgs, new_size)  # Resize the image

def display_cv(listoo, waitkey=0, title="Title_here"):
    if isinstance(listoo, np.ndarray):
        cv.imshow(title, listoo)
    elif isinstance(listoo, list):
        for i, img in enumerate(listoo):
            if isinstance(img, np.ndarray):  
                cv.imshow(f"{title}_{i}", img)
            else:
                print(f"Item {i} is not an image.")
    else:
        print("Input is neither an image nor a list of images.")

    key = cv.waitKey(waitkey)
    if key & 0xFF == ord('d'):
        cv.destroyAllWindows()
