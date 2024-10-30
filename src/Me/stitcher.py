import pdb
import glob
import cv2
import os
import matplotlib.pyplot as plt
from src.JohnDoe.utils import *
from src.JohnDoe.some_folder import folder_func
# from src.
class PanaromaStitcher():
    fov = None
    resize_size = 1200
    Ratio = 0.75
    def __init__(self):
        
        pass
    
    def Convert_xy(self,x, y,f,center):
        xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
        yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
        
        return xt, yt
    def Cylinder_Projection(self, img_in):
        h, w = img_in.shape[:2]
        center = [w // 2, h // 2]
        f = w / (2 * np.tan(np.radians(self.fov) / 2))

        img_out = np.zeros(img_in.shape, dtype=np.uint8)
        coords = np.array([np.array([x, y]) for x in range(w) for y in range(h)])
        x_out = coords[:, 0]
        y_out = coords[:, 1]

        x_in, y_in = self.Convert_xy(x_out, y_out, f, center)
        x_in_floor = x_in.astype(int)
        y_in_floor = y_in.astype(int)

        valid_mask = (x_in_floor >= 0) & (x_in_floor <= (w - 2)) & \
                    (y_in_floor >= 0) & (y_in_floor <= (h - 2))

        x_out = x_out[valid_mask]
        y_out = y_out[valid_mask]
        x_in = x_in[valid_mask]
        y_in = y_in[valid_mask]
        x_in_floor = x_in_floor[valid_mask]
        y_in_floor = y_in_floor[valid_mask]

        dx = x_in - x_in_floor
        dy = y_in - y_in_floor

        weight_tl = (1.0 - dx) * (1.0 - dy)
        weight_tr = dx * (1.0 - dy)
        weight_bl = (1.0 - dx) * dy
        weight_br = dx * dy

        img_out[y_out, x_out, :] = (weight_tl[:, None] * img_in[y_in_floor, x_in_floor, :]) + \
                                (weight_tr[:, None] * img_in[y_in_floor, x_in_floor + 1, :]) + \
                                (weight_bl[:, None] * img_in[y_in_floor + 1, x_in_floor, :]) + \
                                (weight_br[:, None] * img_in[y_in_floor + 1, x_in_floor + 1, :])

        min_x_out = min(x_out)
        img_out = img_out[:, min_x_out: -min_x_out, :]
        return img_out, x_out - min_x_out, y_out

    def SIFTMatches(self, img_base, img_sec):
        sift = cv2.SIFT_create()
        kp_base, des_base = sift.detectAndCompute(cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY), None)
        kp_sec, des_sec = sift.detectAndCompute(cv2.cvtColor(img_sec, cv2.COLOR_BGR2GRAY), None)

        bf_matcher = cv2.BFMatcher()
        matches_init = bf_matcher.knnMatch(des_base, des_sec, k=2)

        matches_good = []
        for m, n in matches_init:
            if m.distance < self.Ratio * n.distance:
                matches_good.append([m])

        return matches_good, kp_base, kp_sec

    def normalize_points(self,points):
        mean = np.mean(points, axis=0)
        std_dev = np.std(points)
        scale = np.sqrt(2) / std_dev

        T = np.array([
            [scale, 0, -scale * mean[0]],
            [0, scale, -scale * mean[1]],
            [0, 0, 1]
        ])
        
        points = np.dot(T, np.vstack((points.T, np.ones((1, points.shape[0])))))
        return T, points.T

    def compute_H(self,src_pts, dst_pts):
        A = []
        for (x, y), (xp, yp) in zip(src_pts, dst_pts):
            A.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
            A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
        A = np.array(A)
        
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))
        return H / H[2, 2]

    def ransac_H(self,src_pts, dst_pts, threshold=4.0, iterations=1000):
        max_inliers = []
        best_H = None
        
        for _ in range(iterations):
            idx = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[idx]
            dst_sample = dst_pts[idx]
            
            H = self.compute_H(src_sample, dst_sample)
            inliers = []
            for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
                projected = np.dot(H, np.array([src[0], src[1], 1]))
                projected /= projected[2]
                error = np.linalg.norm(np.array([dst[0], dst[1]]) - projected[:2])
                if error < threshold:
                    inliers.append(i)
            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                best_H = H
        if best_H is not None:
            best_H = self.compute_H(src_pts[max_inliers], dst_pts[max_inliers])
        
        return best_H, max_inliers

    def FindHomography(self, matches, kp_base, kp_sec):
        if len(matches) < 4:
            print("\nNot enough matches found between the images.\n")
            exit(0)
            
        pts_base = []
        pts_sec = []
        for match in matches:
            pts_base.append(kp_base[match[0].queryIdx].pt)
            pts_sec.append(kp_sec[match[0].trainIdx].pt)

        pts_base = np.float32(pts_base)
        pts_sec = np.float32(pts_sec)
        
        (H, status) = cv2.findHomography(pts_sec, pts_base, cv2.RANSAC, 4.0)
        # (H, status) = self.ransac_H(pts_sec, pts_base)
        return H, status
       
    def GetNewFrameSizeAndMatrix(self, H, sec_shape, base_shape):
        (h_sec, w_sec) = sec_shape
        initial_pts = np.array([[0, w_sec - 1, w_sec - 1, 0],
                                [0, 0, h_sec - 1, h_sec - 1],
                                [1, 1, 1, 1]])
        final_pts = np.dot(H, initial_pts)
        x, y, c = final_pts
        x = np.divide(x, c)
        y = np.divide(y, c)
        min_x, max_x = int(round(min(x))), int(round(max(x)))
        min_y, max_y = int(round(min(y))), int(round(max(y)))
        
        new_w = max_x
        new_h = max_y
        offset = [0, 0]
        
        if min_x < 0:
            new_w -= min_x
            offset[0] = abs(min_x)
        if min_y < 0:
            new_h -= min_y
            offset[1] = abs(min_y)
            
        if new_w < base_shape[1] + offset[0]:
            new_w = base_shape[1] + offset[0]
        if new_h < base_shape[0] + offset[1]:
            new_h = base_shape[0] + offset[1]
            
        x = np.add(x, offset[0])
        y = np.add(y, offset[1])
        
        src_pts = np.float32([[0, 0],
                            [w_sec - 1, 0],
                            [w_sec - 1, h_sec - 1],
                            [0, h_sec - 1]])
        dst_pts = np.float32(np.array([x, y]).transpose())
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        return [new_h, new_w], offset, H


    def StitchImages(self, base_img, sec_img):
        sec_img_cyl, mask_x, mask_y = self.Cylinder_Projection(sec_img)
        sec_img_mask = np.zeros(sec_img_cyl.shape, dtype=np.uint8)
        sec_img_mask[mask_y, mask_x, :] = 255

        matches, base_kp, sec_kp = self.SIFTMatches(base_img, sec_img_cyl)
        H, status = self.FindHomography(matches, base_kp, sec_kp)
        
        new_size, offset, H = self.GetNewFrameSizeAndMatrix(H, sec_img_cyl.shape[:2], base_img.shape[:2])
        sec_img_warp = cv2.warpPerspective(sec_img_cyl, H, (new_size[1], new_size[0]))
        sec_img_mask_warp = cv2.warpPerspective(sec_img_mask, H, (new_size[1], new_size[0]))

        base_img_transformed = np.zeros((new_size[0], new_size[1], 3), dtype=np.uint8)
        base_img_transformed[offset[1]:offset[1] + base_img.shape[0], offset[0]:offset[0] + base_img.shape[1]] = base_img

        # Create a blending mask for feathering
        blend_mask = cv2.GaussianBlur(sec_img_mask_warp, (51, 51), 0) / 255.0  # Adjust kernel size as needed for feathering
        base_blend_mask = 1.0 - blend_mask

        # Apply the blending mask
        blended_img = (sec_img_warp * blend_mask + base_img_transformed * base_blend_mask).astype(np.uint8)

        return blended_img, H
    def StitchImages_MiddleBase(self, images):
        homography_mats = []

        mid_idx = len(images) // 2
        mid_img = self.Cylinder_Projection(images[mid_idx])[0]
        
        base_img = mid_img.copy()
        for i in range(mid_idx - 1, -1, -1):
            stitched_img, H = self.StitchImages(base_img, images[i]) 
            homography_mats.insert(0, H)
            base_img = stitched_img.copy()

        for i in range(mid_idx + 1, len(images)):  
            stitched_img, H = self.StitchImages(base_img, images[i]) 
            homography_mats.append(H) 
            base_img = stitched_img.copy()

        return base_img, homography_mats

    def get_Images( self,path):
        all_images = sorted(glob.glob(path+os.sep+'*'))
        Images = []
        k = 1100
        for i in all_images:
            img = cv.imread(i)
            img = cv2.resize(img , (self.resize_size , self.resize_size))
            Images.append(img)
        return Images
    
    def make_panaroma_for_images_in(self,path,fov=50,resis = 1200,Ratio = 0.75):
        self.Ratio = Ratio
        self.resize_size = resis
        self.fov = fov
        Images = self.get_Images(path) 
        print('Found {} Images for stitching'.format(len(Images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        self.say_hi()

        stitched_image,homography_matrix_list = self.StitchImages_MiddleBase(Images)  
        
        return stitched_image, homography_matrix_list 

    def say_hi(self):
        print('Hii From John Doe..')