#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import os
from matplotlib import image
import numpy as np
import cv2
import glob

# Global variable for anms points - used in feature descriptor step
# global anms_pts
# anms_pts = []

# Add any python libraries here

#Function to find local maxima coordinates
def imreginalmax(input, nloxMax = 100, minDist = 10):
    temp_img = input.copy()
    locations = []
    for i in range(nloxMax):
        _, max_val, _, max_loc = cv2.minMaxLoc(temp_img)

        if max_val <= 0:
            break
        locations.append(max_loc)
        cv2.circle(temp_img, max_loc, minDist, 0, -1)

    return locations




def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    # Shakthi 
    # img_path = "/home/alien/YourDirectoryID_p1/Phase1/Data/Train/Set1"
    # out_dir = "/home/alien/YourDirectoryID_p1/Phase1/Outputs"

    # Aditya      
    img_path = "/home/adipat/Documents/Spring 26/CV/P1/Traditional_panaroma/YourDirectoryID_p1/Phase1/Data/Test/P1Ph1TestSet/Phase1/TestSet2"
    out_dir = "/home/adipat/Documents/Spring 26/CV/P1/YourDirectoryID_p1/Phase1Outputs"

    os.makedirs(out_dir, exist_ok=True)
    extensions=("*.jpeg","*.jpg")
    img_paths=[]
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(img_path, ext)))
    
    img_paths= sorted(img_paths)

    
    # Shi tomasi corner detection - good features to track

    for img in img_paths:	
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = np.float32(gray)
        dst = cv2.goodFeaturesToTrack(gray,5000,0.01,10)
        for p in dst:
            x, y = p.ravel()
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
        img_name = os.path.basename(img)
        out_path = os.path.join(out_dir, f"corners_{img_name}")

        cv2.imwrite(out_path, image)
        
    # Cylindrical warping 

    def cylinderical_warping(img, focal_length):
        h, w = img.shape[:2]
        # Camera internsic matrix
        k_matrix=np.array([[focal_length, 0, w/2],
                          [0,focal_length, h/2],
                          [0,           0,  1]])
        
        # Meshgrid for pixel coordinates

        y_i, x_i = np.indices((h,w))

        theta = (x_i - w / 2) / focal_length
        
        
        h_cyl = (y_i - h / 2) / focal_length
        
        x_map = (focal_length * np.tan(theta)) + (w / 2)
        y_map = (focal_length * h_cyl / np.cos(theta)) + (h / 2)

        cyl_img= cv2.remap(img, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        gray =cv2.cvtColor(cyl_img, cv2.COLOR_BGR2GRAY)

        _, threshold=cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        countours, _= cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if countours:
            x, y, w_c, h_c= cv2.boundingRect(countours[0])
            cyl_img=cyl_img[y:y+h_c,x:x+w_c]

        return cyl_img
    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    # TO get ANMS points for a single image used in feature matching step
    def anms_single_image(image, num_features=1000):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        score_map = cv2.cornerMinEigenVal(gray, blockSize=2, ksize=3)
        local_maxima = imreginalmax(score_map, nloxMax=1200, minDist=10)

        N = len(local_maxima)
        r = [float('inf')] * N

        for i in range(N):
            xi, yi = local_maxima[i]
            si = score_map[yi, xi]

            for j in range(N):
                xj, yj = local_maxima[j]
                sj = score_map[yj, xj]

                if sj > si:
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    r[i] = min(r[i], dist)

        maxima_with_r = list(zip(local_maxima, r))
        maxima_with_r.sort(key=lambda x: x[1], reverse=True)

        return [pt for pt, _ in maxima_with_r[:num_features]]



###################################################
#---Returns ANMS points for all images---#
###################################################
    # def anms(img_paths, out_dir, num_features=100):
    #     global anms_pts

    #     for img in img_paths:
    #         image = cv2.imread(img)
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    #         score_map = cv2.cornerMinEigenVal(gray, blockSize=2, ksize=3)

    #         local_maxima = imreginalmax(score_map, nloxMax=300, minDist=10)

    #         N = len(local_maxima)
    #         r = [float('inf')] * N

    #         for i in range(N):
    #             xi, yi = local_maxima[i]
    #             si = score_map[yi, xi]

    #             for j in range(N):
    #                 xj, yj = local_maxima[j]
    #                 sj = score_map[yj, xj]

    #                 if sj > si:
    #                     dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
    #                     if dist < r[i]:
    #                         r[i] = dist

    #         maxima_with_r = list(zip(local_maxima, r))
    #         maxima_with_r.sort(key=lambda x: x[1], reverse=True)

    #         top_n = min(num_features, len(maxima_with_r))
    #         anms_pts = [maxima_with_r[i][0] for i in range(top_n)]

    #         vis = image.copy()
    #         for x, y in anms_pts:
    #             cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

    #         out_path = os.path.join(out_dir, f"anms_{os.path.basename(img)}")
    #         cv2.imwrite(out_path, vis)
    # anms(img_paths, out_dir, num_features=150)
    # print("Stage1: ANMS points:", len(anms_pts))


    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    def feature_descriptor(image, keypoints_xy, patch_size=41, out_size=8):
        assert patch_size % 2 == 1, "patch size must be odd"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        half = patch_size // 2

        descriptors = []
        valid_kps = []

        for (x, y) in keypoints_xy:
            x = int(x)
            y = int(y)

            # To skip border points
            if x - half < 0 or x + half >= W or y - half < 0 or y + half >= H:
                continue

            patch = gray[y-half:y+half+1, x-half:x+half+1]

            # Gaussian blur
            patch_blur = cv2.GaussianBlur(patch, (0, 0), sigmaX=1.0)

            # Subsample to 8x8
            patch_small = cv2.resize(
                patch_blur, (out_size, out_size), interpolation=cv2.INTER_AREA
            )

            vec = patch_small.astype(np.float32).reshape(-1)

            # Standardization
            mu = vec.mean()
            sigma = vec.std()
            if sigma < 1e-5:
                continue

            vec = (vec - mu) / sigma

            descriptors.append(vec)
            valid_kps.append((x, y))
        if len(descriptors) == 0:
            return np.zeros((0, out_size*out_size), np.float32), np.zeros((0, 2), np.int32)

        return np.array(descriptors, dtype=np.float32), np.array(valid_kps, dtype=np.int32)
    
    image = cv2.imread(img_paths[0])

    # descriptors, keypoints = feature_descriptor(image, anms_pts) # Feature Descriptor obtained here
    # print("ANMS points:", len(anms_pts))
    # print(descriptors.shape)  # (N, 64)
    # print(descriptors[0].shape) # (64,)
    # print(keypoints.shape) 


    
    # To get features of individual images
    features = []
    for img_path in img_paths:
        img =cv2.imread(img_path)
        
        # img =cv2.resize(img,(640,480))
        img_width=img.shape[1]
        FOCAL_LENGTH=img_width*0.8

        # Cylindrical Projection

        img=cylinderical_warping(img,FOCAL_LENGTH)

        anms_pts = anms_single_image(img, num_features=500)
        desc, kps = feature_descriptor(img, anms_pts)
        features.append({
            "image" : img,
            "keypoints" : kps,
            "descriptors" : desc
        })



    """
        Feature Matching
        Save Feature Matching output as matching.png
    """
    # Feature matching step
    
    def feature_matching(desc1, desc2, kp1, kp2, ratio_thresh=0.90):
        matches = []

        for i, d1 in enumerate(desc1):
            dists = np.linalg.norm(desc2 - d1, axis = 1)
            if len(dists) < 2:
                continue
            idxs = np.argsort(dists)
            best, second_best = dists[idxs[0]], dists[idxs[1]]
            if best / second_best < ratio_thresh:
                matches.append((i, idxs[0]))
        return matches

    #srict feature matching
    # def feature_matching(desc1, desc2, kp1, kp2, ratio_thresh=0.8): # thresh ignored here
    #     # 1. Normalize descriptors (if not already done)
    #     # 2. Compute full distance matrix
    #     # Optimization: Use cv2.BFMatcher for speed, or numpy broadcasting if comfortable
        
    #     # We'll stick to numpy for your constraints, but optimized
    #     matches = []
        
    #     if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
    #         return []
    #     # Brute force all distances
    #     # Distance matrix calculation (N x M)
    #     # Using (a-b)^2 = a^2 + b^2 - 2ab for speed
    #     d1_sq = np.sum(desc1**2, axis=1, keepdims=True)
    #     d2_sq = np.sum(desc2**2, axis=1, keepdims=True)
    #     dist_matrix = np.sqrt(d1_sq + d2_sq.T - 2 * np.dot(desc1, desc2.T))

    #     # Forward Check: For each i in desc1, find best j in desc2
    #     forward_idx = np.argmin(dist_matrix, axis=1)
        
    #     # Backward Check: For each j in desc2, find best i in desc1
    #     backward_idx = np.argmin(dist_matrix, axis=0)

    #     # Cross Check: Keep only if they agree (Mutual consent)
    #     for i, j in enumerate(forward_idx):
    #         if backward_idx[j] == i:
    #             matches.append((i, j))

    #     print(f"  -> Cross-Check Matches: {len(matches)}")
    #     return matches
    # for i in range(len(features) - 1):
    #     img1 = features[i]["image"]
    #     img2 = features[i+1]["image"]

    #     kps1 = features[i]["keypoints"]
    #     kps2 = features[i+1]["keypoints"]

    #     desc1 = features[i]["descriptors"]
    #     desc2 = features[i+1]["descriptors"]

    #     matches = feature_matching(desc1, desc2, kps1, kps2, ratio_thresh=0.7)

    #     kp1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps1]
    #     kp2_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps2]

    #     cv_matches = [
    #         cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0)
    #         for i, j in matches
    #     ]

    #     vis = cv2.drawMatches(
    #         img1, kp1_cv,
    #         img2, kp2_cv,
    #         cv_matches, None,
    #         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    #     )

    #     cv2.imwrite(f"matches_{i}_{i+1}.png", vis)

    #Getting Feature Correspondence points are array
    def build_feature_correspondence(matches, kp1, kp2):
        pts1 = []
        pts2 = []
        for i, j in matches:
            pts1.append(kp1[i])
            pts2.append(kp2[j])
        return np.array(pts1), np.array(pts2)
    
        
    """
    Refine: RANSAC, Estimate Homography
    """
    
    def computer_homography(pts1, pts2):
        A= []
        for (x,y), (xp,yp) in zip(pts1, pts2):
            A.append([-x,-y,-1,0,0,0,x*xp,y*xp,xp])
            A.append([0,0,0,-x,-y,-1,x*yp,y*yp,yp])
        A = np.array(A)
        _, _, VT = np.linalg.svd(A)
        H = VT[-1].reshape(3,3)
        return H / H[2,2]
    
    def ssd(p2, Hp1):
        return np.sum((p2 - Hp1)**2)
    
    def apply_homography(H, pts):
        x, y = pts
        p = np.array([x, y, 1.0])
        Hp = H @ p
        Hp = Hp / Hp[2]
        return Hp[:2]
    
    def ransac(kps1, kps2, matches, Nmax=5000, tau=10.0):
        best_inliers = []
        best_H = None

        pts1_all = np.array([kps1[i] for i, _ in matches])
        pts2_all = np.array([kps2[j] for _, j in matches])
        M = len(matches)

        if M < 4:
            return None, []

        for _ in range(Nmax):
            idx = np.random.choice(M, 4, replace=False)
            pts_1 = pts1_all[idx]
            pts_2 = pts2_all[idx]

            H = computer_homography(pts_1, pts_2)
            inliers = []

            for i in range(M):
                Hp1 = apply_homography(H, pts1_all[i])
                error = np.linalg.norm(pts2_all[i] - Hp1)
                if error < tau:
                    inliers.append(i)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H

        if best_H is None or len(best_inliers) < 4:
            return None, []

        H_final = computer_homography(
            pts1_all[best_inliers],
            pts2_all[best_inliers]
        )
        return H_final, best_inliers

    
    # H, inliers = ransac(kps1, kps2, matches)

    # print("Inliers:", len(inliers))

    def accumulate_homographies(homographies, ref_idx):
        H_acc = [None] * (len(homographies) + 1)
        H_acc[ref_idx] = np.eye(3)
        for i in range(ref_idx - 1, -1, -1):
            H_acc[i] = H_acc[i+1] @ homographies[i]
        for i in range(ref_idx, len(homographies)):
            H_acc[i+1] = H_acc[i] @ np.linalg.inv(homographies[i])
        return H_acc

    
    def get_valid_homography(feat1, feat2, threshold=5):
        """Computes H only if enough matches/inliers exist."""
        matches = feature_matching(feat1["descriptors"], feat2["descriptors"], 
                                    feat1["keypoints"], feat2["keypoints"])

        if len(matches) < threshold:
            return None, 0

        H, inliers = ransac(feat1["keypoints"], feat2["keypoints"], matches)

        if H is None or len(inliers) < threshold:
            return None, 0
        
        return H, len(inliers)
    

    # homographies = []
    # ref_idx = len(features)//2

    # for i in range(len(features) - 1):
    #     img1 = features[i]["image"]
    #     img2 = features[i+1]["image"]

    #     kps1 = features[i]["keypoints"]
    #     kps2 = features[i+1]["keypoints"]

    #     desc1 = features[i]["descriptors"]
    #     desc2 = features[i+1]["descriptors"]

    #     matches = feature_matching(desc1, desc2, kps1, kps2)

    #     H, inliers = ransac(kps1, kps2, matches)
    #     homographies.append(H)
        

        
    #     # print("Inliers:", len(inliers))
    #     # print(H)
    #     # print(len(H))
    #     m = len(H)
    #     n = len(H[0])

    #     cv_matches = [
    #         cv2.DMatch(_queryIdx=matches[i][0],
    #                 _trainIdx=matches[i][1],
    #                 _distance=0)
    #         for i in inliers
    #     ]

    #     kp1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps1]
    #     kp2_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kps2]

    #     vis = cv2.drawMatches(
    #         img1, kp1_cv,
    #         img2, kp2_cv,
    #         cv_matches, None,
    #         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    #     )
         
    #     cv2.waitKey(0)

    #     out_path = os.path.join(
    #         out_dir, f"ransac_matches_{i}_{i+1}.png"
    #     )
    #     cv2.imwrite(out_path, vis)
    
    # H_acc = accumulate_homographies(homographies, ref_idx)


    # --- SMART HOMOGRAPHY CHAINING (Center-Outward) ---
    n_images = len(features)
    ref_idx = n_images // 2  # Use the middle image as the anchor to minimize distortion
    
    H_to_ref = [None] * n_images
    H_to_ref[ref_idx] = np.eye(3)

    print(f"Reference Image Index: {ref_idx}")

    # Chain Forward (From Center -> Right)
    for i in range(ref_idx, n_images - 1):
        # Match i to i+1
        H_curr_next, inliers = get_valid_homography(features[i], features[i+1])
        
        if H_curr_next is not None:
            # We have H that takes points from (i) to (i+1).
            # We need H that takes (i+1) to Reference.
            # Relation: H_(i+1)->Ref = H_i->Ref @ inv(H_i->i+1)
            # Since our get_valid_homography returns H mapping pts1 to pts2 (i -> i+1),
            # we simply accumulate the inverse to move "backwards" to the reference.
            
            # Note: Verify your H direction. 
            # If get_valid_homography(src, dst) computes H * src = dst:
            # To map dst back to src, we need inv(H).
            
            H_to_ref[i+1] = H_to_ref[i] @ np.linalg.inv(H_curr_next)
        else:
            print(f"Break in chain at image {i} -> {i+1}")
            break

    # Chain Backward (From Center -> Left)
    for i in range(ref_idx, 0, -1):
        # Match i to i-1
        # We compute H that maps (i-1) to (i)
        H_prev_curr, inliers = get_valid_homography(features[i-1], features[i])
        
        if H_prev_curr is not None:
            # We have H that takes points from (i-1) to (i).
            # We already know how to get from (i) to Ref.
            # So: H_(i-1)->Ref = H_i->Ref @ H_(i-1)->i
            H_to_ref[i-1] = H_to_ref[i] @ H_prev_curr
        else:
            print(f"Break in chain at image {i} -> {i-1}")
            break

    # Filter out any images that got disconnected
    valid_indices = [i for i, h in enumerate(H_to_ref) if h is not None]
    if len(valid_indices) < 2:
        print("Error: Could not stitch enough images.")
        return

    images = [features[i]["image"] for i in valid_indices]
    H_acc = [H_to_ref[i] for i in valid_indices]

    def compute_panaroma_size(images, H_acc):
        corners_all = []
        for img, H in zip(images, H_acc):
            h, w = img.shape[:2]
            corners = np.array([
                [0,0], [w,0], [w,h], [0,h]
            ], dtype = (np.float32)).reshape(-1,1,2)

            warped = cv2.perspectiveTransform(corners, H)
            corners_all.append(warped)

        corners_all = np.vstack(corners_all)
        x_min, y_min = np.min(corners_all, axis=0).ravel().astype(np.int32)
        x_max, y_max = np.max(corners_all, axis=0).ravel().astype(np.int32)

        return (x_min, y_min, x_max, y_max)

    images = [f["image"] for f in features]
    x_min, y_min, x_max, y_max = compute_panaroma_size(images, H_acc)

    h = y_max - y_min
    w = x_max - x_min
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])


    def feather_weight(warped):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8) * 255  
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist = dist / (dist.max() + 1e-6)
        return dist
    
    panaroma = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for img_pan, H_i in zip(images, H_acc):
        H_warp = T @ H_i
        warped = cv2.warpPerspective(
            img_pan, H_warp, (w, h)
        )
        #Without blending
        # mask = (warped > 0)
        # panaroma[mask] = warped[mask]
        weight = feather_weight(warped)
        
        panaroma += warped.astype(np.float32) * weight[... , None]
        weight_sum += weight
    
    weight_sum[weight_sum == 0] = 1
    panaroma = panaroma/weight_sum[..., None]
    panaroma = np.clip(panaroma, 0, 255).astype(np.uint8)

    out_path = os.path.join(out_dir, "panaroma.png")
    cv2.imwrite(out_path, panaroma)




if __name__ == "__main__":
    main()
