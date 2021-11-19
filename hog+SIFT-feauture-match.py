from skimage import feature
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

########HOG Donusumu#####3
#####

dizin = 'yol_yok' #klasor sec!

scale_percent = 80 # orijinal boyuta uygulanak olcek


image1 = cv2.imread(f'C:/Users/Burak/Desktop/HOG+SIFT+FeatureMatching/{dizin}/test_yolyok.jpg') #aranilan hedef #Dizin Degis!!!
(hog, hog_image1) = feature.hog(image1, orientations=9, 
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                    block_norm='L2-Hys', visualize=True, transform_sqrt=True)
width = int(hog_image1.shape[1] * scale_percent / 100)
height = int(hog_image1.shape[0] * scale_percent / 100)
dim1 = (width, height)
hog_image1 = cv2.resize(hog_image1, dim1, interpolation=cv2.INTER_AREA)
cv2.imshow('HOG_ARANILAN', hog_image1)
cv2.imwrite(f'{dizin}/hog_aranilan.jpg', hog_image1*255.)    #Dizin Degis!!!
cv2.waitKey(0)

image2 = cv2.imread(f'C:/Users/Burak/Desktop/HOG+SIFT+FeatureMatching/{dizin}/yolyok.jpg') #aranan goruntu  #Dizin Degis!!!
(hog, hog_image2) = feature.hog(image2, orientations=9, 
                    pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                    block_norm='L2-Hys', visualize=True, transform_sqrt=True)
width2 = int(hog_image2.shape[1] * scale_percent / 100)
height2 = int(hog_image2.shape[0] * scale_percent / 100)
dim2 = (width2, height2)
hog_image2 = cv.resize(hog_image2, dim2, interpolation=cv2.INTER_AREA)
cv2.imshow('HOG_ARANAN', hog_image2)
cv2.imwrite(f'{dizin}/hog_aranan.jpg', hog_image2*255.)     #Dizin Degis!!!
cv2.waitKey(0)


########HOG Ozellikleriyle kaydedilen goruntulerin SIFT dedektorune sokulup, ozellik eslemesi yapilamasi##########
######
MIN_MATCH_COUNT = 10

img1 = cv.imread(f'C:/Users/Burak/Desktop/HOG+SIFT+FeatureMatching/{dizin}/hog_aranilan.jpg')      #aranilan hedef_HOG   #Dizin Degis!!!
img2 = cv.imread(f'C:/Users/Burak/Desktop/HOG+SIFT+FeatureMatching/{dizin}/hog_aranan.jpg')       #aranan goruntu_HOG    #Dizin Degis!!!

# SIFT dedektoru hazirlaniyor
sift = cv.SIFT_create()
# SIFT ile anahtar noktalar(keypoints) ve tanımlayıcılar(descriptors) bulunuyor
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Tüm iyi eşleşmeler Lowe oranina göre saklanir.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    print( "Matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()