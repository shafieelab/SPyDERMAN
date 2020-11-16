import cv2
import numpy as np
import os
import itertools

save_path = "../Data/Test/mb1/Negative/"
fold_path = "../Data/Test/mb1/Negative/"
# fold1_path = "E:/PEOPLE files/Deeksha/Bubble Project/Data/Individual Experiments for Comparative Assay/Control_Tile_Test/"
# fold2_path = "E:/PEOPLE files/Deeksha/Bubble Project/Data/Individual Experiments for Comparative Assay/Positive_Tile_Test/"
# save_path = "E:/PEOPLE files/Deeksha/Bubble Project/Data/Individual Experiments for Comparative Assay/Positive_Control_Tile_Test/"

fold1 = []
fold2 = []
#final = []

# for root, dirs, files in os.walk(fold1_path):
#     for file in files:
#         fold1.append(fold1_path + file)
#
# for root, dirs, files in os.walk(fold2_path):
#     for file in files:
#         fold2.append(fold2_path + file)
        
# print(len(fold1))
# print(len(fold2))
#
# res = [[i, j] for i in fold1
#               for j in fold2]
#
# for couple in res:
#     im1 = cv2.imread(couple[0])
#     name1 = couple[0].split('/')[-1]
#     im2 = cv2.imread(couple[1])
#     name2 = couple[1].split('/')[-1]
#     h1, w1, _ = im1.shape
#     h2, w2, _ = im2.shape
#     mon = np.hstack((im1,im2))
#     cv2.imwrite(save_path + name1[:-4] + "_" + name2[:-4] + ".jpg", mon)
    

for root, dirs, files in os.walk(fold_path):
    for file in files:
        print(file)
        im = cv2.imread(fold_path + file)
        im = cv2.resize(im, (250,2250), interpolation = cv2.INTER_AREA) #resize image to 250 x 2250
        h, w, _ = im.shape
        split = int(h/3) #split image horizontally into three equal parts
        s1 = im[:split,:,:]
        s2 = im[split:split*2,:,:]
        s3 = im[split*2:,:,:]
        mon = np.hstack((s1,s2,s3)) #tile the parts adjacently
        cv2.imwrite(save_path + file[:-4] + "_mon.jpg", mon)

		
                                                                                                                                       
