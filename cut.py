import cv2
import scipy.io as io
import os
import numpy as np

mat_path = "/home/disk1/zwz/v2/other_code/result/mscsc/heavy_21.mat"
# mat_path = "/home/disk1/zwz/other_code/FastDeRain-local/Data/light_23s_rainy_case1.mat"
out = io.loadmat(mat_path)
out = out["B_MSCSC"]
o = out[:, :, :, 0]

mat_b = out[:,:,0,:]
mat_g = out[:,:,1,:]
mat_r = out[:,:,2,:]
mat = np.zeros((500, 889, 3, 10))
mat[:,:,0,:] = mat_r
mat[:,:,1,:] = mat_g
mat[:,:,2,:] = mat_b

# o = o*255.0
cv2.imshow("src", mat[:,:,:,0])
cv2.imshow("src1", mat[:,:,:,1])
cv2.imshow("src2", mat[:,:,:,2])
cv2.imshow("src3", mat[:,:,:,3])
cv2.imshow("src4", mat[:,:,:,4])
cv2.imshow("src5", mat[:,:,:,5])
cv2.imshow("src6", mat[:,:,:,6])
cv2.imshow("src7", mat[:,:,:,7])
cv2.imshow("src8", mat[:,:,:,8])
cv2.imshow("src9", mat[:,:,:,9])

cv2.waitKey(0)