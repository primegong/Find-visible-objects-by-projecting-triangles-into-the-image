import os, sys, cv2
import numpy as np
import skimage
import skimage.transform
import pickle as pkl
import matplotlib.pyplot as plt


####################################### masks ##################################################
def get_total_masks(path, segments_list, name):
	#################### read ########################
	total_masks = {}
	total_contours = {}
	
	for pkl_file in segments_list:
		f = open(os.path.join(path, pkl_file +'.pkl'),'rb')
		f.seek(0)
		r = pkl.load(f)

		try:
			masks = r['masks']
			rois = r['rois']
		except:
			masks = r

		obj_mask = {}
		obj_coutours = {}
		for i, [y1, x1, y2, x2] in enumerate(rois):
			maskImage = masks[:,:, i]

			#### added by yiping : resize 1024 to 1000
			resized_maskImage = skimage.transform.resize(maskImage, (1000,1000))
			contours, hierarchy = cv2.findContours(resized_maskImage.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
			contours = contours[0][:,0,:]
			obj_coutours[str(i)] = contours


			scale = 1000.0/1024.0
			yy1 = int(round(y1 * scale))
			xx1 = int(round(x1 * scale))
			yy2 = int(round(y2 * scale))
			xx2 = int(round(x2 * scale))
			outshape=(yy2-yy1+1, xx2-xx1+1)
			mini_mask = skimage.transform.resize(resized_maskImage[y1:y2+1, x1:x2+1], outshape)
			obj_mask[(yy1, xx1, yy2, xx2)] = mini_mask

			
		total_masks[os.path.splitext(pkl_file)[0]] = obj_mask
		total_contours[os.path.splitext(pkl_file)[0]] = obj_coutours

	return total_masks, total_contours

def shrink(masks, pixel=20):
	new_masks = masks.copy()
	N = masks.shape[-1]
	for j in range(N):
		mask = masks[:,:,j]
		
		mask = np.where(mask == False, 0, 1)
		mask = mask.astype(np.uint8)
		
		for i in range(pixel):
			n, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			for cont in contours:
				y = cont[:,:,0]
				x = cont[:,:,1]
				mask[x,y] = 0
		new_masks[:,:,j] = mask
	return new_masks 


def reconstruct(masks, r):
	rois = r['rois']
	new_masks = {}
	for i, [y1, x1, y2, x2] in enumerate(rois):
		new_mask = masks[y1:y2,x1:x2,i]
		new_masks[(y1, x1, y2, x2)] = new_mask
	return new_masks



class Border_Optimizer:
	##### 消除两个mask的边界重叠区域 
	def __call__( masks):
		N = masks.shape[-1]
		new_masks = masks.copy()
		for i in range(N):
			mask = masks[:,:,i]
			
			for j in range(i+1,N):
				cgroup= masks[:,:,j]

				mask_m = np.where(mask==True, 1,0)
				cgroup_m = np.where(cgroup==True, 1,0)
				added = mask_m + cgroup_m
				
				new_masks[:,:,i] = np.where(added==2, False, new_masks[:,:,i])
		return new_masks
