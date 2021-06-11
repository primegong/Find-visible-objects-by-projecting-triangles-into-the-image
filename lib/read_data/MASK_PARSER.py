import os, sys, cv2
import numpy as np
import skimage
import skimage.transform
import pickle as pkl



def compute_polygon_area(points):
	point_num = len(points)
	if(point_num < 3): 
		return 0.0
	s = points[0][1] * (points[point_num-1][0] - points[1][0])
	#for i in range(point_num): # (int i = 1 i < point_num ++i):
	for i in range(1, point_num): # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
		s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
	return abs(s/2.0)

####################################### masks ##################################################
def get_total_masks(path, segments_list, name):
	# global rois

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

	
	print('read from masks files....')
	# bo = border_optimizer()
	# pkl_files = os.listdir(masks_path)
	# for pkl_file in pkl_files:
	# 	if os.path.splitext(pkl_file)[-1] != '.pkl':
	# 		continue
	# 	mask_file = os.path.join(masks_path, pkl_file)
	# 	mask_f = open(mask_file,'rb')
	# 	mask_f.seek(0)
	# 	r = pkl.load(mask_f)
	# 	masks = r['masks']
	# 	masks = bo(masks)
	# 	masks = shrink(masks, pixel=20)
	# 	masks = reconstruct(masks, r)

	# 	r['masks'] = masks
	# 	save_f = open('../visualize_new/' + pkl_file,'wb')
	# 	pkl.dump(r, save_f)

	#################### read ########################
	total_masks = {}
	
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
		for i, [y1, x1, y2, x2] in enumerate(rois):
			#### added by yiping : resize 1024 to 1000
			scale = 1000.0/1024.0
			yy1 = int(round(y1 * scale))
			xx1 = int(round(x1 * scale))
			yy2 = int(round(y2 * scale))
			xx2 = int(round(x2 * scale))
			outshape=(yy2-yy1+1, xx2-xx1+1)
			mask = skimage.transform.resize(masks[y1:y2+1, x1:x2+1, i], outshape)

			## 计算面积：
			contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
			mask_area = compute_polygon_area(contours)

			obj_mask[(yy1, xx1, yy2, xx2, mask_area)] = mask
			# obj_mask[(y1, x1, y2, x2)] = masks[y1:y2+1, x1:x2+1, i]
		total_masks[os.path.splitext(pkl_file)[0]] = obj_mask

	# total_masks_pkl = os.path.join(path, '..', name)
	# save_f = open(total_masks_pkl,'wb')
	# pkl.dump(total_masks, save_f)
	return total_masks


class Border_Optimizer:
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
