import os, sys, cv2
import numpy as np
import xml.dom.minidom as DM
import pickle as pkl
import scipy
import skimage
import skimage.transform
# import kdtree
# import codecs
# from skimage.measure import find_contours
# from matplotlib.patches import Polygon
# import matplotlib.pyplot as plt
# import colorsys,random
from progressbar import *
import time
import random
import multiprocessing


def get_pixelPoint(vertice, R, center, K, distortion):
	[k1, k2, k3, p1, p2] = distortion
	vertice = vertice[1:] if len(vertice)==4 else vertice
	vertice = np.array(vertice).reshape((3,1))
	center = center.reshape((len(center),1))

	T = -np.dot(R, center)

	#camera coordination
	XYZ = np.dot(R,vertice) + T  
	u_ = XYZ[0,0]/XYZ[2,0]
	v_ = XYZ[1,0]/XYZ[2,0]
	r_2 = u_* u_ + v_ * v_
	u = u_*(1+k1*r_2+k2*r_2**2+k3*r_2**3)+2*p1*u_*v_+p2*(r_2+2*u_*u_)
	v = v_*(1+k1*r_2+k2*r_2**2+k3*r_2**3)+2*p2*u_*v_+p1*(r_2+2*v_*v_)

	px = K[0,0]*u+K[0,2]
	py = K[1,1]*v+K[1,2]
	return px, py


def cal_area(triangle_1):
	a = L2_dist(triangle_1[0],triangle_1[1])
	b = L2_dist(triangle_1[0],triangle_1[2])
	c = L2_dist(triangle_1[1],triangle_1[2])
	p = (a+b+c)/2
	area = np.sqrt(p*(p-a)*(p-b)*(p-c))
	return area

def write_into_obj(relation, save_dir, triangles):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for key, value in relation.items():
		indexes = list(np.array(value['id']).astype(np.int))
		# if len(indexes)<=100:
		# 	continue
		group_triangles = triangles[indexes,:,:]
		with open(os.path.join(save_dir, str(key) + '.obj'),'w') as f:
			all_vertice = []
			for triangle in group_triangles:
				for vertice in triangle:
					vertice_str = 'v '+str(vertice[0]) + ' ' + str(vertice[1]) + ' ' + str(vertice[2]) + '\n'
					all_vertice.append(vertice_str)
			all_vertice_set = list(set(all_vertice))
			all_vertice_set.sort(key = all_vertice.index)
			f.writelines(all_vertice_set)

			vertice_indexes = [all_vertice_set.index(vertice_str) + 1 for vertice_str in all_vertice] #### 顶点编号从1开始
			vertice_indexes = np.array(vertice_indexes).reshape(-1,3)
			vertice_indexes = ['f ' + str(vi[0]) + ' ' + str(vi[1]) + ' ' + str(vi[2]) + '\n' for vi in vertice_indexes]
			f.writelines(vertice_indexes)

			

def angle(ray_c1, ray_c2):
	norm1 = np.sqrt(np.sum(ray_c1**2))
	norm2 = np.sqrt(np.sum(ray_c2**2))
	if norm1*norm2 < 0.000001:
		angle = -1
	else:
		angle = np.dot(ray_c1, ray_c2)/(norm1*norm2)
	return angle


def get_sub_loc(segment, x_num_region):
	segment_file = '_'.join(segment.split('_')[:2])
	region_id = segment.split('_')[-1]
	row,col = divmod(int(region_id)-1, x_num_region)
	return segment_file, region_id, row, col

def get_normal(triangle):
	v11 = triangle[1, :] - triangle[0, :]
	v22 = triangle[2, :] - triangle[0, :]
	normal = np.cross(v11, v22)
	normal_unit = normal / np.linalg.norm(normal)
	return normal_unit, normal

def L2_dist(a, b):
	a = np.array(a)
	b = np.array(b)
	return np.sqrt(np.sum((a - b) ** 2))

def get_bbox(tria_2D):
	tria_2D = np.array(tria_2D)
	[x_left, y_top] = np.min(tria_2D, axis = 0)
	[x_right, y_bottom] = np.max(tria_2D, axis = 0)
	bbox = [x_left, y_top, x_right, y_bottom]
	return bbox

def get_instance_id(bbox, masks, tria_2D):
	#先求box是否有交集，如果有，再求像素交集
	[x_left, y_top, x_right, y_bottom] = bbox
	instance_id = -1
	fg_pixels = 0
	for idx, key in enumerate(masks.keys()):
		(y1, x1, y2, x2) = key
		#overlap region
		delta_x =  min(x2, x_right) - max(x1, x_left)
		delta_y =  min(y2, y_bottom) - max(y1, y_top)
		if delta_x> 0  and delta_y>0:
			### 如果box方面有交集，再统计交集box内同时落在mask和tria_2D内的像素，即落在tria_2D的前景实例像素
			overlap_region = [max(x1, x_left), max(y1, y_top), min(x2, x_right), min(y2, y_bottom)]
			obj_mask = masks[key]
			overlap = count_fg_pixels_in_tria_2D(overlap_region, tria_2D, obj_mask, key)
			if overlap > fg_pixels:
				#这里选交集像素最多的，记录其instance id
				fg_pixels = overlap
				instance_id = idx
		else:
			continue
	return instance_id, fg_pixels

def in_triangle(point, tria_2D):
	flag = False
	tria_2D = np.array(tria_2D)
	
	if len(tria_2D.shape) == 2:

		a = tria_2D[0,:]
		b = tria_2D[1,:]
		c = tria_2D[2,:]
		
		direct1 = np.cross((b-a), (point-a))*1.0
		direct2 = np.cross((c-b), (point-b))*1.0
		direct3 = np.cross((a-c), (point-c))*1.0

		# print(direct1,direct2, direct3)
		#cause O locate in left-top cornor, so when direct <0, the p is inside the tria
		if direct1<=0 and direct2<=0 and direct3<=0:
			flag = True
	
	if len(tria_2D.shape) == 3:

		a = tria_2D[:,0,:]
		b = tria_2D[:,1,:]
		c = tria_2D[:,2,:]
		
		direct1 = np.cross((b-a), (point-a))*1.0
		direct2 = np.cross((c-b), (point-b))*1.0
		direct3 = np.cross((a-c), (point-c))*1.0

		# print(direct1, direct2, direct3)
		indexes = (np.where(direct1<=0) and np.where(direct2<=0) and np.where(direct3<=0))[0]
		flag = indexes

	return flag

def count_pixels_in_tria_2D(region, tria_2D):
	[x_left, y_top, x_right, y_bottom] = region
	pixels_in_tria = 0
	for pi in range(x_left, x_right+1):
		for pj in range(y_top, y_bottom + 1):
			point = np.array([pi, pj])+0.5
			if in_triangle(point, tria_2D):
				pixels_in_tria += 1
	return pixels_in_tria

def count_fg_pixels_in_tria_2D(region, tria_2D, obj_mask, box):
	[y1,x1,y2,x2] = box
	[x_left, y_top, x_right, y_bottom] = region
	fg_pixels_in_tria = 0
	for pi in range(x_left, x_right + 1):
		for pj in range(y_top, y_bottom + 1):
			label = obj_mask[pj-y1, pi-x1]
			if label:
				point = np.array([pi, pj])+0.5
				if in_triangle(point, tria_2D):
					fg_pixels_in_tria += 1

	return fg_pixels_in_tria


def is_fg_in_image( tria_2D, masks):
	#### 根据分割结果，判断三角形是否时前景类别 ####
	instance_id = -1
	# find the bounding box of the tria_2D
	bbox = get_bbox(tria_2D)
	# count the pixels that in the tria_2D
	all_pixels = count_pixels_in_tria_2D(bbox, tria_2D)

	if all_pixels > 0:
		#当三角形内部像素数不为0时，求与该三角形交集最大的instance mask，以及交集部分前景像素数量；
		# get the corresponding instance_id
		_instance_id, fg_pixels = get_instance_id(bbox, masks, tria_2D)
		# threshold:0.5
		if fg_pixels / all_pixels >= 0.5:
			instance_id = _instance_id
	else:
		##当三角形极小时，三角形内部像素数为0， 直接考虑三个点是否都落在instance mask内部，返回所对应的instance id
		# consider the three vertices, get the corresponding instance_id
		for idx, key in enumerate(masks.keys()):
			(y1, x1, y2, x2) = key
			obj_mask = masks[key]
			ct = 0
			for [xx, yy] in tria_2D:
				if xx<x1 or xx>x2 or yy<y1 or yy>y2:
					break
				label = obj_mask[yy - y1, xx - x1]
				ct += label
			if ct == 3:
				instance_id = idx
				break
			else:
				continue
	return instance_id, bbox, all_pixels


def project_to_subregion(j, triangle, segment, args, K, distortion, height, width, stride, region_size, x_num_region, total_masks, relation, obj_files):
	
	segment_file, region_id, row, col = get_sub_loc(segment, x_num_region)

	# rotation and camera center
	R = args[segment_file]['Rotation']
	center = args[segment_file]['Center']

	##### 1. 判断三角形法向量是否朝向摄影中心, whether the angle between light and normal is larger than 90 ########
	normal_unit, normal = get_normal(triangle)  # normal of triangle
	direct = center - triangle[0, :] #light
	direct_unit = direct / np.linalg.norm(direct)
	angle_cos = np.dot(direct_unit, normal_unit)
	if angle_cos < 0:  # angle > 90
		return


	#####2. 判断是否在图幅范围内，并记录二维坐标信息 ######
	# transfer to image coordinate, all the three vertices should locate in the image, otherwise, go into the next image
	delta_x = 4 
	tria_2D = []
	for vertice in triangle:
		px, py = get_pixelPoint(vertice, R, center, K, distortion)
		px = int(round(px+delta_x))
		py = int(round(py+delta_x))
		if px < width and px >= 0 and py <height and py >= 0:
			xx = int(px - stride * col)
			yy = int(py - stride * row)
			if xx<0 or yy<0:
				continue
			if xx < region_size and yy < region_size:
				tria_2D.append([xx, yy])	
		else:
			break
	if len(tria_2D) != 3:
		return


	# # 3. 根据分割结果判断是前景还是背景；同时，若有多个三角形对应同一像素，根据距离选出可视三角形
	mask_file = segment_file + '_' + str(region_id)
	masks = total_masks[mask_file]
	instance_id, bbox, all_pixels = is_fg_in_image(tria_2D, masks)

	if instance_id != -1:
		#如果落在前景mask上，即找到对应的instance id,则判断距离远近，只保留离相机重心近的三角形，并存入对应的instance_id组内
		flag = False
		key = mask_file + '_' + str(instance_id)
		distance = L2_dist(np.mean(triangle, axis=0), center)

		if not key in relation.keys():
			# 该目标无已有三角形，直接保存
			#id 为三角形索引
			#pixels 为三角形内部前景像素数量
			relation[key] = {'id': [], 'normal': [], 'tria_2D': [], 'bbox': [], 'distance': [], 'pixels': []}
			flag = True
		else:
			# # no compare:
			#  flag = True

			# # 和已有三角形进行比较（重叠度，距离等），判断该三角形为前景还是背景:
			info = relation[key]
			flag, bg_indexes = is_fg_in_3Dspace(info, tria_2D, distance, bbox, all_pixels)
			## flag 返回当前tria_2D是否时前景；bg_indexes返回距离远的三角形索引

			if len(bg_indexes):
				# del bg trias
				# delete from the end to the start, or error will occur
				bg_indexes = sorted(list(set(bg_indexes)), reverse=True)
				for ind in bg_indexes:
					del (relation[key]['id'][ind])
					del (relation[key]['normal'][ind])
					del (relation[key]['tria_2D'][ind])
					del (relation[key]['bbox'][ind])
					del (relation[key]['distance'][ind])
					del (relation[key]['pixels'][ind])

		if flag:
			relation[key]['id'].append(j)
			relation[key]['normal'].append(normal)
			relation[key]['tria_2D'].append(tria_2D)
			relation[key]['bbox'].append(bbox)
			relation[key]['distance'].append(distance)
			relation[key]['pixels'].append(all_pixels)


def projection(triangles, total_masks, segments_list, para, save_dir, region_size = 1024, step = 0.8):
	N = len(segments_list)
	print('segments_2D file: '+str(N))

	projection_results = {}
	args = para['args']
	projection_results_files = os.path.join(save_dir,'3D_Segments.pkl')
	if os.path.exists(projection_results_files):
		pkl_f = open(projection_results_files,'rb')
		pkl_f.seek(0)
		projection_results = pkl.load(pkl_f)		
	else:
		pbar = ProgressBar().start()
		#parameters
		f = para['f']
		K = para['K']
		Coord = para['Coord']
		distortion = para['Distortion']
		width = para['Image_width']
		height = para['Image_height']
		#### 图像切割参数
		stride = int(region_size*step)
		x_num_region= int(width/stride)
		y_num_region = int(height/stride)

		obj_files = os.listdir(save_dir)
		obj_files = ['_'.join(file.split('_')[:3]) for file in obj_files]
		for i, segment_file in enumerate(segments_list):
			pbar.update(i*1.0/N*100)

			if segment_file in obj_files:
				# already exists!
				continue
			
			##project to subregion
			relation= {}
			for j, triangle in enumerate(triangles):
				project_to_subregion(j, triangle, segment_file, args, K, distortion, height, width, stride, region_size, x_num_region, total_masks, relation, obj_files)
			
			### save the projection results of one segment file into OBJ file and pkl file ####
			if len(relation.keys()):
				print('writing to obj...')
				write_into_obj(relation, save_dir, triangles)

				print('writing to pkl...')
				for key, value in relation.items():
					f = open(os.path.join(save_dir,str(key)+'.pkl'), 'wb')
					pkl.dump(value, f)

				projection_results.update(relation)

		### 保存所有图像的投影结果
		if len(projection_results.keys()):
			f = open(projection_results_files,'wb')
			pkl.dump(projection_results, f)
		pbar.finish()

	return projection_results


def compute_overlap(tria_2D, info, all_pixels, bbox, threshold = 0.5):
	total_overlap_indexes=[]
	[x_left, y_top, x_right, y_bottom] = bbox
	
	# 1. 先求box的层面交集 get the inds of trias that have an overlap with the tria_2D
	bboxes = np.array(info['bbox'])
	delta_x = np.minimum(bboxes[:,2], x_right) - np.maximum(bboxes[:,0], x_left)
	delta_y = np.minimum(bboxes[:,3], y_bottom) - np.maximum(bboxes[:,1], y_top)
	inds_1 = np.where(delta_x > 0)[0]
	inds_2 = np.where(delta_y > 0)[0]
	overlap_inds = np.array(list(set(inds_1).intersection((inds_2))))

	if len(overlap_inds):
		# 2. 再求像素层面的交集
		trias_all = np.array(info['tria_2D'])[overlap_inds]
		all_pixles_of_trias = np.array(info['pixels'])[overlap_inds]
		all_pixles_of_trias = np.maximum(all_pixles_of_trias, 0.001)

		N = len(trias_all)
		count_list = np.zeros(N)
		overlap = []
		iou = []

		if all_pixels==0:
			### 直接考虑三角形三个顶点是否有交集 ###
			points = np.array(tria_2D).astype(np.float)
			for point in points:
				indexes = in_triangle(point+0.5, trias_all)
				count_list[indexes] +=1	
			new_all_pixels = points.shape[0]
			overlap = count_list*1.0/np.minimum(new_all_pixels, all_pixles_of_trias)
			iou = count_list*1.0/(new_all_pixels + all_pixles_of_trias)
		elif all_pixels != 0:
			for pi in range(x_left, x_right + 1):
				for pj in range(y_top, y_bottom + 1):
					point = np.array([pi, pj])+0.5
					if in_triangle(point, tria_2D):
						indexes = in_triangle(point, trias_all)
						count_list[indexes] +=1	
			overlap = count_list *1.0 /np.minimum(all_pixels, all_pixles_of_trias)
			iou = count_list *1.0 /(all_pixels + all_pixles_of_trias)
	
		inds = np.where(overlap>threshold)[0]
		if len(inds):
			overlap_indexes = overlap_inds[inds]
			# combine the small trias and the ovelap>0.5
			total_overlap_indexes.extend(overlap_indexes)
	return np.array(list(set(total_overlap_indexes)))

def is_fg_in_3Dspace(info, tria_2D, distance, bbox, all_pixels):
	#### second step: whether the triangle in 3D space(tria_2D in image) is the fg?
	#### compare it with the trianales already exists. keep the fg and delete bg ####
	'''
	flag 记录当前tria_2D是否为距离最近的
	bg_indexes 返回距离远的三角形索引
	'''
	flag = False
	bg_indexes = []
	#距离比较时，容差为墙的宽度，即用很小的值代替0
	wall_width=0.05
	wall_length = 0.22

	# 计算三角形和已有三角形的交集，返回iou>0.5的所有三角形，然后比较距离
	# find the trias with a iou>0.5 of tria_2D, return their indexes
	overlap_index = compute_overlap(tria_2D, info, all_pixels, bbox)

	if len(overlap_index) == 0:
		flag = True
	elif len(overlap_index) == 1:
		ind = overlap_index[0]
		distance_pre = info['distance'][ind]
		if distance-distance_pre<=-wall_width:
			bg_indexes = overlap_index
			flag = True
		elif abs(distance-distance_pre)<=wall_width:
			flag = True
	elif len(overlap_index) > 1:
		distances_all = np.array(info['distance'])[overlap_index]
		dis_min = min(min(distances_all), distance)

		# bg trias
		bg_ind = np.where(distances_all-dis_min>wall_width)[0]
		bg_indexes = overlap_index[bg_ind]

		# if the tria is fg,add to relation
		if abs(distance-dis_min) <=wall_width:
			flag = True
	return flag, bg_indexes

def get_rectify_coord( px, py, f, R, Coord, center):
	center = np.array(center).reshape((3,1))
	pt_cam = (np.array([px,py,-f])-np.array([Coord[0],Coord[1],0])).reshape((3,1))
	[u0, v0, w0] = np.dot(R, pt_cam)
	u = -f*u0/w0 
	v = -f*v0/w0
	return u[0], v[0]

def get_rectify_coord_2( vertice, R, center,K):
	vertice = vertice[1:] if len(vertice)==4 else vertice
	vertice = vertice.reshape((3,1))
	center = center.reshape((len(center),1))
	T = -np.dot(R, center)
	Ximg = (np.dot(R,vertice) + T)

	return Ximg

def disp_measure( center_i, center_j, vertice, f):
	p = center_j-center_i
	q = center_j-vertice
	s = np.sqrt(np.sum(np.cross(p,q)**2))
	baseline = np.sqrt(np.sum(p**2))
	H = s/baseline
	if baseline<1:
		print(baseline,'##################################')
	disp_m = f*baseline/H
	return disp_m