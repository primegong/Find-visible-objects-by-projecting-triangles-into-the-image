import os, sys, cv2
import numpy as np
import xml.dom.minidom as DM
import pickle as pkl
import scipy
import skimage
import skimage.transform
from progressbar import *
import time
import random
import multiprocessing
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def projection(triangles, total_contours, segments_list, para, save_dir, region_size = 1024, step = 0.8):
	N = len(segments_list)
	print('segments_2D file: '+str(N))
	
	#parameters
	f = para['f']
	K = para['K']
	Coord = para['Coord']
	distortion = para['Distortion']
	width = para['Image_width']
	height = para['Image_height']
	args = para['args']

	#### 图像切割参数
	stride = int(region_size*step)
	x_num= int(width/stride)
	y_num = int(height/stride)

	#### 跳过已经处理过的影像
	obj_files = os.listdir(save_dir)
	obj_files = ['_'.join(file.split('_')[:3]) for file in obj_files]

	#### 进行投影：记录每张影像上，每个目标mask对应的空间三角形######
	pbar = ProgressBar().start()
	for i, segment_file in enumerate(segments_list):
		pbar.update(i*1.0/N*100)
		if segment_file in obj_files:
			#### 跳过已经处理过的影像
			continue

		## 1. colloct triangles located at this image
		##### 依据图像名称解析空三信息
		bigImageName, subRegion_id, row, col = get_sub_loc(segment_file, x_num)
		R = args[bigImageName]['Rotation']
		center = args[bigImageName]['Center']
		objectBoundary = total_contours[segment_file]
		projResult= {}
		for j, triangle in enumerate(triangles):
			instance_id, tria_2D, normal = project_to_2DImage(triangle, R, center, K, distortion, row, col, height, width, stride, region_size, x_num, objectBoundary)
			if instance_id == -1:
				continue 
			else:
				### 如果落在mask感兴趣目标区域，存入对应的instance id组内。
				groupName = segment_file + '_' + str(instance_id)
				if not groupName in projResult.keys():
					projResult[groupName] = {'id': [], 'tria_2D': [],  'normal': []}
				projResult[groupName]['id'].append(j)
				projResult[groupName]['tria_2D'].append(tria_2D)
				projResult[groupName]['normal'].append(normal)

		
		## 2. 如果组内三角形存在重叠，根据到摄影中心的距离选出可视三角形
		for groupName, result in projResult.items():
			trianglesIndex = result['id']
			trias_2D = result['tria_2D']
			trianglesNormal = result['normal']
			distances = [L2_dist(np.mean(triangle, axis=0), center) for triangle in triangles[trianglesIndex]]
			fgIndex= filter_backgroundCluster(trias_2D, distances)
			if len(fgIndex):
				trianglesIndex = np.array(trianglesIndex)[fgIndex]
				trias_2D = np.array(trias_2D)[fgIndex]
				distances = np.array(distances)[fgIndex]
				trianglesNormal = np.array(trianglesNormal)[fgIndex]

				projResult[groupName] = {
					'id': trianglesIndex,
					'tria_2D':trias_2D,
					'distance':distances,
					'normal':trianglesNormal
				}
			else:
				del projResult[groupName]

		## 3. save the projection results of one segment file into OBJ file and pkl file ####
		if len(projResult.keys()):
			print('writing to obj...')
			write_into_obj(projResult, save_dir, triangles)

			print('writing to pkl...')
			for groupName, value in projResult.items():
				f = open(os.path.join(save_dir,str(groupName)+'.pkl'), 'wb')
				pkl.dump(value, f)
		


def project_to_2DImage(triangle, R, center, K, distortion, row, col, height, width, stride, region_size, x_num, objectBoundary):
	instance_id = -1
	tria_2D = []


	##### 1. 首先判断三角形法向量是否朝向摄影中心, whether the angle between light and normal is larger than 90 ########
	normal_unit, normal = get_normal(triangle)  # normal of triangle
	direct = center - triangle[0, :] #light
	direct_unit = direct / np.linalg.norm(direct)
	angle_cos = np.dot(direct_unit, normal_unit)
	if angle_cos < 0:  # angle > 90
		return instance_id, tria_2D, normal



	#####2. 判断是否在图幅范围内，如果在，记录投影后的二维三角形坐标 ######
	# transfer to image coordinate, all the three vertices should locate in the image, otherwise, go into the next image
	delta_x = 4 
	for vertice in triangle:
		px, py = get_pixelPoint(vertice, R, center, K, distortion)
		px = int(round(px+delta_x))
		py = int(round(py+delta_x))

		if px < width and px >= 0 and py <height and py >= 0:
			####  如果在大图范围内，就转到小图 
			xx = int(px - stride * col)
			yy = int(py - stride * row)
			if xx<0 or yy<0:
				continue
			if xx < region_size and yy < region_size:
				tria_2D.append([xx, yy])	
		else:
			break
	if len(tria_2D) != 3:
		return instance_id, tria_2D, normal

	# # 3. 来到小图之后，根据分割结果判断是前景还是背景，并返回所属实例目标ID
	instance_id = ifForground(tria_2D, objectBoundary)
	return instance_id, tria_2D, normal_unit


def filter_backgroundCluster(trias_2D, distances):
	wall_width=0.05
	### 如果与组内三角形存在重叠，根据到摄影中心的距离选出可视三角形
	flag = np.ones(len(trias_2D))
	for i, tria_2D_i in enumerate(trias_2D):
		if flag[i] == -1:
			continue

		for j, tria_2D_j in enumerate(trias_2D[i+1:]):
			if flag[i+1+j] == -1:
				continue

			iou = compute_tri_iou(tria_2D_i, tria_2D_j)
			if iou >0.5:
				dis_i = distances[i]
				dis_j = distances[i+1+j]
				delta_dis = abs(dis_i - dis_j)
				if delta_dis>wall_width:
					if dis_i > dis_j:
						flag[i] = -1
					else:
						flag[i+1+j] = -1
	fgIndex = np.where(flag>0)[0]
	
	return fgIndex



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



def ifForground(tria_2D, objectBoundary):
	instance_id = -1
	maxOverlap = 0
	for objID, boundary in objectBoundary.items():	
		overlap = compute_overlap(tria_2D, boundary)
		if overlap > 0.5:
			if overlap > maxOverlap:
				maxOverlap = overlap
				instance_id = objID

	return instance_id


def compute_overlap(list1, list2):
	"""
	Intersection over union between two shapely polygons.
	"""
	polygon_points1 = np.array(list1).reshape(3, 2)
	poly1 = Polygon(polygon_points1).convex_hull
	polygon_points2 = np.array(list2).reshape(-1, 2)
	poly2 = Polygon(polygon_points2).convex_hull
	if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
		overlap = 0
	else:
		try:
			inter_area = poly1.intersection(poly2).area
			if poly1.area == 0:
				return 1
			overlap = float(inter_area) / poly1.area
		except shapely.geos.TopologicalError:
			print('shapely.geos.TopologicalError occured, iou set to 0')
			overlap = 0
	return overlap


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



def compute_tri_iou(list1, list2):
	"""
	Intersection over union between two shapely polygons.
	"""
	polygon_points1 = np.array(list1).reshape(3, 2)
	poly1 = Polygon(polygon_points1).convex_hull
	polygon_points2 = np.array(list2).reshape(3, 2)
	poly2 = Polygon(polygon_points2).convex_hull
	# union_poly = np.concatenate((polygon_points1, polygon_points2))
	if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
		iou = 0.0
	else:
		try:
			inter_area = poly1.intersection(poly2).area
			union_area = poly1.area + poly2.area - inter_area
			# union_area = MultiPoint(union_poly).convex_hull.area
			if union_area == 0 or poly1.area==0 or poly2.area==0:
				return 1.0
			

			##### 考虑三角形之间可能存在包含关系，因此取最大交集指标
			overlap1 = inter_area/poly1.area
			overlap2 = inter_area/poly2.area
			iou = float(inter_area) / union_area
			iou = max(iou, max(overlap1,overlap2))
		except shapely.geos.TopologicalError:
			print('shapely.geos.TopologicalError occured, iou set to 0')
			iou = 0.0
	return iou

def angle(ray_c1, ray_c2):
	norm1 = np.sqrt(np.sum(ray_c1**2))
	norm2 = np.sqrt(np.sum(ray_c2**2))
	if norm1*norm2 < 0.000001:
		angle = -1
	else:
		angle = np.dot(ray_c1, ray_c2)/(norm1*norm2)
	return angle


def get_sub_loc(segment_file, x_num):
	bigImageName = '_'.join(segment_file.split('_')[:2])
	subRegion_id = segment_file.split('_')[-1]
	row, col = divmod(int(subRegion_id)-1, x_num)
	return bigImageName, subRegion_id, row, col

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

def write_into_obj(projResult, save_dir, triangles):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for groupName, result in projResult.items():
		triasID = list(np.array(result['id']).astype(np.int))
		group_triangles = triangles[triasID,:,:]
		with open(os.path.join(save_dir, str(groupName) + '.obj'),'w') as f:
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

