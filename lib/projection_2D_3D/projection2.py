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
import time
import multiprocessing


class Projection:
	def __init__(self, model_file):
		self.model_file = model_file
		self.vertices = []
		self.faces = []
		self.triangles = []

	def read_data(self, case = 0):
		try:
			f_generator = open(self.model_file,'r')
		except Exception as e:
			print(e)
		else:
			for index, line in enumerate(f_generator):

				if case == 0:
					pass
				if case == 1:   
					if line[:2] == 'v ':
						v,x,y,z = line.strip().split(' ')
						self.vertices.append([x,y,z])
				if case == 2:
					if line[:2] == 'vt':
						vt,u,v = line.strip().split(' ')
						self.vts.append([j,u,v])
				if case == 3:
					if line[:2] == 'f ':
						f,_index1,_index2,_index3 = line.strip().split(' ')
						try:
							coord_index1, vt_index1 = _index1.split('/')
							coord_index2, vt_index2 = _index2.split('/')
							coord_index3, vt_index3 = _index3.split('/')
							self.faces.append([int(coord_index1), int(vt_index1), int(coord_index2), int(vt_index2), int(coord_index3), int(vt_index3)])
						except:
							coord_index1 = _index1
							coord_index2 = _index2
							coord_index3 = _index3
							self.faces.append([int(coord_index1), int(0), int(coord_index2), int(0), int(coord_index3), int(0)])


						


	def get_faces(self):
		# print('reading faces from obj file...')
		self.read_data(case = 3)
		return self.faces

	def get_vertices(self):
		# print('reading vertices from obj files...')
		self.read_data(case = 1)
		# print('total vertices: ', len(self.vertices))
		return self.vertices

	def get_vts(self):
		self.read_data(case = 2)
		return self.vts

	def get_triangles(self):
		faces = self.get_faces()
		vertices = self.get_vertices()
		# print('loading triangles...')
		for index in faces:
			triangle = [vertices[index[0]-1], vertices[index[2]-1], vertices[index[4]-1]]
			self.triangles.append(triangle)
		self.triangles = np.array(self.triangles).astype(np.float32)
		# print('total triangles: ',len(self.triangles))
		return self.triangles, faces, vertices


	def get_pixelPoint(self, vertice, R, center, K, distortion):
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
	


	def get_rectify_coord(self, px, py, f, R, Coord, center):
		center = np.array(center).reshape((3,1))
		pt_cam = (np.array([px,py,-f])-np.array([Coord[0],Coord[1],0])).reshape((3,1))
		[u0, v0, w0] = np.dot(R, pt_cam)
		u = -f*u0/w0 
		v = -f*v0/w0
		return u[0], v[0]

	def get_rectify_coord_2(self, vertice, R, center,K):
		vertice = vertice[1:] if len(vertice)==4 else vertice
		vertice = vertice.reshape((3,1))
		center = center.reshape((len(center),1))
		T = -np.dot(R, center)
		Ximg = (np.dot(R,vertice) + T)

		return Ximg

	def disp_measure(self, center_i, center_j, vertice, f):
		p = center_j-center_i
		q = center_j-vertice
		s = np.sqrt(np.sum(np.cross(p,q)**2))
		baseline = np.sqrt(np.sum(p**2))
		H = s/baseline
		if baseline<1:
			print(baseline,'##################################')
		disp_m = f*baseline/H
		return disp_m

	def cal_area(self,triangle_1):
		a = self.L2_dist(triangle_1[0],triangle_1[1])
		b = self.L2_dist(triangle_1[0],triangle_1[2])
		c = self.L2_dist(triangle_1[1],triangle_1[2])
		p = (a+b+c)/2
		area = np.sqrt(p*(p-a)*(p-b)*(p-c))
		return area

	def write_into_obj(self, relation, save_dir, triangles):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for key, value in relation.items():
			# try:
			# 	indexes = list(np.array(value)[:,0].astype(np.int))
			# except:
			# 	indexes = [int(item[0]) for item in value]
			indexes = list(np.array(value['id']).astype(np.int))
			# if len(indexes)<=100:
			# 	continue

			group_triangles = triangles[indexes,:,:]
			with open(os.path.join(save_dir, str(key) + '.obj'),'w') as f:
				for triangle in group_triangles:
					trias_str = ['v '+str(v[0]) + ' ' + str(v[ 1]) + ' ' + str(v[2]) + '\n' for v in triangle]
					f.writelines(trias_str)
				for i in range(int(len(group_triangles))):
					f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')
	
	def angle(self,ray_c1, ray_c2):
		norm1 = np.sqrt(np.sum(ray_c1**2))
		norm2 = np.sqrt(np.sum(ray_c2**2))
		if norm1*norm2 < 0.000001:
			angle = -1
		else:
			angle = np.dot(ray_c1, ray_c2)/(norm1*norm2)
		return angle

	def is_fg_in_image(self, tria_2D, masks):
		####  first step: if the tria_2D is fg in image? if so ,find the corresponding obj_id  ####
		obj_id = -1

		# find the bounding box of the tria_2D
		bbox = get_bbox(tria_2D)

		# count the pixels that in the tria_2D
		all_pixels = count_pixels_in_tria_2D(bbox, tria_2D)

		if all_pixels == 0:
			# consider the three vertices, get the corresponding obj_id
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
					obj_id = idx
					break
				else:
					continue

		if all_pixels > 0:
			# get the corresponding obj_id
			_obj_id, fg_pixels = get_obj_id(bbox, masks, tria_2D)

			# threshold:0.5
			if fg_pixels / all_pixels >= 0.5:
				obj_id = _obj_id
		return obj_id, bbox, all_pixels

	def is_fg_in_3Dspace(self, trias, tria_2D, distance, bbox, all_pixels):
		#### second step: if the triangle in 3D space(tria_2D in image) is the fg?
		#### compare it with the trianales already exists. keep the fg and delete bg ####

		flag = False
		bg_indexes = []
		wall_width=0.05
		wall_length = 0.22

		# find the trias with a iou>0.5 of tria_2D, return their indexes
		overlap_index = compute_overlap(tria_2D, trias, all_pixels, bbox)


		if len(overlap_index) == 0:
			flag = True

		if len(overlap_index) == 1:
			ind = overlap_index[0]
			distance_pre = trias['distance'][ind]
			if distance-distance_pre<=-wall_width:
				bg_indexes = overlap_index
				flag = True
			elif abs(distance-distance_pre)<=wall_width:
				flag = True

		if len(overlap_index) > 1:
			distances = np.array(trias['distance'])[overlap_index]
			dis_min = min(min(distances), distance)

			# bg trias
			bg_ind = np.where(distances-dis_min>wall_width)[0]
			bg_indexes = overlap_index[bg_ind]

			# if the tria is fg,add to relation
			if abs(distance-dis_min) <=wall_width:
				flag = True

		return flag, bg_indexes
	
	def get_sub_loc(self, segment, x_num_region):
		segment_file = '_'.join(segment.split('_')[:2])
		region_id = segment.split('_')[-1]
		row,col = divmod(int(region_id)-1, x_num_region)
		return segment_file, region_id, row, col

	def project_to_subregion(self, j, triangle, segment, args, K, distortion, height, width, stride, region_size, x_num_region, total_masks, relation, obj_files):
		
		segment_file, region_id, row, col = self.get_sub_loc(segment, x_num_region)

		# rotation and camera center
		R = args[segment_file]['Rotation']
		center = args[segment_file]['Center']

		##### 1. if the angle between light and normal is larger than 90, then go into the next image ########
		v11 = triangle[1, :] - triangle[0, :]
		v22 = triangle[2, :] - triangle[0, :]
		normal = np.cross(v11, v22)
		normal_unit = normal / np.linalg.norm(normal)  # normal of triangle
		direct = center - triangle[0, :] #light
		direct_unit = direct / np.linalg.norm(direct)
		angle_cos = np.dot(direct_unit, normal_unit)
		if angle_cos < 0:  # angle > 90
			return


		#####2. project to subregion:######
		# transfer to image coordinate, all the three vertices should locate in the image, otherwise, go into the next image
		tria_2D = []
		for vertice in triangle:
			px, py = self.get_pixelPoint(vertice, R, center, K, distortion)
			px = int(round(px+4))
			py = int(round(py+4))
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
	


		# # 3. if the tria_2D is fg in image and in 3D space
		mask_file = segment_file + '_' + str(region_id)
		masks = total_masks[mask_file]
		obj_id, bbox, all_pixels = self.is_fg_in_image(tria_2D, masks)

		if obj_id != -1:
			flag = False
			key = mask_file + '_' + str(obj_id)
			
			distance = L2_dist(np.mean(triangle, axis=0), center)

			if not key in relation.keys():
				relation[key] = {'id': [], 'normal': [], 'tria_2D': [], 'bbox': [], 'distance': [], 'pixels': []}
				flag = True
			else:
				# # no compare:
				#  flag = True

				# # with compare:
				trias = relation[key]
				flag, bg_indexes = self.is_fg_in_3Dspace(trias, tria_2D, distance, bbox, all_pixels)

				if len(bg_indexes):
					# if j == 36913:
					# 	print(bg_indexes, len(relation[key]))
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


	def build(self, triangles, total_masks, segments_list, para, save_dir, region_size = 1024, step = 0.8):
		print('project 2D segments to 3D space .........')
		N = len(segments_list)
		print('segments_2D is '+str(N))

		

		projection_results = {}
		args = para['args']
		projection_results_files = os.path.join(save_dir,'3D_Segments.pkl')
		if os.path.exists(projection_results_files):
			pkl_f = open(projection_results_files,'rb')
			pkl_f.seek(0)
			projection_results = pkl.load(pkl_f)
				
		else:
			pbar = ProgressBar().start()
			f = para['f']
			K = para['K']
			Coord = para['Coord']
			distortion = para['Distortion']
			width = para['Image_width']
			height = para['Image_height']
			#parameters
			stride = int(region_size*step)
			x_num_region= int(width/stride)
			y_num_region = int(height/stride)
			

			obj_files = os.listdir(save_dir)
			obj_files = ['_'.join(file.split('_')[:3]) for file in obj_files]
			for i, segment_file in enumerate(segments_list):
				# if segment_file in obj_files:
				# 	continue
				pbar.update(i*1.0/N*100)

				# already exists!
				if segment_file in obj_files:
					continue
					

				relation= {}
				
				# p = multiprocessing.Pool(processes = 12)#cpu(s)
				# for j, triangle in enumerate(triangles):
				# 	p.apply_async(func = self.project_to_subregion, args = (j, triangle, segment, args, K, distortion, height, width, stride, region_size, x_num_region, total_masks, relation, obj_files))
				# p.close()
				# p.join()
				
				for j, triangle in enumerate(triangles):
					if j%100000 == 0:
						print(j)
					self.project_to_subregion(j, triangle, segment_file, args, K, distortion, height, width, stride, region_size, x_num_region, total_masks, relation, obj_files)
				
				# ####################################### write the projection results of one segment file into OBJ and pkl file ############################################
				if len(relation.keys()):
					print('writing to obj...')
					self.write_into_obj(relation, save_dir, triangles)

					print('writing to pkl...')
					for key, value in relation.items():
						f = open(os.path.join(save_dir,str(key)+'.pkl'), 'wb')
						pkl.dump(value, f)

					projection_results.update(relation)

			if len(projection_results.keys()):
				f = open(projection_results_files,'wb')
				pkl.dump(projection_results, f)
			pbar.finish()

		return projection_results, args




################################################  parse xml  ########################################################################################
def parse_xml(xml_file, delta=None):

		if not os.path.exists(xml_file):
			raise Exception(xml_file + 'is not exists!!!')
		
		try:
			print('parsing ' + xml_file)
			tree = DM.parse(xml_file)
			root = tree.documentElement
		except Exception as e:
			print(e)
			sys.exit(1)
		
		try:
			ImageWidth = float(root.getElementsByTagName('Width')[0].childNodes[0].nodeValue)
			ImageHeight = float(root.getElementsByTagName('Height')[0].childNodes[0].nodelValue)
			FocalLength = float(root.getElementsByTagName('FocalLength')[0].childNodes[0].nodeValue)
			SensorSize = float(root.getElementsByTagName('SensorSize')[0].childNodes[0].nodeValue)
		except:
			ImageWidth = float(root.getElementsByTagName('Width')[0].childNodes[0].data)
			ImageHeight = float(root.getElementsByTagName('Height')[0].childNodes[0].data)
			FocalLength = float(root.getElementsByTagName('FocalLength')[0].childNodes[0].data)
			SensorSize = float(root.getElementsByTagName('SensorSize')[0].childNodes[0].data)
			
		f = ImageWidth*FocalLength/SensorSize


		PrincipalPoint = root.getElementsByTagName('PrincipalPoint')[0]
		x0 = float(PrincipalPoint.getElementsByTagName('x')[0].childNodes[0].nodeValue)
		y0 = float(PrincipalPoint.getElementsByTagName('y')[0].childNodes[0].nodeValue)
		Coord = [x0,y0]
		K = np.array([[f,0,x0],[0,f,y0],[0,0,1]])
	


		Distortion = tree.getElementsByTagName('Distortion')[0]
		K1 = float(Distortion.getElementsByTagName('K1')[0].childNodes[0].nodeValue)
		K2 = float(Distortion.getElementsByTagName('K2')[0].childNodes[0].nodeValue)
		K3 = float(Distortion.getElementsByTagName('K3')[0].childNodes[0].nodeValue)
		P1 = float(Distortion.getElementsByTagName('P1')[0].childNodes[0].nodeValue)
		P2 = float(Distortion.getElementsByTagName('P2')[0].childNodes[0].nodeValue)
		distortion = np.array([K1,K2,K3,P1,P2])

		Photos = tree.getElementsByTagName('Photo')
		args = {}
		for photo in Photos:
			abs_path = photo.getElementsByTagName('ImagePath')[0].childNodes[0].nodeValue
			folder = abs_path.split('/')[-2]
			# extra = abs_path.split('/')[-2]
			base_name = os.path.basename(photo.getElementsByTagName('ImagePath')[0].childNodes[0].nodeValue)
			image_name = folder + '_' + os.path.splitext(base_name)[0]
			
			try:
				Pose = photo.getElementsByTagName('Pose')[0]
			except:
				pass
	
			try:
				Rotation = Pose.getElementsByTagName('Rotation')[0]
			except:
				print('warning:' + image_name + ' file has no rotation parameter...')
				continue
			Center = Pose.getElementsByTagName('Center')[0]
			x = float(Center.getElementsByTagName('x')[0].childNodes[0].nodeValue)
			y = float(Center.getElementsByTagName('y')[0].childNodes[0].nodeValue)
			z = float(Center.getElementsByTagName('z')[0].childNodes[0].nodeValue)
			
			if delta:
				center = np.array([x,y,z])-delta
			else:
				center = np.array([x,y,z])
					

			M_00 = float(Rotation.getElementsByTagName('M_00')[0].childNodes[0].nodeValue)
			M_01 = float(Rotation.getElementsByTagName('M_01')[0].childNodes[0].nodeValue)
			M_02 = float(Rotation.getElementsByTagName('M_02')[0].childNodes[0].nodeValue)
			M_10 = float(Rotation.getElementsByTagName('M_10')[0].childNodes[0].nodeValue)
			M_11 = float(Rotation.getElementsByTagName('M_11')[0].childNodes[0].nodeValue)
			M_12 = float(Rotation.getElementsByTagName('M_12')[0].childNodes[0].nodeValue)
			M_20 = float(Rotation.getElementsByTagName('M_20')[0].childNodes[0].nodeValue)
			M_21 = float(Rotation.getElementsByTagName('M_21')[0].childNodes[0].nodeValue)
			M_22 = float(Rotation.getElementsByTagName('M_22')[0].childNodes[0].nodeValue)
			
			R = np.array([[M_00, M_01, M_02], [M_10, M_11, M_12],[M_20, M_21, M_22]])

			arg = {'Rotation': R,
				   'Center': center
			}

			args[image_name] = [] if image_name not in args else arg
			args[image_name] = arg
		# print('total images in the xml: ', len(args))
		return {'f':f,
				'K':K,
				'Coord':Coord,
				'Distortion':distortion,
				'args':args,
				'Image_height':ImageHeight,
				'Image_width': ImageWidth}

####################################### masks ##################################################
class border_optimizer:
	def __call__(self, masks):
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
			obj_mask[(yy1, xx1, yy2, xx2)] = skimage.transform.resize(masks[y1:y2+1, x1:x2+1, i], outshape)
			# obj_mask[(y1, x1, y2, x2)] = masks[y1:y2+1, x1:x2+1, i]
		total_masks[os.path.splitext(pkl_file)[0]] = obj_mask

	# total_masks_pkl = os.path.join(path, '..', name)
	# save_f = open(total_masks_pkl,'wb')
	# pkl.dump(total_masks, save_f)
	return total_masks


################################################## geometry  ######################################################################################
def get_bbox(tria_2D):
	tria_2D = np.array(tria_2D)
	[x_left, y_top] = np.min(tria_2D, axis = 0)
	[x_right, y_bottom] = np.max(tria_2D, axis = 0)
	bbox = [x_left, y_top, x_right, y_bottom]
	return bbox

def get_obj_id(bbox, masks, tria_2D):
	[x_left, y_top, x_right, y_bottom] = bbox
	obj_id = -1
	fg_pixels = 0
	for idx, key in enumerate(masks.keys()):
		(y1, x1, y2, x2) = key
		#overlap region
		delta_x =  min(x2, x_right) - max(x1, x_left)
		delta_y =  min(y2, y_bottom) - max(y1, y_top)
		if delta_x> 0  and delta_y>0:
			overlap_region = [max(x1, x_left), max(y1, y_top), min(x2, x_right), min(y2, y_bottom)]
			obj_mask = masks[key]
			overlap = count_fg_pixels_in_tria_2D(overlap_region, tria_2D, obj_mask, key)
			if overlap > fg_pixels:
				fg_pixels = overlap
				obj_id = idx
		else:
			continue
	return obj_id, fg_pixels

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

		# print(direct1,direct2, direct3)
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

def L2_dist(a, b):
	a = np.array(a)
	b = np.array(b)
	return np.sqrt(np.sum((a - b) ** 2))

def compute_overlap(tria_2D, trias, all_pixels, bbox, threshold = 0.5):
	total_overlap_indexes=[]
	[x_left, y_top, x_right, y_bottom] = bbox
	
	# get the inds of trias that have an overlap with the tria_2D
	bboxes = np.array(trias['bbox'])
	delta_x = np.minimum(bboxes[:,2], x_right) - np.maximum(bboxes[:,0], x_left)
	delta_y = np.minimum(bboxes[:,3], y_bottom) - np.maximum(bboxes[:,1], y_top)
	inds_1 = np.where(delta_x > 0)[0]
	inds_2 = np.where(delta_y > 0)[0]
	overlap_inds = np.array(list(set(inds_1).intersection((inds_2))))

	if len(overlap_inds):

		pixles = np.array(trias['pixels'])
		small_inds = np.where(pixles == 0)[0]
		small_inds = list(set(overlap_inds).intersection(set(small_inds)))
		left_inds = np.array(list(set(overlap_inds).difference(set(small_inds))))

		# #### for very small tria_2D which has no pixel inside, directly pick them for distance comparison.
		# if len(small_inds):
		# 	trias_2D = np.array(trias['tria_2D'])[small_inds]
		# 	all_pixles_of_trias = np.array(trias['pixels'])[small_inds]
		# 	N = len(trias_2D)
		# 	count_list = np.zeros(N)

		# 	if all_pixels == 0:
		# 		### small to very smalls ###


		# 	if all_pixels != 0:
		# 		### normal to very smalls ###

		total_overlap_indexes.extend(small_inds)



		##### for not small trias_2D ###############################

		if len(left_inds):
			trias_2D = np.array(trias['tria_2D'])[left_inds]
			all_pixles_of_trias = np.array(trias['pixels'])[left_inds]

			N = len(trias_2D)
			count_list = np.zeros(N)
			overlap = []
			iou = []

			if all_pixels==0:
				points = np.array(tria_2D).astype(np.float)

				gravity = np.round(np.mean(tria_2D,axis = 0)).astype(np.float)
				if not 0 in np.sum((tria_2D-gravity)**2, axis = 1):
					points = np.vstack((points, gravity))
			
				for point in points:
					inds = in_triangle(point+0.5, trias_2D)
					count_list[inds] +=1
				
				all_pixels=len(points)
				overlap = count_list*1.0/np.minimum(all_pixels, all_pixles_of_trias)
				iou = count_list*1.0/(all_pixels + all_pixles_of_trias)
		

			if all_pixels != 0:
				for pi in range(x_left, x_right + 1):
					for pj in range(y_top, y_bottom + 1):
						point = np.array([pi, pj])+0.5
						if in_triangle(point, tria_2D):
							inds = in_triangle(point, trias_2D)
							count_list[inds] +=1		
				overlap = count_list *1.0 /np.minimum(all_pixels, all_pixles_of_trias)
				iou = count_list *1.0 /(all_pixels + all_pixles_of_trias)
		
		
			# # if return many (return those with iou>0.5 and overlap_mini>0.7)
			# inds = list(np.where(overlap>0.7)[0])
			# inds += list(np.where(iou>threshold)[0])
			# inds = np.array(list(set(inds)))

			inds = np.where(overlap>threshold)[0]
			if len(inds):
				overlap_indexes = overlap_inds[inds]

				# combine the small trias and the ovelap>0.5
				total_overlap_indexes.extend(overlap_indexes)
	return np.array(list(set(total_overlap_indexes)))