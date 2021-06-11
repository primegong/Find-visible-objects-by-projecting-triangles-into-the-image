import os,  shutil
import numpy as np
import copy, time
import pickle as pkl
import random
from scipy import spatial
from progressbar import *
from cluster.cluster import Cluster_MultiOutput, get_R, get_R2
from functools import reduce
from collections import Counter
from read_data.OBJ_PARSER import *

class Optimization:

	def __init__(self, input_segments_path, output_path):
		self.input = input_segments_path
		self.output = output_path

	def optimize(self):
		save_dir = self.output
		fragments_list = os.listdir(self.input)
		fragments_list = [fragment for fragment in fragments_list if os.path.splitext(fragment)[-1] == '.obj']

		pbar = ProgressBar().start()
		N = len(fragments_list)
		for j, fragment_file in enumerate(fragments_list):

			if os.path.exists(os.path.join(save_dir, fragment_file)):
				continue
			# print(fragment_file)
			pbar.update(j*1.0/N*100)
			
			obj_file = os.path.join(self.input, fragment_file)
			obj_parser = OBJ_PARSER(obj_file)
			triangles, faces, vertices = obj_parser.get_triangles()

			index_InGroup = []
			opt_trias, index_InGroup = self.filter_noise(triangles, np.array(faces))

			# opt_group, opt_trias, index_InGroup = self.fileter_visible_noises(triangles, faces)
			# left_group, left_trias, left_InGroup = self.Cluster_SingleOutput(opt_group, triangles)
			# index_InGroup = np.array(index_InGroup)[left_InGroup]

			if len(index_InGroup) != 0:
				# save
				print('writing to obj...')
				self.write_into_obj(fragment_file, opt_trias, save_dir)

		pbar.finish()
	
	
	
	def filter_noise(self, triangles, group_faces):
		'''
		根据空间连通性进行聚类，从一个三角形开始，根据是否存在公共边进行生长，直到所有的三角形都分入不同的组。
		保留三角形数目最多的组作为主体。
		group_faces = np.array([[1,2,3],[3,4,5],...])
		'''
		#用重心坐标构树，搜索最近的11个点，然后判断是否存在公共边，对存在公共边的三角形再搜索最近20个点，循环上述操作。
		trias_gravity = triangles.mean(1)
		dataset = trias_gravity
		collections = []
		indexes = []
		used = []
		N = len(dataset)
		R, tree, neighbor_Radius = get_R(dataset, point_num = 21)
		#从最密集的区域开始
		sorted_inds = sorted(range(len(neighbor_Radius)), key=lambda k: neighbor_Radius[k])
		# R, tree = get_R2(group_faces, dataset)

		for i in sorted_inds:
			target = i
			if target in used:
				continue

			collection = [target]
			target_used = []
			inds=[]
			while True:
				[v1,_,v2,_,v3,_] = group_faces[target,:]
				target_used.append(target)

				#####1. connectivity ###################
				index1 = np.where(group_faces == v1)[0].tolist()
				index2 = np.where(group_faces == v2)[0].tolist()
				index3 = np.where(group_faces == v3)[0].tolist()

				#common edges require two vertices:
				index_dict = Counter(index1+index2+index3)

				ind = []
				for key, value in index_dict.items():
					if value == 2:
						ind.append(key)

				##### 2. adaptive-threshold-research ########
				#搜索距离为R内的三角形
				# ind_search = tree.query_ball_point(dataset[target,:], r = R)

				####3. 将找到的三角形添加到结果中
				# ind = list(set(ind).union(set(ind_search)))

				finds =ind
				if len(finds):
					collection = list(set(collection).union(set(finds)))
					inds = list(set(inds).union(set(ind)))
				
				if len(collection) == N:
					break

				diff = list(set(collection).difference(set(target_used)))
				if len(diff):
					target = diff[0]
				else:
					break
			
			collections.append(collection)
			indexes.append(inds)
			used = list(set(used).union(set(collection)))

		
		###### the main body(base) that with most triangles ##############
		num = [len(c) for c in collections]
		ind_max = np.argmax(np.array(num))
		
		###### initialize with the main body and find those have overlap with it #############
		out_index = np.array(indexes[ind_max])
		out_trias = triangles[out_index,:,:]
		return out_trias, out_index


	def find_connect_trias(self, face, group_faces):
		iteration  = 5
		indexes = []

		for j in range(iteration):
			collection = [face]
			ind = []
			for item in collection:
		

				[v1,v2,v3] = item

				#####1. #common edges require two vertices: ###################
				ind1 = np.where(group_faces == v1)[0].tolist()
				ind2 = np.where(group_faces == v2)[0].tolist()
				ind3 = np.where(group_faces == v3)[0].tolist()
				
				ind_dict = Counter(ind1+ind2+ind3)
			
				for key, value in ind_dict.items():
					if value == 2:
						ind.append(key)
			if len(ind):
				collection = group_faces[ind,:]

				indexes.extend(ind)
			else:
				break

		return indexes


	def fileter_visible_noises(self, group_trias, group_faces):
		''' 
		group: [j1,j2,j3...], j is the index of triangle
		triangles: total triangle
		faces: NX3 numpy

		'''

		
		v1 = group_trias[:,1,:] - group_trias[:,0,:]
		v2 = group_trias[:,2,:] - group_trias[:,0,:]

		normals = [np.cross(vv1, vv2) for (vv1,vv2) in list(zip(v1,v2))]
		normals = np.array(normals)
		norm = np.linalg.norm(normals, axis = 1).reshape(-1,1)
		norm = np.tile(norm,(1,3))
		normals = normals/norm
	

		local_normals = []
		for i, face in enumerate(group_faces):

			neighbor_normals = []
			find_ind = self.find_connect_trias(face, group_faces)
			if len(find_ind):
				neighbor_normals = np.array(normals[find_ind,:])
				local_normals.append(neighbor_normals.mean(0))
			else:
				# print('isolated',i)
				###### if islolated, use itself ##########
				local_normals.append(normals[i,:])

		local_normals = np.array(local_normals)



		N = len(local_normals)
		used = []
		collections = []
		indexes = []
		for i in range(N):
			target = i
			if target in used:
				continue

			collection = []
			target_used = []
			inds=[]
			while True:

				[v1,v2,v3] = group_faces[target]
				target_used.append(target)


				#####1. connectivity ###################
				ind1 = np.where(group_faces == v1)[0].tolist()
				ind2 = np.where(group_faces == v2)[0].tolist()
				ind3 = np.where(group_faces == v3)[0].tolist()
				#common edges require two vertices:

				ind_dict = Counter(ind1+ind2+ind3)
				ind = []
				for key, value in ind_dict.items():
					if value == 2:
						ind.append(key)


				# ##### 2. adaptive-threshold-research ########
				# ind_search = tree.query_ball_point(triangles[target].mean(0), r = R)
				# ind = list(set(ind).union(set(ind_search)))


				# #### 3. 
				if len(ind):
					num = len(ind)
					target_normal = local_normals[target,:]
					temp_normals = local_normals[ind,:]
					target_gravity = group_trias[target,:,:].mean(0)
					temp_gravitys = group_trias[ind,:,:].mean(1)

					cline =  np.tile(target_gravity,(num, 1)) - temp_gravitys 
					norm = np.linalg.norm(cline, axis = 1).reshape(-1,1)
					cline_unit = cline / np.tile(norm, (1,3))
	
					angle_cos1 = np.dot(-cline_unit, target_normal)
					angle_cos2 = []

					for t in range(num):
						cos2 = np.dot(cline_unit[t,:], temp_normals[t,:])
						angle_cos2.append(cos2)
					angle_cos2 = np.array(angle_cos2)
		
					min_angle = np.where(angle_cos1 < angle_cos2, angle_cos1, angle_cos2)
	

					ind4 = np.where(min_angle < 0.5)[0]
					if len(ind4):
						ind = [ind[idx] for idx in ind4]
						finds = ind
					else: 
						finds = []


					if len(finds):
						collection = list(set(collection).union(set(finds)))
						inds = list(set(inds).union(set(ind)))
					
				if len(collection) == N:
					break

				diff = list(set(collection).difference(set(target_used)))
				if len(diff):
					target = diff[0]
				else:
					break
			
			if len(collection)>20:
				collections.append(collection)
				indexes.append(inds)

			used = list(set(used).union(set(collection)))

		num = [len(c) for c in collections]
		ind_max = np.argmax(np.array(num))
		###### initialize with the main body and find those have overlap with it #############
		out_index = np.array(indexes[ind_max])
		out_trias = group_trias[out_group,:,:]

		return  out_trias, out_index



	def grouping_using_normal(self, group_trias, group_faces):
		v1 = group_trias[:,1,:] - group_trias[:,0,:]
		v2 = group_trias[:,2,:] - group_trias[:,0,:]

		normals = [np.cross(vv1, vv2) for (vv1,vv2) in list(zip(v1,v2))]
		normals = np.array(normals)
		norm = np.linalg.norm(normals, axis = 1).reshape(-1,1)
		norm = np.tile(norm,(1,3))
		normals = normals/norm

		local_entroy = []
		for i, face in enumerate(group_faces):

			neighbor_normals = []
			find_ind = self.find_connect_trias(face, group_faces)
			if len(find_ind):
				neighbor_normals = np.array(normals[find_ind,:])

				traget_normal = normals[i,:]
				cos_dis = 1 - np.cross(target_normal, neighbor_normals)
				local_entroy.append(sum(cos_dis))
			else:
				# print('isolated',i)
				###### if islolated, use itself ##########
				local_entroy.append(np.inf)

		local_entroy = np.array(local_entroy)

	



	def filter_after_union(self, triangles, group_faces):

		####group = [j1,j2,j3,.....]
		#group_faces = np.array([[1,2,3],[3,4,5],...])

		trias_gravity = triangles.mean(1)

		v1 = triangles[:,1,:] - triangles[:,0,:]
		v2 = triangles[:,2,:] - triangles[:,0,:]

		normals = [np.cross(vv1, vv2) for (vv1,vv2) in list(zip(v1,v2))]
		normals = np.array(normals)
		norm = np.linalg.norm(normals, axis = 1).reshape(-1,1)
		norm = np.tile(norm,(1,3)) + 0.000000001
		normals = normals/norm
		# print(norm)

		dataset = normals

		
		collections = []
		indexes = []
		used = []
		N = len(dataset)
		R, tree, neighbor_R = get_R(dataset, point_num = 21)
		sorted_inds = sorted(range(len(neighbor_R)), key=lambda k: neighbor_R[k])
		# R, tree = get_R2(group_faces, dataset)

		for i in sorted_inds:
			
			target = i
			if target in used:
				continue

			collection = []
			target_used = []
			inds=[]
			while True:

				[v1,v2,v3] = group_faces[target,:]
				target_used.append(target)


				# ####1. connectivity ###################
				# ind1 = np.where(group_faces == v1)[0].tolist()
				# ind2 = np.where(group_faces == v2)[0].tolist()
				# ind3 = np.where(group_faces == v3)[0].tolist()
				# #common edges require two vertices:

				# ind_dict = Counter(ind1+ind2+ind3)
				# ind = []
				# for key, value in ind_dict.items():
				# 	if value == 2:
				# 		ind.append(key)


				##### 2. adaptive-threshold-research ########
				ind_search = tree.query_ball_point(dataset[target,:], r = R)
				# ind = list(set(ind).intersection(set(ind_search)))
				ind = ind_search


				finds = ind
				if len(finds):
					collection = list(set(collection).union(set(finds)))
					inds = list(set(inds).union(set(ind)))
				
				if len(collection) == N:
					break

				diff = list(set(collection).difference(set(target_used)))
				if len(diff):
					target = diff[0]
				else:
					break
			
			if len(collection)>20:
				collections.append(collection)
				indexes.append(inds)

			used = list(set(used).union(set(collection)))

		

		# ###### the main body(base) that with most triangles ##############
		# num = [len(c) for c in collections]
		# ind_max = np.argmax(np.array(num))
		# ###### initialize with the main body and find those have overlap with it #############
		# out_index = np.array(indexes[ind_max])
		# out_trias = triangles[out_group,:,:]

		num = [len(c) for c in collections]
		
		print(num)
		sorted_inds = sorted(range(len(num)), key=lambda k: num[k])
		keep = sorted_inds[:len(num)]

		# threshold = max(num)/10
		# keep = np.where(num>threshold)
		out_index = []
		out_group = []
		out_trias = []
		for ind in keep:

			if len(out_index) == 0:
				out_index = indexes[ind]
				out_trias = triangles[out_index,:,:]
			else:
				out_index.extend(indexes[ind])
				out_trias = np.vstack((out_trias, triangles[out_index,:,:]))



		
		# range_base = self.get_range(out_trias)
		# N = len(num)
		
		# for j in range(N):
		# 	if j == ind_max:
		# 		continue
		# 	trias_j = triangles[np.array(collections[j]),:,:]
		# 	range_j = self.get_range(trias_j)
			
		# 	delta, overlap_ratio, overlap_bbox = self.get_bbox_overlap(range_base, range_j)
		# 	(delta_x, delta_y, delta_z) = delta
			
		# 	if delta_x > 0 and delta_y > 0 and delta_z > 0:
		# 		sign_base = self.has_trias_in_bbox(out_trias, overlap_bbox)
		# 		sign_j = self.has_trias_in_bbox(trias_j, overlap_bbox)
		# 		if sign_base and sign_j:
		# 			out_group = np.concatenate((out_group, np.array(collections[j])))
		# 			out_index = np.concatenate((out_index, np.array(indexes[j])))
		# 			out_trias = np.vstack((out_trias, triangles[np.array(collections[j]),:,:]))

		return out_trias, out_index





	def write_into_obj(self, segment_file, triangles, save_dir):

		with open(os.path.join(save_dir, segment_file),'w') as f:
			trias_str = ['v '+str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for triangle in triangles for v in triangle]
			f.writelines(trias_str)
			for i in range(int(len(triangles))):
				f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')



	def get_avg_gravity(self, trais):
		avg_gravity = trais.mean(1).mean(0)
		return avg_gravity



	def Cluster_SingleOutput(self, group, triangles, center):
		#计算该组所有三角形的重心
		trias = triangles[group,:,:]
		gravitys = trias.mean(1)
		grouped_trias, index_OfTrias = Cluster_MultiOutput(gravitys)
		print('grouped_trias:', len(grouped_trias))


		# ############################ restore the main body with most triangles ##############
		num = [len(group) for group in grouped_trias]
	
		#### initialization ######################
		ind_max = np.argmax(np.array(num))
		out_trias = np.array(grouped_trias[ind_max])
		out_index = np.array(index_OfTrias[ind_max])
		out_group = np.array(group)[out_index]
		
		
		# ############################ restore the group that neareat to the camera center ##############
		# out_trias, out_index = [], []
		# if len(index_OfTrias) == 0:
		# 	out_trias, out_index = [], []
		# elif len(index_OfTrias) == 1:
		# 	out_trias = grouped_trias[0]
		# 	out_index = index_OfTrias[0]
		# else:
		# 	min_dis = float("inf")
		# 	# print(grouped_trias,'##original')
		# 	for i, trias in enumerate(grouped_trias):
		# 		# print(i)
	
		# 		gravity = np.array(trias).mean(1).mean(0)
		# 		s_2 = np.sum((center-gravity)**2)

		# 		if s_2 < min_dis:
		# 			min_dis = s_2
		# 			out_index = index_OfTrias[i]
		# 			out_trias = grouped_trias[i]
		return out_group, out_trias, out_index
	


	
	def get_range(self,trias):
		xs = trias[:,:,0]
		ys = trias[:,:,1]
		zs = trias[:,:,2]
		x_min = min(xs.flatten())
		y_min = min(ys.flatten())
		z_min = min(zs.flatten())
		x_max = max(xs.flatten())
		y_max = max(ys.flatten())
		z_max = max(zs.flatten())
		ranges = [x_min, y_min, z_min, x_max, y_max, z_max]
		return ranges



	def get_bbox_overlap(self, range_i, range_j):
		overlap_x_min = max(range_i[0], range_j[0])
		overlap_y_min = max(range_i[1], range_j[1])
		overlap_z_min = max(range_i[2], range_j[2])
		overlap_x_max = min(range_i[3], range_j[3])
		overlap_y_max = min(range_i[4], range_j[4])
		overlap_z_max = min(range_i[5], range_j[5])
		overlap_range = [overlap_x_min, overlap_y_min, overlap_z_min, overlap_x_max, overlap_y_max, overlap_z_max]
		
		delta_x = overlap_x_max - overlap_x_min 
		delta_y = overlap_y_max - overlap_y_min
		delta_z = overlap_z_max - overlap_z_min

		bbox_overlap = delta_x * delta_y *delta_z

		bbox_i = np.array(range_i).reshape((2,3))
		area_i = reduce(lambda x,y : x*y, bbox_i[1,:]-bbox_i[0,:])
		bbox_j = np.array(range_j).reshape((2,3))
		area_j = reduce(lambda x,y : x*y, bbox_j[1,:]-bbox_j[0,:])
		overlap_ratio = bbox_overlap/min(area_i, area_j)


		return (delta_x, delta_y, delta_z), overlap_ratio, overlap_range



	def has_trias_in_bbox(self, trias, bbox):
		[x_min, y_min, z_min, x_max, y_max, z_max] = bbox
		xs = trias[:,:,0]
		ys = trias[:,:,1]
		zs = trias[:,:,2]

		x_1 = np.where(xs >= x_min)[0]
		x_2 = np.where(xs <= x_max)[0]
		y_1 = np.where(ys >= y_min)[0]
		y_2 = np.where(ys <= y_max)[0]
		z_1 = np.where(zs >= z_min)[0]
		z_2 = np.where(zs <= z_max)[0]

		x_ = list(set(x_1).intersection(set(x_2)))
		y_ = list(set(y_1).intersection(set(y_2)))
		z_ = list(set(z_1).intersection(set(z_2)))
		xy = list(set(x_).intersection(set(y_)))
		xyz = list(set(xy).intersection(set(z_)))

		sign = len(xyz)
		return sign

	