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

		#从最密集的区域开始
		trias_gravity = triangles.mean(1)
		R, tree, neighbor_Radius = get_R(trias_gravity, point_num = 21)
		sorted_inds = sorted(range(len(neighbor_Radius)), key=lambda k: neighbor_Radius[k])
		
		collections = []
		used = []
		for target in sorted_inds:
			if target in used:
				continue
			collection = self.find_connect_trias(target, group_faces)
			collections.append(collection)
			used = list(set(used).union(set(collection)))

		###### the main body(base) that with most triangles ##############
		num = [len(c) for c in collections]
		ind_max = np.argmax(np.array(num))
		
		###### initialize with the main body and find those have overlap with it #############
		out_index = np.array(indexes[ind_max])
		out_trias = triangles[out_index,:,:]
		return out_trias, out_index


	def find_connect_trias(self, targetID, group_faces):
		
		collection = [targetID]
		target_used = []
		while True:
			[v1,_,v2,_,v3,_] = group_faces[targetID,:]
			target_used.append(targetID)

			#####1. connectivity ###################
			index1 = np.where(group_faces == v1)[0].tolist()
			index2 = np.where(group_faces == v2)[0].tolist()
			index3 = np.where(group_faces == v3)[0].tolist()
			#common edges require two vertices:
			index_dict = Counter(index1+index2+index3)
			finds = []
			for key, value in index_dict.items():
				if value == 2:
					finds.append(key)

			##### 2. adaptive-threshold-research ########
			#搜索距离为R内的三角形
			# ind_search = tree.query_ball_point(dataset[targetID,:], r = R)

			####3. 将找到的三角形添加到结果中
			if len(finds):
				collection = list(set(collection).union(set(finds)))

			if len(collection) == N:
				break

			diff = list(set(collection).difference(set(target_used)))
			if len(diff):
				targetID = diff[0]
			else:
				break

		return collection



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

	

	def write_into_obj(self, segment_file, triangles, save_dir):

		with open(os.path.join(save_dir, segment_file),'w') as f:
			trias_str = ['v '+str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for triangle in triangles for v in triangle]
			f.writelines(trias_str)
			for i in range(int(len(triangles))):
				f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')

