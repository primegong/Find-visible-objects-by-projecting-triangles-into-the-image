from progressbar import *
import pickle as pkl
import numpy as np
from read_data.OBJ_PARSER import *
from cluster.cluster import Cluster_MultiOutput
from functools import reduce





class Union:
	def __init__(self, input_segments_path, output_path, model_name):
		self.input = input_segments_path
		self.output = output_path
		self.model_name = model_name
		self.same_side_path = os.path.join(output_path,'same_side')
		self.opposite_side_path = os.path.join(output_path,'opposite_side')
	
	def union_segments_3D(self):
		print('union fragments in the same side.............')
		self.union_with_condition(self.input, function = self.has_common_triangles, save_path = self.same_side_path)
		
		print('union fragments in the opposite side.............')
		self.union_with_condition(self.same_side_path, function = self.is_the_opposite_side, save_path = self.opposite_side_path)


	def union_with_condition(self, input_path, function, save_path):
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		if len(os.listdir(save_path)) == 0:
			######### 1th step: union the segments in the same side  #############
			trias_of_all_piers = self.get_trias_of_all_piers(input_path)
			pbar = ProgressBar().start()
	
			N = len(trias_of_all_piers)
			grouped_trais = []
			groups = []
			used = []
			for i in range(N): 
				pbar.update(i*1.0/N*100)
				
				if i in used:
					continue 

				if i == N-1:
					group = [i]
					groups.append(group)
					grouped_trais.append(trias_of_all_piers[i])
					break
				 
				group = [i]
				used.append(i)
				TriasObj = trias_of_all_piers[i]
				
				while True:
					
					temp = group.copy()
					
					for j in range(i+1,N):
						if j in used:
							continue                    
						
						TriasSearch  = trias_of_all_piers[j]


						flag, TriasObj_union = function(TriasObj, TriasSearch)
						if flag:
							####  use the new combined TrisObj to search in next iteration ###########
							TriasObj = TriasObj_union
							group.append(j)
							used.append(j)
					

					diff = list(set(group).difference((set(temp))))
					if len(diff) == 0:
						#### if no more added in, then break #####
						break
				
				groups.append(group)
				TriasObj = np.array(list(map(list,TriasObj)))
				if len(TriasObj)>20:
					grouped_trais.append(TriasObj)
			pbar.finish()
			
			#############  write...   ################
			if len(grouped_trais):
				print('save into obj file.............')
				for i, group in enumerate(grouped_trais):
					self.write_into_obj(group, os.path.join(save_path,  str(i) + '.obj'))



	
	def has_common_triangles(self, TriasObj , TriasSearch, thresh = 0.5):
		'''
		### 三角形重叠度大于0.5，即合并为一个新的整体 ###
		union: True or False
		TriasObj: 合并后的三角形

		'''
		n_obj = len(TriasObj)
		TriasObj = TriasObj.reshape(n_obj, -1)
		TriasObj = list(map(tuple,TriasObj))

		n_search = len(TriasSearch)
		TriasSearch = TriasSearch.reshape(n_search, -1)
		TriasSearch = list(map(tuple,TriasSearch))

		common_clouds = len(list(set(TriasObj).intersection(set(TriasSearch))))
		minimum = min(n_obj, n_search)
		total =  n_obj + n_search
		union = False
		ratio = common_clouds *1.0/minimum 
		iou = common_clouds*1.0/total
		if ratio>= thresh or iou>=0.5:
			union = True

			TriasObj = list(set(TriasObj).union(set(TriasSearch)))
			TriasObj = list(map(list, TriasObj))
			TriasObj = np.array(TriasObj).reshape(-1,3,3)
		return union, TriasObj


	def is_the_opposite_side(self, trias_i, trias_j):
		### 先判断空间box是否重合
		range_i = self.get_bbox3D((trias_i))
		range_j = self.get_bbox3D((trias_j))

		union = False
		delta, overlap_ratio = self.get_bbox_overlap(range_i, range_j) 
		(delta_x, delta_y, delta_z) = delta

		if delta_x > 0 and delta_y >0 and delta_z > 0:
			#### if have overlap region ##############
			if overlap_ratio>=0.5:
				union = True


			# if not union:
			# 	#################### method 1： 法向量夹角接近180 #####################
			# 	#cross over point: t
			# 	t = np.dot(normal_i, gravity_i - gravity_j)/np.dot(normal_i, normal_j)
			# 	tx, ty, tz = normal_j * t + gravity_j

			# 	x_min, y_min, z_min, x_max, y_max, z_max = range_i
			# 	if tx > x_min and tx < x_max and ty > y_min and ty < y_max and tz > z_min and tz < z_max:
			# 		union  = True
			# 		#calculat the distance between (tx,ty,tz) and the gravity_i
			# 		distance = np.sum((np.array([tx, ty, tz]) - np.array(gravity_i))**2)


			if not union:
				#################### method 2： 如果三维包围盒重叠度小于0.5， 判断投影重叠长 #####################
				vertice_i = trias_i.reshape(-1,3)[:,[0,1]]
				vertice_j = trias_j.reshape(-1,3)[:,[0,1]]

				## 拟合局部坐标轴，计算斜率和截距
				[k,b] = np.polyfit(list(vertice_i[:,0]), list(vertice_i[:,1]), 1)
				theta = np.arctan(k)
				# 构建旋转矩阵
				rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], 
									[-np.sin(theta), np.cos(theta)]])
				# 坐标原点为目标重心
				translation = vertice_i.mean(0)

				#转换到局部坐标轴上。两个目标所有点都投影到局部坐标轴上
				trans_vi = vertice_i.transpose()
				trans_vj = vertice_j.transpose()
				rotate_vi = np.dot(rotation_matrix, trans_vi - translation.reshape(-1,1))
				rotate_vj = np.dot(rotation_matrix, trans_vj - translation.reshape(-1,1))
				rotate_vi = rotate_vi.transpose()
				rotate_vj = rotate_vj.transpose()

				proj_xi_min = min(rotate_vi[:,0])
				proj_xi_max = max(rotate_vi[:,0])
				proj_xj_min = min(rotate_vj[:,0])
				proj_xj_max = max(rotate_vj[:,0])

				#投影重叠长
				overlap_x_min = max(proj_xi_min, proj_xj_min)
				overlap_x_max = min(proj_xi_max, proj_xj_max)
				proj_len = overlap_x_max - overlap_x_min
				ratio = proj_len *1.0/(proj_xj_max - proj_xj_min + proj_xi_max - proj_xi_min - proj_len)
				if ratio >= 0.5:
					union  = True

		union_results = []
		if union:
			union_results = np.vstack((trias_i, trias_j))
			num_trias = len(union_results)
			union_results = union_results.reshape(num_trias, -1)

			union_results = list(map(tuple, union_results))
			union_results = list(set(union_results))
			union_results = list(map(list, union_results))
			union_results = np.array(union_results).reshape(-1,3,3)

		return union, union_results


	def get_trias_of_all_piers(self, input_path):	
		### read vertices from all obj files ##############################
		trias_of_all_piers = []
		obj_lists = os.listdir(input_path)
		print('read files, total num of files: '+ str(int(len(obj_lists))))
	
		for obj_file in obj_lists:
			if os.path.splitext(obj_file)[-1] != '.obj':
				continue
			obj_parser = OBJ_PARSER(os.path.join(input_path, obj_file))
			triangles, faces, vertices = obj_parser.get_triangles()
			trias_of_all_piers.append(triangles)
			
		return trias_of_all_piers


	def write_into_obj(self, triangles, save_file):
		with open(save_file,'w') as f:
			
			trias_str = ['v '+str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for triangle in triangles for v in triangle]
			f.writelines(trias_str)

			trias_num = int(len(triangles))
			for i in range(trias_num):
				f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')

	def get_bbox3D(self,trias):
		xs = trias[:,:,0]
		ys = trias[:,:,1]
		zs = trias[:,:,2]
		x_min = min(xs.flatten())
		y_min = min(ys.flatten())
		z_min = min(zs.flatten())
		x_max = max(xs.flatten())
		y_max = max(ys.flatten())
		z_max = max(zs.flatten())
		bbox3D = [x_min, y_min, z_min, x_max, y_max, z_max]
		return bbox3D


	def get_bbox_overlap(self, range_i, range_j):
		overlap_x_min = max(range_i[0], range_j[0])
		overlap_y_min = max(range_i[1], range_j[1])
		overlap_z_min = max(range_i[2], range_j[2])
		overlap_x_max = min(range_i[3], range_j[3])
		overlap_y_max = min(range_i[4], range_j[4])
		overlap_z_max = min(range_i[5], range_j[5])
		
		delta_x = overlap_x_max - overlap_x_min 
		delta_y = overlap_y_max - overlap_y_min
		delta_z = overlap_z_max - overlap_z_min


		bbox_overlap = delta_x * delta_y *delta_z

		bbox_i = np.array(range_i).reshape((2,3))
		area_i = reduce(lambda x,y : x*y, bbox_i[1,:]-bbox_i[0,:])
		bbox_j = np.array(range_j).reshape((2,3))
		area_j = reduce(lambda x,y : x*y, bbox_j[1,:]-bbox_j[0,:])
		overlap_ratio = bbox_overlap/min(area_i, area_j)


		return (delta_x, delta_y, delta_z), overlap_ratio



	

