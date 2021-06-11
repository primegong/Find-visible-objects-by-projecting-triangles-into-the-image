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
				grouped_trais.append(TriasObj)
			pbar.finish()
			
			#############  write...   ################
			if len(grouped_trais):
				print('save the grouped_trais.............')
				self.write_into_obj(grouped_trais, save_path, self.model_name)



	
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

	def write_into_obj(self, results, save_dir, model_name, name = None):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		if type(results) == list:
			for i, triangles in enumerate(results):
				with open(os.path.join(save_dir, model_name[:-4] + '_' + str(i) + '.obj'),'w') as f:
					
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



	
	def check_and_delete_line(self, trias):
	######### detect and delete the triangle that with two same points (looks like a line) ###################
		p1 = trias[:, 0, :]
		p2 = trias[:, 1, :]
		p3 = trias[:, 2, :]

		inds1 = np.where(np.sum(p2-p1, axis = 1) == 0)[0]
		inds2 = np.where(np.sum(p3-p1, axis = 1) == 0)[0]
		inds3 = np.where(np.sum(p3-p2, axis = 1) == 0)[0]
		line_inds = list(set(list(inds1) + list(inds2) + list(inds3)))
		if len(line_inds):
			line_inds = sorted(line_inds, reverse = True)

			trias = trias.tolist()
			for ind in line_inds:
				del(trias[ind])
			trias = np.array(trias)
		return trias, line_inds




	def get_vertices(self, trias):
		N,H,W = trias.shape
		vertices = trias.reshape(-1,W)
		return list(set(map(tuple,vertices)))



	def get_vertice_inds(self, faces, inds):
		faces = np.array(faces)[:,[0,2,4]]
		vertice_inds = faces[inds,:]
		vertice_inds = vertice_inds.flatten()
		vertice_inds = list(set(vertice_inds))
		return vertice_inds



	def angle(self, ray_c1, ray_c2):
		norm1 = np.sqrt(np.sum(ray_c1**2))
		norm2 = np.sqrt(np.sum(ray_c2**2))
		angle = np.dot(ray_c1, ray_c2)/(norm1*norm2)
		return angle

















	def get_principle_normal(self, trias,i):
		######### get the normal of each triangle ###################################
		normals = self.get_normal(trias)

		grouped_normals, index_Ofnormals = Cluster_MultiOutput(normals)

		# return the group with the most normals
		num = [len(group) for group in grouped_normals]
		ind_max = np.argmax(np.array(num))
		principle_normals = np.array(grouped_normals[ind_max])
		principle_normal = principle_normals.mean(0)
		principle_trias = trias[index_Ofnormals[ind_max],:,:]
		principle_gravity = self.get_avg_gravity(principle_trias)
		# print(principle_gravity)

		# ##########  show the principle triangles, principle normals and the whole normals ################
		# # print(principle_normals[0])
		# tuple_normals = list(map(tuple,normals))
		# index = [tuple_normals.index(tuple(d)) for d in principle_normals]
		# principle_trias = trias[index,:,:]
		# f = open('pinciple_triangles.txt','w')
		# trias_str = ['v '+ str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for tria in principle_trias for v in tria]
		# f.writelines(trias_str)
		# for i in range(int(len(principle_trias))):
		# 	f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')

		f = open('normals_' + str(i) + '.txt','w')
		for normal in normals:
			f.write(str(normal[0]) + ' ' + str(normal[1]) + ' ' + str(normal[2]) +'\n')
		f.close()

		f = open('principle_normals_' + str(i) + '.txt','w')
		for normal in principle_normals:
			f.write(str(normal[0]) + ' ' + str(normal[1]) + ' ' + str(normal[2]) + '\n')
		f.close()
		
		return principle_normal, principle_gravity



	def get_normal(self, trias):
		v11 = trias[:, 1, :] - trias[:, 0, :]
		v22 = trias[:, 2, :] - trias[:, 0, :]
		normal = np.cross(v11, v22)
		norm = np.linalg.norm(normal, axis = 1)
		norm = np.tile(norm.reshape(len(norm),1),(1,3))
		normal_unit = normal / norm

		##########  deal with the tria 
		nan_inds = np.where(np.sum(normal_unit - normal_unit,axis = 1) != 0)[0]
		if len(nan_inds):
			print("the %s triangles are lines.   please check the three points of these triangles." %(nan_inds))
			nan_inds = nan_inds.tolist()
			nan_inds = sorted(nan_inds,reverse = True)
			normal_unit = normal_unit.tolist()
			for ind in nan_inds:
				del(normal_unit[ind])
			normal_unit = np.array(normal_unit)
		
		# #####################check if there is nan in normal_unit #################
		# for i, n in enumerate(normal_unit):
		# 	if np.sum(n-n) != 0:
		# 		print(n, i, normal[i], norm[i], trias[i])
	
		return normal_unit

	def get_avg_gravity(self, trais):
		avg_gravity = trais.mean(1).mean(0)
		return avg_gravity





	