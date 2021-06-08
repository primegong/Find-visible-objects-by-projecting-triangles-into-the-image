from progressbar import *
import pickle as pkl
import numpy as np
from cluster.cluster import Cluster_MultiOutput
from functools import reduce


class Union:
	def __init__(self, input_segments_path, output_path):
		self.input = input_segments_path
		self.output = output_path


	def union_segments_3D(self, triangles, faces, model_name):
	###################### . combine masks those has lots of common clouds ###################
		
		save_dir = self.output
		segments_3D_path = self.input

		if os.path.exists(os.path.join(save_dir, 'union_results.pkl')):
			print('load union_results from pkl files...')
			f = open(os.path.join(save_dir, 'union_results.pkl'), 'rb')
			f.seek(0)
			union_results = pkl.load(f)
			f.close()
			return union_results
		else:
			pbar = ProgressBar().start()
			adict = {}
			segments_list = os.listdir(segments_3D_path)
			segments_list = [segment for segment in segments_list if os.path.splitext(segment)[-1] == '.pkl' and '3D_Segments' not in segment]
			for segment_file in segments_list:
				f = open(os.path.join(segments_3D_path, segment_file),'rb')
				f.seek(0)
				relation = pkl.load(f)
				key = segment_file[:-4]
				adict[key] = relation


			######################################################################################
			########## 1th step: union the segments in the same side  ##############################
			print('union the segments in the same side..........')
			segment_name_list = list(adict.keys())

			# segment_name_list = ['1208-1_DSC00412_24_0', '1208-1_DSC00398_49_3', '1207-1_DSC00525_24_4', '1207-1_DSC00525_23_5']
			
			N = len(segment_name_list)
			groups = []
			used = []
			for i in range(N): 
				pbar.update(i*1.0/N*100)
				TargetIndex = segment_name_list[i]
				if TargetIndex in used:
					continue 
				if i == N-1:
					group = [TargetIndex]
					groups.append(group)
					break
				 
				group = [TargetIndex]
				used.append(TargetIndex)
				
				# normal_i = normals[TargetIndex]
				# gravity_i = gravitys[TargetIndex]
				TrisObj = adict[TargetIndex]['id'] 
				while TargetIndex:
					for j in range(i+1,N):
						SearchIndex = segment_name_list[j]
						if SearchIndex in used:
							continue                    
						
						TrisSearch  = adict[SearchIndex]['id']
						# normal_j = normals[SearchIndex]
						# gravity_j = gravitys[SearchIndex]
						if self.has_common_clouds(TrisObj, TrisSearch):
							#### use the new avg normal and gravity #############
							# normal_i = (normal_i + normal_j)/2
							# gravity_i = (gravity_i + gravity_j)/2
							TrisObj = list(set(TrisObj).union(set(TrisSearch)))
							group.append(SearchIndex)
							used.append(SearchIndex)
					####next: use the new growed TrisObj to search again ###########
					if group.index(TargetIndex) == len(group)-1:
						#### if no more added in, then break #####
						break
					else:
						TargetIndex = group[len(group)-1]
				groups.append(group)
			pbar.finish()
			
			########2th step: combine the j, gravity, normal in the same group, group = [segment_name1, segment_name2,....]#####
			print('get union_inds_OfTriangle, union_gravity and principle normal in each group......')
			union_inds_OfTriangle = []
			union_gravitys = []
			union_normals = []
			union_inds_OfVertice = []
			union_range = []

			pbar = ProgressBar().start()
			N = len(groups)
			for i, group in enumerate(groups):
				pbar.update(i*1.0/N*100)

				#### inds ##############
				inds = [] 
				for segment_name in group:
					inds = inds + list(np.array(adict[segment_name]['id']).astype(np.int))
				inds = list(set(inds))

				########## vertices #############
				vertice_inds = self.get_vertice_inds(faces, inds)

				########## range ##################
				trias = triangles[inds,:,:]
				ranges = self.get_range(trias)

				####### check and delete lines #########################
				trias, line_inds = self.check_and_delete_line(trias)
				# for ind in inds:
				# 	del(inds[ind])

				### principle_normal , principle_gravity ################				
				principle_normal, principle_gravity = self.get_principle_normal(trias,i)
				principle_gravity = self.get_avg_gravity(trias)

				###### save ###########################################
				union_inds_OfTriangle.append(inds)
				union_inds_OfVertice.append(vertice_inds)
				union_range.append(ranges)
				union_normals.append(principle_normal)
				union_gravitys.append(principle_gravity)
			pbar.finish()
		

			#write...################
			if len(union_inds_OfTriangle):
				print('save the union_inds_OfTriangle.............')
				path = os.path.join(save_dir,'same_side')
				if not os.path.exists(path):
					os.makedirs(path)
				self.write_into_obj(union_inds_OfTriangle, triangles, path, model_name)


            ############################################################################
			######### 3th step: union the segments in two opposite sides  #############
			print('union the segments in two opposite sides........')
			N = len(union_inds_OfTriangle)
			groups = []
			used = []
			for i in range(N): 
				pbar.update(i*1.0/N*100)
				TargetIndex = i
				if TargetIndex in used:
					continue 
				if i == N-1:
					group = [TargetIndex]
					groups.append(group)
					break

				# print('TargetIndex:', TargetIndex,'##############')
				group = [TargetIndex]
				target_used = []
				while True:
					normal_i = union_normals[TargetIndex]
					gravity_i = union_gravitys[TargetIndex]
					inds_i = union_inds_OfTriangle[TargetIndex]
					vertice_i = union_inds_OfVertice[TargetIndex]
					range_i = union_range[TargetIndex]
					
		
					for j in range(i+1,N):
						if j in used:
							continue                    
						normal_j = union_normals[j]
						gravity_j = union_gravitys[j]
						inds_j = union_inds_OfTriangle[j]
						vertice_j = union_inds_OfVertice[j]
						range_j = union_range[j]
						flag, distance = self.is_the_opposite_side(normal_i, normal_j, gravity_i, gravity_j, vertice_i, vertice_j, range_i, range_j)

						if flag:
							group.append(j)
							used.append(j)
							
					group = list(set(group))
					#next
					target_used.append(TargetIndex)
					diff = list(set(group).difference(set(target_used)))
					if len(diff):
						TargetIndex = diff[0]
					else:
						break
				# print('group:', group)
				groups.append(group)
			
			############## combine the j in the same group, groups = [[i1, i3, i5....],[i2, i4,...],...]##########
			union_results = []
			for group in groups:
				result = []
				for i in group:
					result += union_inds_OfTriangle[i]
				result = list(set(result))
				union_results.append(result)


			############################# 4th step: save #######################################
			if len(union_results):
				print('save the union results.............')
				path = os.path.join(save_dir,'opposite_side')
				if not os.path.exists(path):
					os.makedirs(path)
				self.write_into_obj(union_results, triangles, path, model_name)
		
			return union_results
	
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

	def has_common_clouds(self, TrisObj , TrisSearch, thresh = 0.7):
		common_clouds = len(list(set(TrisObj).intersection(set(TrisSearch))))
		minimum = min(len(TrisObj),len(TrisSearch))
		total =  len(TrisObj) + len(TrisSearch)
		union = False
		ratio = common_clouds *1.0/minimum 
		iou = common_clouds*1.0/total
		if ratio>= thresh or iou>=0.5:
			union = True
		return union

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


		
	def is_the_opposite_side(self, normal_i, normal_j, gravity_i, gravity_j, vertice_i, vertice_j, range_i, range_j, wall_width = 0.05):
		union = False
		distance = 0
		
		# if self.angle(normal_i, normal_j) >= 0:
		# 	return union, distance

		
		# if len(list(set(inds_i).intersection(set(inds_j)))):
		# # if i and j have the same triangles, then do:
		
		# if len(list(set(vertice_i).intersection(set(vertice_j)))):
		# 	#if i and j have the same points, then do:

		# print(TargetIndex, j)
		
		delta, overlap_ratio = self.get_bbox_overlap(range_i, range_j) 
		(delta_x, delta_y, delta_z) = delta
		if delta_x > 0 and delta_y >0 and delta_z > 0:
		########## if there are overlap between segment i and segment j in 3D space, then do:
			# print(TargetIndex, j)

			# ##############  method 1 ##############################
			# angle = self.angle(normal_i, normal_j)
			# gravity_direct = gravity_i - gravity_j
			# gravity_distance = np.sum(gravity_direct**2)
			# angle_i = self.angle(gravity_direct, normal_i)
			# angle_j = self.angle(gravity_direct, normal_j)
			# print(angle, angle_i, angle_j)
			# if angle<-0.7 and angle_i>0.5 and angle_j<-0.5:
			# 	union = True
			# print(union)

			

			################# method 3 ########################
			
			if overlap_ratio>=0.5:
				union = True


			# if not union:
			# 	#################### method 2 #####################
			# 	#cross over point: t
			# 	t = np.dot(normal_i, gravity_i - gravity_j)/np.dot(normal_i, normal_j)
			# 	tx, ty, tz = normal_j * t + gravity_j

			# 	x_min, y_min, z_min, x_max, y_max, z_max = range_i
			# 	if tx > x_min and tx < x_max and ty > y_min and ty < y_max and tz > z_min and tz < z_max:
			# 		union  = True
			# 		#calculat the distance between (tx,ty,tz) and the gravity_i
			# 		distance = np.sum((np.array([tx, ty, tz]) - np.array(gravity_i))**2)


			if not union:
				vertice_i = np.array(list(map(list, vertice_i)))
				vertice_j = np.array(list(map(list, vertice_j)))
				vertice_i[:,-1] = 0 
				vertice_j[:,-1] = 0

				[k,b] = np.plotfit(vertice_i[:,0], vertice_j[:,1], c = 1)
				theta = np.arctan(k)
				rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], 
									[-np.sin(theta), np.cos(theta)]])
				translation = vertice_i.mean(0)


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


				overlap_x_min = max(proj_xi_min, proj_xj_min)
				overlap_x_max = min(proj_x_max, proj_xj_max)
				proj_len = overlap_x_max - overlap_x_min

				ratio = proj_len *1.0/(proj_x_max - proj_x_min + proj_x_max_next - proj_x_min_next - proj_len)

				if ratio >= 0.5:
					union  = True





		return union, distance


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

	def write_into_obj(self, results, triangles, save_dir, model_name, name = None):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		if type(results) == dict:
			Num = len(results.keys()) 
			for k, groups in results.items():
				for j, group in enumerate(groups):
					if name != None:
						j = name
					with open(os.path.join(save_dir, model_name[:-4] + '_' + str(k) + '_' + str(j) + '.obj'),'w') as f:
						trias = triangles[group,:,:]
						trias_str = ['v '+str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for tria in trias for v in tria]
						f.writelines(trias_str)
						for i in range(int(len(trias))):
							f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')


		if type(results) == list:
			for i, group in enumerate(results):
				with open(os.path.join(save_dir, model_name[:-4] + '_' + str(i) + '.obj'),'w') as f:
					left_triangles = triangles[group,:,:]
					trias_str = ['v '+str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for triangle in left_triangles for v in triangle]
					f.writelines(trias_str)
					for i in range(int(len(left_triangles))):
						f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')


	