import os,  shutil
import numpy as np
import copy, time
import pickle as pkl
import kdtree
from progressbar import *
import random


class Label_union:

	def angle(self,ray_c1, ray_c2):
		norm1 = np.sqrt(np.sum(ray_c1**2))
		norm2 = np.sqrt(np.sum(ray_c2**2))
		angle = np.dot(ray_c1, ray_c2)/(norm1*norm2)
		return angle

	def has_common_clouds(self, TrisObj , TrisSearch , normal_i, normal_j, gravity_i, gravity_j, thresh = 0.7):
		
		common_clouds = list(set(TrisObj).intersection(set(TrisSearch)))
		n = min(len(TrisObj),len(TrisSearch))
		union = False
		ratio = len(common_clouds) *1.0/n 
		if ratio>= thresh:
			union = True
		elif ratio <thresh and ratio >0.0:
			angle = self.angle(normal_i, normal_j)
			gravity_direct = gravity_i - gravity_j
			angle_i = self.angle(gravity_direct, normal_i)
			angle_j = self.angle(gravity_direct, normal_j)
			if angle<0 and min(np.abs(angle_i),np.abs(angle_j))>0.9:
				union = True
		return union


	def get_meanNormal(self, adict, triangles, save_dir):
		if os.path.exists(os.path.join(save_dir,'normals.pkl')):
			print('load normal from pkl files...')
			f = open(os.path.join(save_dir,'normals.pkl'),'rb')
			f.seek(0)
			normals = pkl.load(f)
			gravitys = pkl.load(f)
		else:
			normals = {}
			gravitys = {}
			for key, value in adict.items():
				normal = np.sum(np.array(value)[:,[1,2,3]],axis = 0)/len(value)
				trias = triangles[list(np.array(value)[:,0].astype(np.int))]
				gravity = trias.mean(0).mean(0)
				normals[key] = normal
				gravitys[key] = gravity
			f = open(os.path.join(save_dir,'normals.pkl'),'wb')
			pkl.dump(normals,f)
			pkl.dump(gravitys,f)
		return normals, gravitys



	def write_into_txt(self, tri_indexInGroup, triangles,save_dir):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for i, group in enumerate(tri_indexInGroup):
			with open(save_dir + str(i) + '.txt','w') as f:
				left_triangles = triangles[group,:,:]
				trias_str = [str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for triangle in left_triangles for v in triangle]
				f.writelines(trias_str)

	def write_into_obj(self, results, triangles, save_dir, model, name = None):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		if type(results) == dict:
			Num = len(results.keys()) 
			for k, groups in results.items():
				for j, group in enumerate(groups):
					if name != None:
						j = name
					with open(os.path.join(save_dir, model[:-4] + '_' + str(k) + '_' + str(j) + '.obj'),'w') as f:
						trias = triangles[group,:,:]
						trias_str = ['v '+str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for tria in trias for v in tria]
						f.writelines(trias_str)
						for i in range(int(len(trias))):
							f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')


		if type(results) == list:
			for i, group in enumerate(results):
				with open(os.path.join(save_dir, model[:-4] + '_' + str(i) + '.obj'),'w') as f:
					left_triangles = triangles[group,:,:]
					trias_str = ['v '+str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n' for triangle in left_triangles for v in triangle]
					f.writelines(trias_str)
					for i in range(int(len(left_triangles))):
						f.write('f ' + str(i*3+1) + ' ' + str(i*3+2) + ' ' + str(i*3+3) + '\n')

	def write_into_obj2(self, results, vertices, faces, save_dir, model, name = None):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		vertices = vertices[:,[1,2,3]]
		lines = []
		for vertice in vertices:
			[x,y,z] = vertice
			line = 'v ' +str(x) + ' ' +str(y) + ' ' +str(z) + '\n'
			lines.append(line)

		for k, trias_indexes in results.items():
			for j, indexes in enumerate(trias_indexes):
				if name != None:
					j = name
				with open(os.path.join(save_dir, model[:-4] + '_' + str(k) + '_' + str(j) + '.obj'),'w') as f:
					f.writelines(lines)
					for tj in indexes:
						index = faces[tj]
						face_index = str(index[0])+ ' ' + \
									str(index[1]) + ' ' + \
									str(index[2])
						f.write('f ' + face_index + ' ' + '\n')

	def union(self, adict, triangles, save_dir, pbar):

		if os.path.exists(os.path.join(save_dir, 'union_results.pkl')):
			print('load union_results from pkl files...')
			f = open(os.path.join(save_dir, 'union_results.pkl'), 'rb')
			f.seek(0)
			union_results = pkl.load(f)
			f.close()
		else:
		
			normals, gravitys = self.get_meanNormal(adict,triangles, save_dir)
			labels= list(adict.keys())
			
			groups = []
			used = []


			N = len(labels)
			for i in range(N):#label的数量
				pbar.update(i*1.0/N*100)

				if len(adict[labels[i]])<500:#每个label中少于10个三角形的删除
					continue

				if labels[i] in used:
					continue 

				if i == N-1:
					group = [labels[i]]
					groups.append(group)
					continue

				#group为临时变量,下一次循环重新生成
				group = [labels[i]]
				used.append(labels[i])
				

				TargetIndex = group[0]
				normal_i = normals[TargetIndex]
				gravity_i = gravitys[TargetIndex]
				# R_i = RS[TargetIndex]
				# cloud_i = clouds[TargetIndex]
				while TargetIndex:
					TrisObj = list(np.array(adict[TargetIndex])[:,0])#mask（label）含有的三角网
					
					for j in range(i+1,N):
						SearchIndex = labels[j]
						if SearchIndex in used:
							continue                    
						TrisSearch  = list(np.array(adict[SearchIndex])[:,0])#mask（label）含有的三角网
						# R_j = RS[SearchIndex]
						# cloud_j = clouds[SearchIndex]
						if len(TrisSearch)<500:
							continue
						normal_j = normals[SearchIndex]
						gravity_j = gravitys[SearchIndex]
						if self.has_common_clouds(TrisObj , TrisSearch , normal_i, normal_j, gravity_i, gravity_j):
							normal_i = (normal_i+normal_j)/2
							gravity_i = (gravity_i + gravity_j)/2
							TrisObj = list(set(TrisObj).union(set(TrisSearch)))
							group.append(SearchIndex)
							used.append(SearchIndex)
					#next
					if group.index(TargetIndex) == len(group)-1:
						break
					TargetIndex = group[group.index(TargetIndex)+1]
				
				#记录所有的分组
				groups.append(group)
			
			# 将一组写进一个txt
			# print(groups,"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
			union_results = []
			for group in groups:
				results = []        
				for g in group:
					triangles_index = np.array(adict[g])[:,0].astype(np.int)
					results = results + list(triangles_index)
				#去重复
				results = list(set(results))
				union_results.append(results)
			f = open(os.path.join(save_dir, 'union_results.pkl'),'wb')
			pkl.dump(union_results, f)
		return union_results

	def Cluster_MuiltiOutput(self, group, triangles, mR = -1.0):
		################################## 用三角形重心构建kdtree############################################
		trias = []
		#计算每个组的所有三角形重心
		for j in group:
			triangle = triangles[j,:,:]
			gravity = list((triangle[0] + triangle[1] + triangle[2])/3)
			trias.append(gravity)
		#将所有组的重心添加到kdtree中
		tree = kdtree.create(trias)
		#计算所有三角形的搜索半径（mR） 
		if mR == -1.0:

			median_dists = []
			for target in trias:
				#The result is an ordered list of (node, distance) tuples.
				results = tree.search_knn(target, 10)
				if len(results)<=1:
					continue

				# results_temp = results.copy()
				# length = len(results)
				# for i in range(length-1,-1,-1):#双向检测删除噪声点
				#   results_new = tree.search_knn(results_temp[i][0].data, 10)
				#   results_new_trias = [r[0].data for r in results_new]
				#   if target not in results_new_trias:
				#       del(results[i])
				if  len(results) % 2==1:
					median_dist = (results[int(len(results)/2)][1]+results[int(len(results)/2)+1][1])/2.0
				elif len(results) % 2==0:
					median_dist = results[int(len(results)/2)][1]
			
				median_dists.append(median_dist)
			if len(median_dists)<1:
				return [group],[[0]]
		
			median_dists = np.array(median_dists)
			median_dists =sorted(median_dists)
			median = median_dists[int(len(median_dists)/2)]
			median_dists = np.array([m for m in median_dists if m <median*1000])

			mean_distance = np.sum(median_dists)/(len(median_dists)-1)
			std = np.sqrt(np.sum((median_dists-mean_distance)**2)/len(median_dists))
			mR = mean_distance + 3*std
			# print(mR)
		#聚类每个组的所有三角形
		tris_ID_InGroup = []
		tris_indexes = []
		seed_used = []
		results_all_used = []
		
		for i, target in enumerate(trias):
			#若三角形已有类别号，不进行处理
			if tuple(target) in results_all_used:
				continue
			results_used = []
			tri_ID_InGroup = []
			seed = target
			while True:
				seed_used.append((seed[0],seed[1],seed[2]))
				results = tree.search_nn_dist(seed, mR)#搜索radius内的点
				results_Merge = list(set(list(map(tuple,results)) + results_used))#新的所有点集
				#点集的索引
				diff_results = list(set(results_Merge).difference(set(results_used)))#新加入的点集
				indices = [trias.index(list(rn)) for rn in diff_results]#新的点集的索引
				tri_ID_InGroup = tri_ID_InGroup + indices           #新的点集的索引
				#种子点的索引
				diff_seed = list(set(results_Merge).difference(set(seed_used)))
				if len(diff_seed):
					seed = diff_seed[0]     #更新种子点
				else:
					break
				results_used = results_Merge                        #新的所有点集
			results_all_used = results_all_used + results_used      #所有已确定类别的三角形
			if len(tri_ID_InGroup)>500:
				#转换到原始triangles的索引
				tri_index = [group[index] for index in tri_ID_InGroup]
				#将group k 的第i个聚类结果保存起来
				tris_ID_InGroup.append(tri_ID_InGroup)
				tris_indexes.append(tri_index)


		return tris_indexes, tris_ID_InGroup
	


	def Cluster_SingleOutput(self, group, triangles, center):
		tris_indexes, tris_ID_InGroup = self.Cluster_MuiltiOutput(group, triangles)
		############################ restore the group that has most triangles##############
		if len(tris_ID_InGroup) == 0:
			left_tri_indexes, left_tri_ID_InGroup = [], []
		elif len(tris_ID_InGroup) == 1:
			left_tri_ID_InGroup = tris_ID_InGroup[0]
			left_tri_indexes = tris_indexes[0]
		else:
			################求算组内所有三角形重心到center的距离,保留到center最近的组#########################
			min_dis = float("inf")
			# print(tris_indexes,'##original')
			for i, tri_index in enumerate(tris_indexes):
				# print(i)
				trias = triangles[tri_index,:,:]
				gravity = []
				for tria in trias:
					gravity_each = np.sum(np.array(tria),axis = 0)/3
					gravity.append(list(gravity_each))
				gravity = np.sum(np.array(gravity), axis = 0)/len(gravity)

				s_2 = np.sum((center-gravity)**2)

				if s_2 < min_dis:
					min_dis = s_2
					left_tri_ID_InGroup = tris_ID_InGroup[i]
					left_tri_indexes = tris_indexes[i]
		return left_tri_indexes, left_tri_ID_InGroup
			

	
	def filter_island(self, results, faces):
		faces = np.array(faces)[:,[0,2,4]]
		for k, groups in results.items():
			results[k] = []
			
			for j, group in enumerate(groups):
				####group = [j1,j2,j3,.....]
				print(len(group))
				

				collections = []
	
				while len(group):
					group_faces = faces[group,:]

					target = group[0]
					collection = []
					target_used = []
					
					while True:

						[v1,v2,v3] = faces[target]
						target_used.append(target)
		
						finds = list(set(np.where(group_faces == v1)[0]))
						finds += list(set(np.where(group_faces == v2)[0]))
						finds += list(set(np.where(group_faces == v3)[0]))
						finds = list(set(finds))
						# print(finds)
						finds = [group[f] for f in finds]
		
	
						if len(finds):
							collection = list(set(collection).union(set(finds)))
						
						if len(collection) == len(group):
							group = []
							print('finish')
							break

						diff = list(set(collection).difference(set(target_used)))
						if len(diff):
							target = diff[0]
						else:
							break
					
					collections.append(collection)
					group = list(set(group).difference(set(collection)))
					print(len(group),len(collection))
				
				
				num = [len(c) for c in collections]
				num = np.array(num)
				order = np.argsort(-num)
				collection_main = collections[order[0]]
				print(len(collection_main))

			results[k].append(collection_main)

		return results	



	def __call__(self, adict, triangles, args, save_dir, model, faces):

		##################### 1. preprocess on each mask(消除二义性) ########################
		print('De-noise.........')
		pbar = ProgressBar().start()
		
		seg_dir = os.path.join(save_dir,'segmentation_preprocess')
		if not os.path.exists(seg_dir):
			os.makedirs(seg_dir)
		if os.path.exists(os.path.join(seg_dir, 'seg_results.pkl')):
			f = open(os.path.join(seg_dir, 'seg_results.pkl'),'rb')
			f.seek(0)
			reconstruct_adict = pkl.load(f)
			time_0 = 0
			while time_0<100:
				pbar.update(time_0)
				time_s = random.uniform(0,1)
				time.sleep(time_s)	
				time_0 += 30*time_s	
		else:
			
			tri_indexes = {}
			reconstruct_adict = {}
			j = 0
			N = len(adict.keys())
			for key, value in adict.items():
				pbar.update(j*1.0/N*100)
				j+=1
				group=list(np.array(value)[:,0].astype(np.int))
				imagefile = key.split('_')[0] + '_' + key.split('_')[1] + '_'+key.split('_')[2] + '.JPG'
				center = args[imagefile]['Center']
				left_tri_indexes, left_tri_ID_InGroup = self.Cluster_SingleOutput(group, triangles, center)

				if len(left_tri_indexes) != 0:
					# tri_indexes[str(key)] = [left_tri_indexes]
					reconstruct_adict[str(key)] = list(np.array(value)[left_tri_ID_InGroup,:])
					# print(reconstruct_adict[str(key)])
			f = open(os.path.join(seg_dir, 'seg_results.pkl'),'wb')
			pkl.dump(reconstruct_adict,f)
			self.write_into_obj(tri_indexes, triangles, seg_dir, model)
		pbar.finish()
		###################### 2. combine masks those has lots of common clouds ###################
		print('Reconstructing.........')
		pbar = ProgressBar().start()
		
		union_dir = os.path.join(save_dir,'union')
		if not os.path.exists(union_dir):
			os.makedirs(union_dir)
		if os.path.exists(os.path.join(union_dir, 'union_results.pkl')):
			f = open(os.path.join(union_dir, 'union_results.pkl'),'rb')
			f.seek(0)
			union_results = pkl.load(f)
			time_0 = 0
			while time_0<100:
				pbar.update(time_0)
				time_s = random.uniform(0,1)
				time.sleep(time_s)	
				time_0 += 30*time_s	
		else:
			
			union_results = self.union(reconstruct_adict, triangles, union_dir, pbar)#adict
			self.write_into_obj(union_results, triangles, union_dir, model)
		pbar.finish()

		####################### 3. recluster object #############################################
		print('Refining.........')
		pbar = ProgressBar().start()
		
		cluster_dir =  os.path.join(save_dir,'clustering')
		if not os.path.exists(cluster_dir):
			os.makedirs(cluster_dir)
		if os.path.exists(os.path.join(cluster_dir, 'tri_indexInGroup.pkl')):
			f = open(os.path.join(cluster_dir, 'tri_indexInGroup.pkl'),'rb')
			f.seek(0)
			tri_indexInGroup = pkl.load(f)
			time_0 = 0
			while time_0<100:
				pbar.update(time_0)
				time_s = random.uniform(0,1)
				time.sleep(time_s)	
				time_0 += 30*time_s
			pbar.finish()
		else:	
			j = 0 
			N = len(union_results)
			tri_indexInGroup = {}
			for key, group in enumerate(union_results):
				pbar.update(j*1.0/N*100)
				j+=1
				tris_indexes, tris_ID_InGroup = self.Cluster_MuiltiOutput(group, triangles)
				if len(tris_ID_InGroup) == 0:
					continue
				################### 保留点数最多的 ###################
				number = [len(item) for item in tris_indexes]
				sorted_number = sorted(number,reverse = True)
				index = number.index(sorted_number[0])
				maximum_tris_indexes = tris_indexes[index]
				######################################################
				if len(maximum_tris_indexes)>500:
					tri_indexInGroup[str(key)] = [maximum_tris_indexes]
			pbar.finish()
			if len(tri_indexInGroup.keys())>0:
				print('Filtering island.........')
				tri_indexInGroup = self.filter_island(tri_indexInGroup, faces)
				self.write_into_obj(tri_indexInGroup, triangles, cluster_dir, model)
		
	


if __name__ == '__main__':

	path = '../results/minTriangles_Cut/'
	file_list = os.listdir(path)

	save_dir = '../results/filter_island/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	from _projection import Projection

	for file in file_list:
		if os.path.exists(os.path.join(save_dir,file)):
			continue

		obj_file = os.path.join(path,file)
		pro = Projection(obj_file)
		
		vertices = pro.get_vertices()
		faces = pro.get_faces()
		[model0, model1, model2, num, num0, ext] = file.split('_')
		results = {num:[]}
		results[num].append(np.arange(len(faces)))

		lu = Label_union()
		results = lu.filter_island(results, faces)
		lu.write_into_obj2(results, vertices, faces, save_dir, model = model0 +'_' + model1 + '_' + model2 + '.obj', name = num0)

	
	