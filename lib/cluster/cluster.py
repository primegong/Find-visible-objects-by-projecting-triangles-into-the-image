from scipy import spatial
import numpy as np 


'''main function: Cluster_MultiOutput
	input : datalist
	output: grouped datalist and their indexes in original datalist'''

def get_R2(group_faces, datalist):
	########  1 step: get the search radius ######################## 
	tree = spatial.KDTree(data = datalist.tolist())
	#1 step: get search radius: R 
	distances = []
	for j, target_gravity in enumerate(datalist):
		[v1,v2,v3] = group_faces[j]

		ind1 = np.where(group_faces == v1)[0].tolist()
		ind2 = np.where(group_faces == v2)[0].tolist()
		ind3 = np.where(group_faces == v3)[0].tolist()
		ind = list(set(ind1+ind2+ind3))	

		find_gravity = datalist[ind,:]

		dists = np.sum((target_gravity-find_gravity)**2, axis = 1)
		dists = np.array(sorted(dists))
		median_localization = int(len(dists)/2)
		median_dist = dists[median_localization]

		distances.append(median_dist)


	distances = np.array(sorted(distances))
	median_localization = int(len(distances)/2)
	R = distances[median_localization]
	return R, tree



def get_R(datalist, point_num):
	########  1 step: get the search radius ######################## 
	if type(datalist) != list:
		datalist = datalist.tolist()
	tree = spatial.KDTree(data = datalist)
	#1 step: get search radius: R 

	datalist = np.array(datalist)
	neighbor_R = []
	for target in datalist:
		(dists, indexes) = tree.query(target, point_num)

		# num = len(dists)
		# if num % 2 == 0:
		# 	median_dist = (dists[int(num/2)] + dists[int(num/2-1)])/2.0	
		# elif num % 2 == 1:
		# 	median_dist = dists[int((num-1)/2)]

		# if median_dist - median_dist != 0:
		# 	print(neighbor_R, 'nan!!!!!!!!')
		# neighbor_R.append(median_dist)

		mean_dis = np.array(dists).mean()
		neighbor_R.append(mean_dis)


	sorted_R = np.array(sorted(neighbor_R))
	median_localization = int(len(neighbor_R)/2)
	R = sorted_R[median_localization]



	return R, tree, neighbor_R



def Cluster_MultiOutput(datalist, R = None, point_num = 11):
	datalist = np.array(datalist)

	row, col = datalist.shape
	if col<3:
		pad = np.zeros((row, 3-col))
		datalist = np.hstack((datalist, pad))
	
	if R == None:
		R, tree, neighbor_R = get_R(datalist, point_num)
	
	
	# median_value = distances[int(len(distances)/2)]
	# distances = np.array([d for d in distances if d <median_value*1000])
	# mean_distance = distances.mean()
	# std = np.sqrt(np.sum((distances-mean_distance)**2)/len(distances))
	# mR = mean_distance


	grouped_Trias, index_OfTrias = cluster(datalist, R, neighbor_R)
	
	return grouped_Trias, index_OfTrias



def cluster(datalist, R, neighbor_R):
	sorted_inds = sorted(range(len(neighbor_R)), key=lambda k: neighbor_R[k])

	tree = spatial.KDTree(data = datalist)

	########  2 step: cluster ######################## 
	grouped_Trias = []
	target_used = []
	used = []
	index_OfTrias = []
	for j in sorted_inds:
		target = datalist[j]
		if tuple(target) in used:
			continue
		if j == len(datalist)-1:
			break

		group = [tuple(target)]
		index = [j]
		while True:
			
			inds = tree.query_ball_point(target, r = R)
			index = list(set(index).union(set(inds)))

			results = datalist[inds,:].tolist()
			results = list(map(tuple,results))
			results = list(set(results))
			
			group = list(set(results).union(set(group)))
			


			target_used.append((target[0], target[1], target[2]))
			if len(results)>1:
				#### find edge points ##############
				edge_points = convex_hull(results)
				inner_points = list(set(results).difference(set(edge_points)))
				target_used.extend(inner_points)
		
			
			#next:
			diff = list(set(group).difference(set(target_used)))
			if len(diff):
				target = diff[0]    
			else:
				break 
				             
		used = used + group
		group = list(map(list,group))
		grouped_Trias.append(group)
		index_OfTrias.append(index)
	return grouped_Trias, index_OfTrias