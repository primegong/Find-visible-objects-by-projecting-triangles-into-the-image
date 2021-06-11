import _init_paths
import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys,random
from read_data.OBJ_PARSER import *
from read_data.XML_PARSER import *
from read_data.MASK_PARSER import *
from projection_2D_3D.projection import *
from segments_optimization_3D.optimization import *
from segments_union_3D.union import *
from segment_estimation.seg_estimation import read_3D_segments, evaluate


def generate_colors(N = 10):
	HSV_tuples = [(x*1.0/N, 0.7, 0.7) for x in range(N)]
	RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
	random.shuffle(RGB_tuples)
	return RGB_tuples


def main(AT_file, model_list, model_path, segments_path):
	if not os.path.exists(model_path):
		raise(model_path + " is not exists, please check.")

	#读取图像名字，便于取图像参数
	segments_list = os.listdir(segments_path)
	segments_list = [seg[:-4] for seg in segments_list if os.path.splitext(seg)[-1] == '.pkl']
	
	#解析内外参
	para = parse_xml(AT_file)

	#读取二维分割结果
	total_masks = get_total_masks(segments_path, segments_list, name = model_path.split('/')[-1].split('_')[0] + '_' + segments_path.split('/')[-1])
	
	for model in model_list:
		print('\n ........................current model is: ' + model + '.............................')

		###读取三维坐标	
		model_file = os.path.join(model_path, model)
		obj_parser = OBJ_PARSER(model_file)
		triangles, faces, vertices = obj_parser.get_triangles()
		print('reading data done .....')


		#### stage 1: project 2D segments to 3D model #############
		proj_output_path = 'output_' + method + '/projection_2D_3D/' + model[:-4] + '/'
		if not os.path.exists(proj_output_path):
			os.makedirs(proj_output_path)
		print('stage 1: project 2D segments to 3D model .........')
		projection_results = projection(triangles, total_masks, segments_list, para, proj_output_path, region_size = 1000)
		print('Finish projection.....\n')
	

		####stage 2: 3D segments optimization ## ###################
		print('stage 2: 3D segments optimization...')
		opt_output_path = 'output_' + method + '/segments_optimization/' + model[:-4] + '/'
		if not os.path.exists(opt_output_path):
			os.makedirs(opt_output_path)
		opt = Optimization(proj_output_path, opt_output_path)
		opt.optimize()
		print('Finish optimization.....\n')


		### stage 3: union 3D fragments that projected from multiview images #####
		print('stage 3: union 3D fragments that projected from multiview images')
		union_output_path = 'output_' + method + '/segments_union_output/' + model[:-4] + '/'
		if not os.path.exists(union_output_path):
			os.makedirs(union_output_path)
		un = Union(opt_output_path, union_output_path, model[:-4]) 
		un.union_segments_3D()
		print('Finish union.....\n')


		# ### stage 4: 3d segment estimation ###############################
		gt_dirpath = 'GT_segments/' + model[:-4]
		pred_dirpath = 'output_' + method + '/segments_union_output/' + model[:-4] + '/opposite_side'
		gt_segments = read_3D_segments(gt_dirpath)
		pred_segments = read_3D_segments(pred_dirpath)
		instance_level_precision, instance_level_recall, mIOU, matches = evaluate(gt_segments, pred_segments)
		print('instance_level_precision: ', instance_level_precision, '\n')
		print('instance_level_recall：', instance_level_recall, '\n')
		print('mIOU: ',mIOU, '\n')
		print('matches_num: ',len(matches), '\n', 'matched obj: ', matches, '\n')


		#### stage 5 : symmetry detection  #############################
		# symmetry_output_path = 'output_' + method + '/symmetry_detection/' + model[:-4] + '/'
		# if not os.path.exists(symmetry_output_path):
		# 	os.makedirs(symmetry_output_path)
		# input_path = os.path.join(union_output_path, 'opposite_side')
		# sy = Symmetry(union_output_path, symmetry_output_path)
		# match_results = sy.symmetry_detection()
		# print('match_results:', match_results)
		

		# ### stage 6: OBB_3d ###############################
		# save_path = 'output_' + method + '/OBB/' + model[:-4] +'/'
		# if not os.path.exists(save_path):
		# 	os.makedirs(save_path)
		# OBB_3D = OBB(os.path.join(symmetry_output_path), save_path)




if __name__ == "__main__":
	# 空三坐标文件
	AT_file = '3D Models/gw - AT - AT - export.xml'

	# 模型文件
	model_path = '3D Models/4_nobg - 2/'
	model_list = ['4_nobg-2 - test -2.obj']
	
	# segmentation results
	method = 'Mask R-CNN'	## Mask R-CNN, Sharp Edge Loss
	segments_path = '2D Segments/4_nobg_images(projection area)/' + method + '/'

	main(AT_file, model_list, model_path, segments_path) 
