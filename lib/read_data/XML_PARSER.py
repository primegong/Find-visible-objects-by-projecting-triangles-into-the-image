import os,sys
import numpy as np
import xml.dom.minidom as DM

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
		except:
			ImageWidth = float(root.getElementsByTagName('Width')[0].childNodes[0].data)
			ImageHeight = float(root.getElementsByTagName('Height')[0].childNodes[0].data)


		try:
			flocalLengthNode = root.getElementsByTagName('FocalLengthPixels')
			f = float(flocalLengthNode[0].childNodes[0].nodeValue)
		except:
			try:
				FocalLength = float(root.getElementsByTagName('FocalLength')[0].childNodes[0].nodeValue)
				SensorSize = float(root.getElementsByTagName('SensorSize')[0].childNodes[0].nodeValue)
			except:
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
			id = photo.getElementsByTagName('Id')[0].childNodes[0].nodeValue
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