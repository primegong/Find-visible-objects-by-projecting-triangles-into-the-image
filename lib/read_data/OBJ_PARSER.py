import numpy as np


class OBJ_PARSER:
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
						try:
							v,x,y,z = line.strip().split(' ')
						except:
							v,x,y,z,_,_,_ = line.strip().split(' ')
						self.vertices.append([x,y,z])
				if case == 2:
					if line[:2] == 'vt':
						vt,u,v = line.strip().split(' ')
						self.vts.append([u,v])
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
		for index in faces:
			triangle = [vertices[index[0]-1], vertices[index[2]-1], vertices[index[4]-1]]
			self.triangles.append(triangle)
		self.triangles = np.array(self.triangles).astype(np.float32)
		# print('total triangles: ',len(self.triangles))
		return self.triangles, faces, vertices

	def get_normals(self):
		if self.triangles == []:
			self.get_triangles()
		normals = []
		for triangle in self.triangles:
			v11 = triangle[1, :] - triangle[0, :]
			v22 = triangle[2, :] - triangle[0, :]
			normal = np.cross(v11, v22)
			normal_unit = normal / np.linalg.norm(normal)
			normals.append(normal_unit)
		self.normals = normals
		return self.normals

