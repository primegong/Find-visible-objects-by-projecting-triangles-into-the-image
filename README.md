# Deep Neural Networks for Quantitative Damage Evaluation of Building Losses Using Aerial Oblique Images: Case Study on the Great Wall (China)

# There are three stages in our approach:

1. building reconstruction
2. 3D object segmentation from mesh model with multiview oblique images.
3. damage estimation 
4. localization of missing objects through symmetry detection 

# This is an implementation of the second stage, which consists of three steps: 

1) Assign category information and instance id to each triangle: Based on the correspondence between oblique images and the 3D mesh model, we reproject each triangle in the mesh model to the oblique images to obtain the category information and instance id.
2) Eliminiate the background cluster by the distance between the center of the triangle and cammera center and remove the visible noise through spatial connectivity.
3) The generated fragments from multiview images are grouped to form a complete 3D object.

![image](graphical_abstract.png)


![image](framework.png)


@article{gong2021deep,
  title={Deep Neural Networks for Quantitative Damage Evaluation of Building Losses Using Aerial Oblique Images: Case Study on the Great Wall (China)},
  author={Gong, Yiping and Zhang, Fan and Jia, Xiangyang and Huang, Xianfeng and Li, Deren and Mao, Zhu},
  journal={Remote Sensing},
  volume={13},
  number={7},
  pages={1321},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}


