import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from matplotlib.patches import Ellipse

import matplotlib.transforms as transforms

#from tensorflow.keras.preprocessing import image

from PIL import Image

import zipfile
import os

class ReconstructStimulusClass: 

	def __init__(self):

		#self.fig, self.ax = plt.subplots()

		self.imgs = []

		self.head_radius = 2/3  # hard coded values based on pixels + average size of female (roughly 4mm)
		self.tail_radius = 0.5
		self.ellipse_radius = 2.

		self.image_width = 45 # 4.  # for stripes model


	### HELPER FUNCTIONS

	def reset_fig(self):
		self.fig, self.ax = plt.subplots()

		self.imgs = []


	def resize_image(self, imgsave_filename):
		# resize jpg
		im = Image.open(imgsave_filename + '.png')
		im = im.resize((256,64), Image.LANCZOS)
		# im.save(imgsave_filename + '.png', quality=100, subsampling=0)
		path = '{}.png'.format(imgsave_filename)
		assert os.path.isfile(path)
		#with open(path, "r") as f:
    	#	pass
		im.save(path) #imgsave_filename + '.png')


	def get_head_dir(self, positions):
		# positions: (3,2) positions head/body/tail, x/y
		#	returns head_dir: (2,) estimated head dir for each frame

		ihead=0; ibody=1; itail=2;
		head_dir = positions[ihead,:] - positions[ibody,:]
		head_dir = head_dir / (np.sqrt(np.sum(head_dir**2)) + 1e-5)

		return head_dir


	def get_orth_head_dir(self, head_dir):
		# finds orth head dir (90 degrees clockwise)
		# INPUT:
		#   head_dir (2,)  original head dir (normalized)
		# OUTPUT:
		#	orth_head_dir (2,) 90 degree clockwise rotation (to the fly's right)

		orth_head_dir = np.zeros((2,)) # rotation matrix causes [x,y] to have orth vector: [y,-x], 90 degree clockwise rotation
		orth_head_dir[0] = head_dir[1]
		orth_head_dir[1] = -head_dir[0]  
		orth_head_dir = orth_head_dir / (np.sqrt(np.sum(orth_head_dir**2)) + 1e-5)

		return orth_head_dir


	def get_position_along_circumfrence(self, head_dir, orth_head_dir, pos_female):
		# INPUT:
		#	head_dir: (2,), head direction of male fly
		#	pos_female: (2,), female position relative to male's head

		# get angle between head_dir and pos_female
		pos = np.copy(pos_female) / (np.sqrt(np.sum(pos_female**2)) + 1e-6)

		angle_head_dir_pos_female = np.arccos(np.sum(head_dir * pos))

		# get proj onto orth head_dir
		proj_orth = np.sum(orth_head_dir * pos)

		if (proj_orth < 0):   # if female on left side, make angle negative
			angle_head_dir_pos_female = -angle_head_dir_pos_female

		# find position along circumfrence (between 0 and 1)
		position_along_circum = angle_head_dir_pos_female / np.pi
		# negative 1 is furthest left, positive 1 is furthest right (both are directly behind fly)

		return position_along_circum


	def get_radius(self, diff_pos_joint_female, joint='head'):
		# returns radius in terms of 1/pi
		
		if joint == 'head':
			joint_radius = self.head_radius
		else:
			joint_radius = self.tail_radius

		dist_between_flies = np.sqrt(np.sum((diff_pos_joint_female)**2)) + 1e-6

		# find radius of object, which is akin to finding angle between center and outer edge of circle
		radius = np.arctan(joint_radius / dist_between_flies) * 2.0

		return radius / np.pi


	def get_stripesmodel_params(self, diff_pos_female, head_dir_male, orth_head_dir_male, head_dir_female):
		# width is a fixed distance in xy space (self.image_width) that I am assuming is the female fly's width (~4 mm)
		#	height is simply a scalar of this

		diff_pos_body_female = diff_pos_female[1,:]

		# lateral position (0 is front facing, negative is to the left, positive is to the right)
		lateral_pos = self.get_position_along_circumfrence(head_dir_male, orth_head_dir_male, diff_pos_body_female)
		# lateral_pos_head = self.get_position_along_circumfrence(head_dir_male, orth_head_dir_male, diff_pos_female[0,:])
		# lateral_pos_tail = self.get_position_along_circumfrence(head_dir_male, orth_head_dir_male, diff_pos_female[2,:])

		# width (use arctan for this)
		dist = np.sqrt(np.sum(diff_pos_body_female**2)) + 1e-6
		view_width = np.arctan(self.image_width /2.0 / dist) / np.pi * 4.0  # * 2 b/c tan^-1 returns in range 0 to pi/2 and another 2 b/c we halved the width

		view_height = 0.46 * view_width # 0.46 scalar b/c image height (225 pixels) is 0.46 of image width (490 pixels)
												# not used for stripes model

		return (lateral_pos, view_height, view_width)


	def get_degree_female_image(self, head_dir_male, orth_head_dir, head_dir_female, rel_body_position_female):
		# computes the angle (in degrees) for the femalefly360 image
		#
		# INPUT:
		#	head_dir_male: (2,), male's body->head dir in absolute coordinates
		#	orth_head_dir: (2,), dir orthogonal to male head dir (90 degrees clockwise assuming we are looking down on top of male)
		#			e.g., if head_dir_male=[1,0], then orth_head_dir=[0,-1]
		#	head_dir_female: (2,), female's body->head dir in absolute coordinates
		#	rel_body_position_female: (2,), relative vector between male's head position and female's body position in absolute coordinates
		#
		# OUTPUT:
		#	angle: (integer between -180 and 180), rotation angle for the female fly
		#		angle of 0 --> female faces away from male
		#		angle of -90 --> female faces to the left
		#		angle of 90 --> female faces to the right
		#		angle of -180/180 --> female faces towards male

		# idea: compute angle between vector of male-female positions and female head direction
		#   step 1: rotate to male-centric coordinates (male facing towards (1,0))
		#   step 2: identify direction orthogonal to the relative body direction
		#   step 3: compute angle between female head dir and relative body direction

		# project head_dir_female and rel_body_position_female onto head_dir_male coordinates
		# so that male is now facing (1,0)
		if True:
			x_new = np.copy(head_dir_female)
			x_new[0] = np.dot(head_dir_male.T, head_dir_female)
			x_new[1] = np.dot(orth_head_dir.T, head_dir_female)
			head_dir_female = x_new

			x_new = np.copy(rel_body_position_female)
			x_new[0] = np.dot(head_dir_male.T, rel_body_position_female)
			x_new[1] = np.dot(orth_head_dir.T, rel_body_position_female)
			rel_body_position_female = x_new

			rel_body_direction = rel_body_position_female / (np.sqrt(np.sum(rel_body_position_female**2)) + 1e-6)

		# compute angle between female head dir and relative body direction
		#   to see if female is facing towards/away from male
		p = np.dot(head_dir_female.T, rel_body_direction)
		p = np.clip(p, a_min=-1, a_max=1)

		# identify if female is facing to the left/right of male
		if True:
			# find orthogonal projection of rel_body_direction (90 degree clockwise)
			orth_rel_body_direction = -self.get_orth_head_dir(rel_body_direction)
			  # negate here b/c we rotated the coordinates so male faces 
			  #  the right (1,0) and the male's right is down but considered (0,1)..which is usually (0,-1)
			  #  so you need to negate the orth dir (b/c here anti-clockwise is positive)
			  #  (this is a very subtle point but important for the signage)

			# compute sign --> positive if female faces toward the right, negative if to the left
			p_orth = np.dot(head_dir_female.T, orth_rel_body_direction)
			p_orth = np.clip(p_orth, a_min=-1, a_max=1) # clips in case of numerical problems
			signer = np.sign(p_orth)
			if signer == 0:
			  signer = 1

		# compute final angle
		angle = signer * np.arccos(p) / np.pi * 180
		  # signer --> denotes if female faces to the left (negative) or right (positive)

		angle = np.round(angle).astype(int)
		angle = np.clip(angle, a_min=-180, a_max=180)

		return angle



### FUNCTIONS FOR RECONSTRUCTING STIMULI

	def reconstruct_image_of_female_stripesmodel(self, positions_male, diff_positions_female, save_filepath):
		# reconstructs image (with respect to male heading dir) given male and female positions
		#    - this is for one image. Saves as png (no loss). Once saved, you should append it to a zip file.
		#
		# INPUT:
		#	positions_males: (3,2) for (head/body/tail, x/y) positions of male fly
		#	diff_positions_female: (3,2) for (head/body/tail, x/y) positions of female *with respect to male head position*
		#   save_filepath: (string), save filename, e.g., './figs/image_for_zip_{:s}_{:s}_fly{:d}'.format(fly_type, stim_type, ifly)
		#
		# OUTPUT:
		#   None.  (saves new image as png in save_filepath)

		fig, ax = plt.subplots()
		fig.set_size_inches(10,2.5)
		ax.set_xlim([-1, 1])
		ax.set_ylim([-0.25, 0.25])

		ix=0; iy=1;
		ihead=0; ibody=1; itail=2;

		# collect head directions
		head_dir_male = self.get_head_dir(positions_male)
		orth_head_dir = self.get_orth_head_dir(head_dir_male)

		head_dir_female = self.get_head_dir(diff_positions_female)

		rel_body_position_female = diff_positions_female[1,:]

		angle_3dmodel = self.get_degree_female_image(head_dir_male, orth_head_dir, head_dir_female, rel_body_position_female)

		# load image
		img_filepath = './stripes360deg/fly{:d}.png'.format(angle_3dmodel)

		im = Image.open(img_filepath)
		img = np.array(im)

		# get position, height, and width
		(lateral_pos, view_height, view_width) = self.get_stripesmodel_params( 
							diff_positions_female, head_dir_male,
							orth_head_dir, head_dir_female)

		left = lateral_pos - view_width/2
		right = lateral_pos + view_width/2
		top = view_height/2
		bottom = -view_height/2

		ax.imshow(img, extent=(left,right,bottom,top))

		bbox = transforms.Bbox([[1.3,0.3],[8.95,2.15]])

		z = 1. #img[0,0,0]/255
		ax.set_facecolor((z,z,z))
		impath = '{}.png'.format(save_filepath)
		fig.savefig(impath, bbox_inches=bbox) #save_filepath + '.png', bbox_inches=bbox)

		plt.close('all')

		self.resize_image(save_filepath)


	def get_parameters_of_stripesmodel(self, positions_male, diff_positions_female):
		# get female_width, female_orientation, and female_lateral_position for one time frame
		#		(these can be used for modeling purposes or to search for params)
		#
		# INPUT:
		#	positions_male: (3,2), (head/body/tail, x/y) for male position at one time frame
		#	diff_positions_female: (3,2), (head/body/tail, x/y) for relative female position (male head subtracted)
		# OUTPUT:
		#	female_width: (float) size of image patch of female (in visual_degrees / 180)
		#	female_orientation: (float), rotation angle of female; 0 --> female directly facing male, +45 --> female faces to right of male
		#								+135 --> female faces away to the right, -135 --> female faces away to left of male, -180/180 --> faces directly away from male
		#	female_lateral_position: (float), -1 to 1, position of female (in visual degrees / 180)

		# collect head directions
		head_dir_male = positions_male[0,:] - positions_male[1,:] # head - body
		head_dir_male /= np.sqrt(np.sum(head_dir_male**2)) + 1e-6
		orth_head_dir = self.get_orth_head_dir(head_dir_male)

		head_dir_female = diff_positions_female[0,:] - diff_positions_female[1,:]
		head_dir_female /= np.sqrt(np.sum(head_dir_female**2)) + 1e-6

		rel_body_position_female = diff_positions_female[1]

		# rel_body_position_female = diff_positions_female[1,:]
		angle_3dmodel = self.get_degree_female_image(head_dir_male, orth_head_dir, head_dir_female, rel_body_position_female)

		# get position, height, and width
		(lateral_pos, view_height, view_width) = self.get_stripesmodel_params( 
							diff_positions_female, head_dir_male,
							orth_head_dir, head_dir_female)

		female_width = view_width
		female_orientation = angle_3dmodel
		female_lateral_position = lateral_pos

		return female_width, female_orientation, female_lateral_position


	def reconstruct_image_of_female_stripesdmodel_for_tuning(self, orientation_angle, width, lateral_position, save_filepath):

		savefolder_pngs_of_3dmodel = './stripes360deg/'
				# filenames: fly{theta}.png

		fig, ax = plt.subplots()
		fig.set_size_inches(10,2.5)
		ax.set_xlim([-1, 1])
		ax.set_ylim([-0.25, 0.25])

		orientation_angle = np.round(orientation_angle).astype('int')

		# load image
		img_filepath = './stripes360deg/fly{:d}.png'.format(orientation_angle)

		im = Image.open(img_filepath)
		img = np.array(im)

		view_width = width
		view_height = 0.56 * view_width

		left = lateral_position - view_width/2
		right = lateral_position + view_width/2
		top = view_height/2
		bottom = -view_height/2

		ax.imshow(img, extent=(left,right,bottom,top))

		bbox = transforms.Bbox([[1.3,0.3],[8.95,2.15]])

		z = img[0,0,0]/255
		ax.set_facecolor((z,z,z))

		fig.savefig(save_filepath + '.png', bbox_inches=bbox)

		plt.close('all')

		self.resize_image(save_filepath)



