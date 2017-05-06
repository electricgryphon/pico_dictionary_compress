#python image convert to pico-8
import matplotlib.image as mpimg
import numpy as np
import sys
import math
import random

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

num_to_string_list=["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"]

encode_string=""
k_max_depth=6

#input_file = sys.argv[1]
 

def cluster_points(X, mu):
	clusters  = {}
	for x in X:
		bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
		for i in enumerate(mu)], key=lambda t:t[1])[0]
		try:
			clusters[bestmukey].append(x)
		except KeyError:
			clusters[bestmukey] = [x]
	return clusters

	
def reevaluate_centers(mu, clusters):
	newmu = []
	keys = sorted(clusters.keys())
	for k in keys:
		newmu.append(np.mean(clusters[k], axis = 0))
	return newmu

def has_converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(x,k):
	# Initialize to K random centers
	oldmu = random.sample(x, k)
	mu = random.sample(x, k)
	while not has_converged(mu, oldmu):
		oldmu = mu
		# Assign all points in X to clusters
		clusters = cluster_points(x, mu)
		# Reevaluate centers
		mu = reevaluate_centers(oldmu, clusters)
	return(mu, clusters)

def load_image(image,target):
	for i in range(0,128):
		for j in range(0,128):
			c=round(image[int(j)][int(i)][0])
			target[j][i]=c

			
def diff_image(first,second,target):
	for i in range(0,128):
		for j in range(0,128):
			c1=first[int(j)][int(i)]
			c2=second[int(j)][int(i)]
			if(c1==c2):
				target[j][i]=3
			else:
				target[j][i]=c2


def tohex(val, nbits):
	return hex((val + (1 << nbits)) % (1 << nbits))
	

			

class	block_data(object):
	def __init__(self,initial_data):
		self.data=np.copy(initial_data)
		self.count=1
		self.score=0
		
	def __repr__(self):
		return '\n{}\ncount: {} score: {}\n'.format(self.data,self.count,self.score)
		
	def __cmp__(self,other):
		if hasattr(other,'count'):
			return self.score.__cmp__(other.score)



			

def	grab_block(image,x,y,w,h):
	new_array=np.full((w*h),0)
	for i in range(0,w):
		for j in range(0,h):
			new_array[j*w+i]=image[j+y][i+x]
	return new_array
			

def compare_block(block_a,block_b):
	return ((block_a==block_b).all())		


all_blocks=[]
def read_image_blocks(image):
		for i in range(0,16):
			for j in range(0,16):
				new_block=grab_block(image,i*8,j*8,8,8)
				duplicate=False
				for comp in all_blocks:
					if(compare_block(new_block,comp)):
						duplicate=True
				if(not duplicate):	
					all_blocks.append(new_block)
				

block_dictionary=[]
def	create_dictionary(list,count):
	global block_dictionary
	block_dictionary=find_centers(list,count)[0]
	for i in range(0,len(block_dictionary)):
		block_dictionary[i]=block_dictionary[i].round()
	#block_dictionary.append(np.full((64),0) )
	#block_dictionary.append(np.full((64),1) )

def find_nearest(target,dictionary):
	shortest_distance=1000
	shortest_index=0
	for i in range(0,len(dictionary)):
		dist=np.linalg.norm(dictionary[i]-target)
		if(dist<shortest_distance):
			shortest_index=i
			shortest_distance=dist
	return shortest_index
	
def quant_image(image,dictionary,bit_len):
	encode_string=""
	new_image = np.full((128,128),0)
	for i in range(0,16):
		for j in range(0,16):
			new_block=grab_block(image,i*8,j*8,8,8)
			close_block=find_nearest(new_block,dictionary)
			encode_string+=tobinary(close_block,bit_len)
			new_image[j*8:j*8+8,i*8:i*8+8]=dictionary[close_block].reshape(8,8)
	
	#return new_image
	#fig = plt.imshow(new_image, cmap='gray')
	#plt.axis('off')
	#fig.axes.get_xaxis().set_visible(False)
	#fig.axes.get_yaxis().set_visible(False)
	#plt.savefig("out.png",bbox_inches='tight', pad_inches = 0)
	
	return encode_string
				
	
def output_dictionary(dictionary):
	encode_string=""
	for entry in dictionary:
		for b in entry:
			#print b
			if(b==0):
				encode_string+="0"
			else:
				encode_string+="1"
	#print(encode_string)
	#print(convert_bit_string(encode_string))
	return encode_string #rle_bit_string(encode_string,8,4)
	

def tohex(val):
	h = hex((int(val,2) + (1 << 16)) % (1 << 16))
	h =h[2:]
	h=h.zfill(2)
	return h

def tobinary(val,bit_len):
	if(val>=pow(2,bit_len)):raise ValueError('to binary: val exceeds len')
	return(('{0:0'+str(bit_len)+'b}').format(val))
	
	
def convert_bit_string(string):
	#global output_file
	encode_string=""
	while(len(string)>=8):
		l=string[0:8]
		string=string[8:]
		#print(tohex(l))
		encode_string+=(tohex(l))
		#output_file.write(tohex(l))
	if(len(string)>0):
		encode_string+=tohex(string.ljust(8, '0'))
		#output_file.write(tohex(string.ljust(8, '0')))
	return encode_string

def rle_bit_string(string,window,run_length):
	#print(string)
	encode_string=""
	last_l=""
	while(len(string)>=window):
		l=string[0:window]
		string=string[window:]	
		encode_string+=(l)
		if(l==last_l):
			#print("match")
			repeat=True
			repeat_count=0
			while(len(string)>=window and repeat):
				if(repeat_count<pow(2,run_length)-1):
					l=string[0:window]
					if(l==last_l):
						string=string[window:]	
						repeat_count+=1
					else:
						repeat=False
						l=""
				else:
					repeat=False
					l=""
			#print(repeat_count)
			encode_string+=(tobinary(repeat_count,run_length))
		
		last_l=l
	return encode_string
	#if(len(string)>0:
	
	
global output_file
def main():
	random.seed(1)
	global output_file
	global block_dictionary
	
	image_first = np.full( (128,128),0)
	image_last = np.full( (128,128),0)
	image_difference = np.full( (128,128),0)
	
	input_name = sys.argv[1]
	frame_start = int(sys.argv[2])
	frame_end = int(sys.argv[3])
	depth = int(sys.argv[4])
	
	output_filename=input_name+"_"+str(pow(2,depth))+"bit_"+str(frame_start)+"-"+str(frame_end)+".txt"
	output_file = open(output_filename,'w')	


	
	print("\nbuilding dictionary")
	#end=min(80,frame_count-1)
	for cur_frame in range(frame_start, frame_end):
		file_a= input_name+"_"+str(cur_frame).zfill(4)+".png"
		source=mpimg.imread(file_a)
		load_image(source,image_first)
		read_image_blocks(image_first)
		print("."),
	
	print("\noptimizing dictionary")
	create_dictionary(all_blocks,pow(2,depth))
	output_file.write("dictionary:\n")
	output_s=output_dictionary(block_dictionary)
	output_file.write(convert_bit_string(rle_bit_string(output_s,8,5)))
	output_file.write("\n\n")
	
	print("compressing frames")
	for cur_frame in range(frame_start,frame_end):
		file_a= input_name+"_"+str(cur_frame).zfill(4)+".png"
		source=mpimg.imread(file_a)
		load_image(source,image_first)
		im_s=quant_image(image_first,block_dictionary,depth)
		output_file.write('"')
		output_file.write(convert_bit_string(rle_bit_string(im_s,depth,5)))
		output_file.write('",\n')
		print("."),
main()
	