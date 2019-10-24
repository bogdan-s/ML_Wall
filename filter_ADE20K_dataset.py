import os 
from shutil import copy2

ade20k_path = 'D:/Python/Datasets/ADE20K_2016_07_26/images/training/a/art_gallery' # where the ADE20K dataset is
# ade20k_path = 'D:/ADE20K_2016_07_26/images/training' # where the ADE20K dataset is

dest = 'D:/Python/DataSets/ADE20K_Filtered' # where the filtered files are copied

labels_to_filter = [" wall ", " ceiling ", " floor "]   # what label do you want to find

file_count = 0

#go through all the folders
for (root,dirs,files) in os.walk(ade20k_path, topdown=True): 
	
	# filter txt files
	txt_files = [f for f in files if f.endswith(".txt")]
	
	for f in txt_files:
		with open(root + '/' + f) as file:  
			data = file.read() 
			if all(labels in data for labels in  labels_to_filter):
				# print (root)
				# print (f)
				file_count += 1
				src_img = root + '/' + f[:-8] + '.jpg'
				copy2(src_img, dest)
				src_seg = root + '/' + f[:-8] + '_seg.png'
				copy2(src_seg, dest)
	print ('--------------------------------')
print("found " + str(file_count) + " files")
