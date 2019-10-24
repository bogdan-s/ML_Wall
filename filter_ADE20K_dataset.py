import os 
from shutil import copy2
import cv2
# import multiprocessing
import concurrent.futures


ade20k_path = 'D:/Python/Datasets/ADE20K_2016_07_26/images/training' # where the ADE20K dataset is
# ade20k_path = 'D:/ADE20K_2016_07_26/images/training' # where the ADE20K dataset is

dest = 'D:/Python/DataSets/ADE20K_Filtered' # where the filtered files are copied

labels_to_filter = [" wall ", " ceiling ", " floor "]   # what label do you want to find

file_count = 0

processes = []

def write_binary_mask(root, f, dest):
	# copy the RGB image
	src_img = root + '/' + f[:-8] + '.jpg'
	copy2(src_img, dest)

	# create binary mask for wall
	src_seg = root + '/' + f[:-8] + '_seg.png'
	img = cv2.imread(src_seg)
	h, w, c = img.shape
	for i in range(h):
		for j in range(w):
			if img[i, j, 1] == 162:
				img[i, j] = [255, 255, 255]
			else:
				img[i, j] = [0, 0, 0]

	# write the mask
	cv2.imwrite(dest + '/' + f[:-8] + '_seg.png', img)
	print(dest + '/' + f[:-8] + '_seg.png')



#go through all the folders
for (root,dirs,files) in os.walk(ade20k_path, topdown=True): 
	roots = []
	fs = []
	dests = []
	# filter txt files
	txt_files = [f for f in files if f.endswith(".txt")]
	
	for f in txt_files:
		with open(root + '/' + f) as file:  
			data = file.read() 
			if all(labels in data for labels in  labels_to_filter):
				file_count += 1
				roots.append(root)
				fs.append(f)
				dests.append(dest)

				# if __name__ == '__main__':
				# 	p = multiprocessing.Process( target = write_binary_mask, args =(root, f, dest))
				# 	p.start()
				# 	processes.append(p)
				
				# if file_count == 10:
				# 	break
	if __name__ == '__main__':
		with concurrent.futures.ProcessPoolExecutor() as executor:
			future = executor.map(write_binary_mask, roots, fs, dests)
	# if __name__ == '__main__':
	# 	for process in processes:
	# 		process.join()
	print ('--------------------------------')

print("found " + str(file_count) + " files")
