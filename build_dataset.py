from flag.randomflag import RandomFlagGenerator
from flag.utils import bb_intersection_over_union

from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import datetime

from torchvision import transforms

class FlagBuilder:
	def __init__(self, root_dir='./', name=""):
		self.root_dir = Path(root_dir)
		(self.root_dir/'data').mkdir(exist_ok=True)

		self.transforms = transforms.Compose([
	                transforms.ToPILImage(),
	#                 transforms.RandomRotation(15,expand=np.random.uniform()>0.5),
	                # transforms.ColorJitter(0.5,0.5,0.5,0.1),
	                transforms.RandomAffine(15,shear=20),
	            ])

	def load_image(self, img_path):
		img = cv2.imread(img_path)
		# img = np.array(Image.open(img_path))
		return img

	def save_image(self, img, img_path):
		cv2.imwrite(img_path, img)

	def build(self, name, iter_img_paths, exist_ok=False, num_train_classes=100, num_test_classes=100, 
				num_flags=5, scaleRange=(0.07,0.11)):
		self.dataset_dir = self.root_dir / 'data' / name
		try:
			self.dataset_dir.mkdir(exist_ok=exist_ok)
		except:
			raise Exception("{} exists!".format(name))
		self.build_randomGallery(num_train_classes, num_test_classes)

		print("Built train and test galleries!")

		iter_img_paths = list(iter_img_paths)
		for phase in ['train','test']:
			self.build_randomDataset(num_flags, iter_img_paths, phase, scaleRange)
			print('Built {} dataset!'.format(phase))

		print('Done!')


	def build_randomGallery(self, num_train_classes=1, num_test_classes=1, size=(120,120)):
		flagGen = RandomFlagGenerator()
		for phase in ['train','test']:
			num_classes = eval('num_{}_classes'.format(phase))
			gallery_dir = self.dataset_dir / phase / 'gallery'
			gallery_dir.mkdir(exist_ok=True, parents=True)
			for idx in range(num_classes):
				img = flagGen.getRandomFlag(size)
				self.save_image(img, str(gallery_dir/'{}.png'.format(idx)))

	def load_gallery(self, phase):
		gallery_dir = self.dataset_dir / phase / 'gallery'
		gallery = {}
		for path in gallery_dir.glob('*'):
			ind = int(path.name.split('.')[0])
			img = self.load_image(str(path))
			gallery[ind] = img
		return gallery

	def random_insert_flag(self, img, flag, scale=0.07, prevbboxes=None):
		size = min(img.shape[:2])
		flag_height = size * scale
		flag_scale = np.random.uniform(0.8, 1.2)
		flag_width = int(flag_height * flag_scale)
		flag_height = int(flag_height)

		while True:
			pos = (np.random.choice(img.shape[0]-flag_height), np.random.choice(img.shape[1]-flag_width))
			bbox = [pos[1],pos[0],pos[1]+flag_width,pos[0]+flag_height]  # xmin, ymin, xmax, ymax

			if prevbboxes is not None:
				for prevbbox in prevbboxes:
					iou = bb_intersection_over_union(prevbbox, bbox)
					if iou > 0.1:
						continue
			break

		flag = np.array(self.transforms(flag))
		flag = cv2.resize(flag, dsize=(flag_width, flag_height))

		mask = (flag==0)
		img[bbox[1]:bbox[3], bbox[0]:bbox[2]] *= mask
		img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += flag
		return img, bbox

	def random_insert_multiflags(self, img, gallery, num_flags, scaleRange):
		flagIndices = list(gallery.keys())
		bboxList = []
		for __ in range(num_flags):
			scale = np.random.uniform(scaleRange[0], scaleRange[1])
			flagIdx = np.random.choice(flagIndices)
			flag = gallery[flagIdx].copy()
			img, bbox = self.random_insert_flag(img, flag, scale=scale, prevbboxes=bboxList)
			bboxList.append(bbox)
		return img, bboxList

	def build_randomDataset(self,num_flags,iter_img_paths,phase, scaleRange):
		imgs_dir = self.dataset_dir / phase / 'imgs'
		imgs_dir.mkdir()

		gallery = self.load_gallery(phase)
		infoList = []
		for idx, path in tqdm(enumerate(iter_img_paths)):
			img = self.load_image(str(path))
			img, bboxList = self.random_insert_multiflags(img, gallery, num_flags, scaleRange)
			save_path = imgs_dir / '{}.png'.format(idx)
			self.save_image(img, str(save_path))
			info = {
				'index': idx,
				'path': str(save_path),
				'source': str(path),
				'bboxList': bboxList,
			}
			infoList.append(info)
		import json
		jsonStr = json.dumps(infoList)
		with open(str(self.dataset_dir / phase / 'infoList.json'), 'w') as hd:
			hd.write(jsonStr)






if __name__ == '__main__':
	builder = FlagBuilder('/home/huangyucheng/MYDATA/dl-github/Siamese-RPN')
	iter_img_paths = Path('/home/huangyucheng/MYDATA/DATASETS/PASCAL_VOC/VOCdevkit/VOC2007/JPEGImages').glob('*.jpg')
	builder.build('5flags', iter_img_paths)

	# builder = FlagBuilder('/home/huangyucheng/MYDATA/dl-github/Siamese-RPN')
	# iter_img_paths = Path('/home/huangyucheng/MYDATA/DATASETS/PASCAL_VOC/VOCdevkit/VOC2007/JPEGImages').glob('*00.jpg')
	# builder.build_randomDataset(5, iter_img_paths, 'train')

	# import argparse
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--gallery', default=False, action="store_true")
	# parser.add_argument('--dir', default="./")
	# parser.add_argument("--size", default=120, type=int)
	# args = parser.parse_args()
	# if args.gallery:
	# 	builder = FlagBuilder(args.dir)
	# 	builder.build_randomGallery(100, 100, size=(args.size, args.size))
