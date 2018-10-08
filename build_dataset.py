from flag.randomflag import RandomFlagGenerator
from flag.utils import bb_intersection_over_union

from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from torchvision import transforms

class FlagBuilder:
	def __init__(self, root_dir='./'):
		self.root_dir = Path(root_dir)

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

	def load_gallery(self, phase):
		gallery_dir = self.root_dir / 'data' / 'flag' / 'gallery' / phase
		gallery = {}
		for path in gallery_dir.glob('*'):
			ind = int(path.name.split('.')[0])
			img = self.load_image(str(path))
			gallery[ind] = img
		return gallery

	def random_insert_flag(self, img, gallery, flagIdx, scale=0.07, prevbboxes=None):
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

		flag = gallery[flagIdx].copy()
		flag = np.array(self.transforms(flag))
		flag = cv2.resize(flag, dsize=(flag_width, flag_height))

		mask = (flag==0)
		# bbox = {
		# 	'xmin': pos[1],
		# 	'ymin': pos[0],
		# 	'xmax': pos[1]+flag_width,
		# 	'ymax': pos[0]+flag_height,
		# }
		# img[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] *= mask
		# img[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] += flag
		img[bbox[1]:bbox[3], bbox[0]:bbox[2]] *= mask
		img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += flag
		return img, bbox

	def random_insert_multiflags(self, img, gallery, num_flags, scaleRange=(0.07, 0.15)):
		flagIndices = list(gallery.keys())
		bboxList = []
		for __ in range(num_flags):
			scale = np.random.uniform(scaleRange[0], scaleRange[1])
			flagIdx = np.random.choice(flagIndices)
			img, bbox = self.random_insert_flag(img, gallery, flagIdx, scale=scale, prevbboxes=bboxList)
			bboxList.append(bbox)
		return img, bboxList

	def build_randomDataset(self,num_flags,iter_img_paths):
		for idx, path in enumerate(iter_img_paths):
			pass


	def build_randomGallery(self, num_train_classes=1, num_test_classes=1, size=(120,120)):
		flagGen = RandomFlagGenerator()
		dataset_dir = self.root_dir / 'data' / 'flag'
		gallery_dir = dataset_dir / 'gallery'
		(gallery_dir/'train').mkdir(exist_ok=True, parents=True)
		(gallery_dir/'test').mkdir(exist_ok=True, parents=True)
		for idx in range(num_train_classes):
			img = flagGen.getRandomFlag(size)
			cv2.imwrite(str(gallery_dir/'train'/'{}.png'.format(idx)), img)
		for idx in range(num_test_classes):
			img = flagGen.getRandomFlag(size)
			cv2.imwrite(str(gallery_dir/'test'/'{}.png'.format(idx)), img)




if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--gallery', default=False, action="store_true")
	parser.add_argument('--dir', default="./")
	parser.add_argument("--size", default=120, type=int)
	args = parser.parse_args()
	if args.gallery:
		builder = FlagBuilder(args.dir)
		builder.build_randomGallery(100, 100, size=(args.size, args.size))
