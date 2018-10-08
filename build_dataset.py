from flag.randomflag import RandomFlagGenerator
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

	def random_insert_flag(self, img, gallery, flagIdx, scale=0.2):
		size = min(img.shape[:2])
		flag_height = size * scale
		flag_scale = np.random.uniform(0.8, 1.2)
		flag_width = int(flag_height * flag_scale)
		flag_height = int(flag_height)
		flag = gallery[flagIdx].copy()
		flag = np.array(self.transforms(flag))
		flag = cv2.resize(flag, dsize=(flag_width, flag_height))
		mask = (flag==0)
		pos = (np.random.choice(img.shape[0]-flag_height), np.random.choice(img.shape[1]-flag_width))
		bbox = {
			'xmin': pos[1],
			'ymin': pos[0],
			'xmax': pos[1]+flag_width,
			'ymax': pos[0]+flag_height,
		}
		img[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] *= mask
		img[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] += flag
		return img, bbox

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


# def build_oneObjectDataset(iter_img_path):
# 	img_paths = list(iter_img_path)
# 	for path in img_paths:



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
