from flag.randomflag import RandomFlagGenerator
from pathlib import Path

def build_randomGallery(num_train_classes=1, num_test_classes=1, root_dir='./data/', size=(120,120)):
	import cv2

	flagGen = RandomFlagGenerator()
	root_dir = Path(root_dir)
	dataset_dir = root_dir / 'data' / 'flag'
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
	parser.add_argument('--dir', default="./data/")
	parser.add_argument("--size", default=120, type=int)
	args = parser.parse_args()
	if args.gallery:
		build_randomGallery(100, 100, root_dir=args.dir, size=(args.size, args.size))
