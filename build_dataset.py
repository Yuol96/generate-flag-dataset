from flag.randomflag import RandomFlagGenerator
from pathlib import Path

def build_randomGallery(num_train_classes=1, num_test_classes=1, dataset_dir='./data/', size=(120,120)):
	import cv2

	flagGen = RandomFlagGenerator()
	dataset_dir = Path(dataset_dir)
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
	build_randomGallery(100, 100)
