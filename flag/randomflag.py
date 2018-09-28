import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid

class RandomFlagGenerator:
    def __init__(self, colorset=None):
        """
        colorset should be a (n, 3) array
        """
        import matplotlib
        self.name2hex = {name:hexStr for name, hexStr in matplotlib.colors.cnames.items()}
        if colorset is None:
            self.colorset = list(map(self.hex2RGB, self.name2hex.values()))
        else:
            for idx,color in enumerate(colorset):
                if isinstance(color, str):
                    if color.startswith('#'):
                        colorset[idx] = self.hex2RGB(color)
                    elif color in self.name2hex:
                        colorset[idx] = self.hex2RGB(self.name2hex[color])
                    else:
                        raise ValueError("'{}'' is a invalid color".format(color))
            self.colorset = colorset
            print(self.colorset)

        self.transforms = transforms.Compose([
                transforms.ToPILImage(),
#                 transforms.RandomRotation(15,expand=np.random.uniform()>0.5),
                transforms.ColorJitter(0.5,0.5,0.5,0.1),
                transforms.RandomAffine(15,shear=20),
            ])

    @classmethod
    def hex2RGB(self, hexStr):
        assert hexStr[0] == '#' and len(hexStr) == 7
        return int(hexStr[1:3], base=16), int(hexStr[3:5], base=16), int(hexStr[5:7], base=16)

    @classmethod
    def displaySingleImg(self, img, name=None):
        import cv2

        if not name:
            name = 'Example'
        # from IPython import embed
        # embed()
        cv2.imshow(name, img)
        return cv2.waitKey(0)

    def getRandomFlag(self, size):
        """
        size should be a tuple of (Height, width)
        """
        img = np.zeros((*size, 3), dtype=np.uint8)
        delta_height = size[0]//3
        delta_width = size[1]//3
        grids_color_indices = np.random.choice(len(self.colorset), 9, replace=True)
        for idx in range(9):
            grid_i = idx//3
            grid_j = idx%3
            color = np.array(self.colorset[grids_color_indices[idx]], dtype=np.uint8)
            assert color.shape == (3, )
            img[grid_i*delta_height:(grid_i+1)*delta_height, grid_j*delta_width:(grid_j+1)*delta_width] = color

        # self.displaySingleImg(img)
        img[img==0] = 1
        return img

    # def randomTransformFlag(self, img):

def test_transform():
    from PIL import Image
    fgen = RandomFlagGenerator(None)
    while True:
        imgs = [fgen.getRandomFlag((120,120))]
        imgs.append(fgen.transforms(Image.fromarray(imgs[0])))
        # imgs = np.stack(imgs)
        # imgs = torch.Tensor(imgs).byte()
        # imgs = imgs.permute(0,3,1,2)
        # imgs = make_grid(imgs)
        # imgs = imgs.permute(1,2,0)
        ret = RandomFlagGenerator.displaySingleImg(imgs.numpy())
        if ret == ord('q'):
            break


def test_getRandomFlag():
    colorset = None
    fgen = RandomFlagGenerator(colorset)
    while True:
        try:
            fgen.getRandomFlag((120,120))
        except:
            break



if __name__ == '__main__':
    test_transform()







