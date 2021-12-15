import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedTripletDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions                                                                                                                                                            
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # 在这里获得图片transform的参数
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))                                                                                                                                                                                                                                                                                          

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        由于使用的是triplet，所以在返回的时候是有一共三张图片
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        # 对于triplet图片就只是使用了normalize操作
        print("Aimg")
        print(A_img.size)
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # 对tensor进行拆分，判断crop
        if 'tricut' in self.opt.preprocess:
            print("A")
            print(A.shape)

            w_total = A.size(2)
            w = int(w_total / 3)
            h = A.size(1)
            
            w = int(w_total / 3)
            # h = A_img.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.crop_size - 1))
            h_offset = random.randint(0, max(0, h - self.opt.crop_size - 1))

            A0 = A[:, h_offset:h_offset + self.opt.crop_size, w_offset:w_offset + self.opt.crop_size]

            A1 = A[:, h_offset:h_offset + self.opt.crop_size, w + w_offset:w + w_offset + self.opt.crop_size]

            A2 = A[:, h_offset:h_offset + self.opt.crop_size, 2 * w + w_offset:2 * w + w_offset + self.opt.crop_size]

            w_total = B.size(2)
            w = int(w_total / 3)
            h = B.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.crop_size - 1))
            h_offset = random.randint(0, max(0, h - self.opt.crop_size - 1))

            B0 = B[:, h_offset:h_offset + self.opt.crop_size, w_offset:w_offset + self.opt.crop_size]

            B1 = B[:, h_offset:h_offset + self.opt.crop_size, w + w_offset:w + w_offset + self.opt.crop_size]

            B2 = B[:, h_offset:h_offset + self.opt.crop_size, 2 * w + w_offset:2 * w + w_offset + self.opt.crop_size]


            return {'A0': A0, 'A1': A1, 'A2': A2, 'B0': B0, 'B1': B1, 'B2': B2, 'A_paths': A_path, 'B_paths': B_path}

        # elif self.opt.preprocess == 'scale_width_trinone':
        else:
            w_total = A_img.size(2)
            w = int(w_total / 3)
            h = A_img.size(1)

            A0 = A_img[:, :, 0: w]

            A1 = A_img[:, :, w: 2 * w]

            A2 = A_img[:, :, 2 * w: 3 * w]

            w_total = B_img.size(2)
            w = int(w_total / 3)
            h = B_img.size(1)
            B0 = A_img[:, :, 0: w]

            B1 = A_img[:, :, w: 2 * w]

            B2 = A_img[:, :, 2 * w: 3 * w]

            return {'A0': A0, 'A1': A1, 'A2': A2, 'B0': B0, 'B1': B1, 'B2': B2, 'A_paths': A_path, 'B_paths': B_path}




        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
