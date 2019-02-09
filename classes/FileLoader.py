"""
This class purpose is to load the .png image of the parasitized and uninfected cells
It should also be able to labelized images as follow:
- Parasitized => 1
- Uninfected => 0

This class takes as input the filepath where it can find the images
"""
import os
import numpy as np
from PIL import Image
import cv2 as cv

class FileLoader():
    def __init__(self, filepath):
        self.filepath = filepath

        # Check if the directory exists or not
        if not os.path.isdir(self.filepath):
            print(self.filepath, ': directory not found')
            quit()

        print('File found')
        
    def loadImage(self):
        # Load paratized images
        parasitized_path = self.filepath + 'Parasitized/'
        path, dirs, files = (next(os.walk(parasitized_path)))

        # Create train base
        cpt = 0
        for img in files:
            if (cpt == np.floor(len(files) * 2/3)):
                break
            print(img)
            img_ = cv.imread(parasitized_path + img)
            foo = np.asarray(Image.open(parasitized_path + img))
            print(img_.shape)
            print(foo.shape)
            gray_image = cv.cvtColor(img_, cv.COLOR_RGB2GRAY)
            
            break;
            cpt = cpt + 1


    def labelizeImage(self):
        pass


