import numpy as np
from classes.FileLoader import FileLoader
import sys

if __name__ == '__main__':

    if not len(sys.argv) >= 2:
        print('Usage: python3 main.py $IMG_DIRECTORY')
        quit()

    # Load and labelize images
    f = FileLoader(sys.argv[1])

    # Cross validation 
