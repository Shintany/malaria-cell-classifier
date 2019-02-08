"""
This class purpose is to load the .png image of the paratized and uninfected cells
It should also be able to labelized images as follow:
- Paratized => 1
- Uninfected => 0

This class takes as input the filepath where it can find the images
"""
import os

class FileLoader():
    def __init__(self, filepath):
        self.filepath = filepath
        self.paratized = 'Paratized'
        self.paratized = 'Uninfected'
        
        # Check if the directory exists or not
        if not os.path.isdir(self.filepath):
            print(self.filepath, ': directory not found')
            quit()
        
        print('File found')
        



