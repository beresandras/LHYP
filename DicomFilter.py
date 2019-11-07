import numpy as np
import pandas as pd
import os
import pydicom

class DicomFilter:
    def __init__(self, sourceDirs, recursive=True):
        self.paths = []
        self.filenames = []
        self.data = {}

        for sourceDir in sourceDirs:
            self._loadDir(sourceDir, recursive)

        if len(self.filenames) == 0:
            print('Error: no dicom files found in the given directories')

    def _loadDir(self, sourceDir, recursive=True):
        filenames = sorted(os.listdir(sourceDir))
        filenames = [filename for filename in filenames if filename.find('.dcm') != -1]


        
        for filename in filenames:
            dicomData = pydicom.dcmread(os.path.join(sourceDir, filename))
            print(dicomData)

        self.paths.append(len(filenames) * sourceDir)
        self.filenames.append(filenames)

