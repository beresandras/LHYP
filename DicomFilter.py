import numpy as np
import pandas as pd
import os
import pydicom
from tqdm import tqdm

class DicomFilter:
    def __init__(self, sourceDirs, keepAttributes=["StudyDescription", "SeriesDescription", "PatientSex", "PatientWeight", "ScanningSequence", "HeartRate", "PatientPosition", "PixelSpacing"], dropAttributes=["PixelData"], recursive=False):
        self.paths = []
        self.filenames = []
        self.data = {}
        self.dataLength = 0

        for sourceDir in sourceDirs:
            if recursive:
                [self._loadDir(subDir[0], dropAttributes, recursive=False) for subDir in os.walk(sourceDir)]
            else:
                self._loadDir(sourceDir, dropAttributes, recursive=False)

        if len(self.filenames) == 0:
            print('Error: no dicom files found in the given directories')

        newData = self.data.copy()
        for attr in self.data:
            if attr not in keepAttributes:
                try:
                    if len(set(self.data[attr])) <= 1:
                        del newData[attr]
                except TypeError:
                    try:
                        if len(set(tuple(e) for e in self.data[attr])) <= 1:
                            del newData[attr]
                    except:
                        del newData[attr]
        self.data = newData

    def _loadDir(self, sourceDir, dropAttributes, recursive):
        filenames = sorted(os.listdir(sourceDir))
        filenames = [filename for filename in filenames if filename.find('.dcm') != -1]

        for filename in filenames:
            dicomData = pydicom.dcmread(os.path.join(sourceDir, filename))
            attributes = [attr for attr in dir(dicomData) if attr[0].isupper()]
            for attr in self.data:
                if attr not in attributes:
                    self.data[attr].append(None)
            for attr in attributes:
                if attr not in dropAttributes:
                    if attr not in self.data:
                        self.data[attr] = [None] * self.dataLength
                    self.data[attr].append(getattr(dicomData, attr))
            self.dataLength += 1

        self.paths.append(len(filenames) * sourceDir)
        self.filenames.append(filenames)

    def getDataFrame(self):
        return pd.DataFrame(data=self.data)

if __name__=='__main__':
    dicomFilter = DicomFilter(["../data/10635813AMR806/"], recursive=True)
    dataFrame = dicomFilter.getDataFrame()
    dataFrame.to_csv("out.csv", sep="\t")