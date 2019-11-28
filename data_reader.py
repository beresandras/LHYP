from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw

import os
import pickle
import numpy as np
import pydicom
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['image.cmap'] = 'Greys'
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 8, 6

def loadDicom(folder_name):
    images = []
    sliceLocations = []

    dcm_files = os.listdir(folder_name)
    dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) < 4]
    if len(dcm_files) == 0:  # sometimes the order number is missing at the end
        dcm_files = os.listdir(folder_name)

    for file in dcm_files:
        if file.find('.dcm') != -1:
            try:
                temp_ds = pydicom.dcmread(os.path.join(folder_name, file))
                images.append(temp_ds.pixel_array)
                sliceLocations.append(int(temp_ds.SliceLocation))
            except Exception as e:
                #print(e)
                pass
            
    return (images, sliceLocations)

class PatientData:
    def __init__(self, data_dir, patient_id):

        self.data_dir = data_dir
        self.patient_id = patient_id

        self.images_sa = []
        self.images_sale = []
        self.images_la = []
        self.images_lale = []

        self.contours_sa = []

    def load_all(self):
        print("\n" + self.patient_id)
        self.load_sa()
        #self.load_sale()
        #self.load_la()
        #self.load_lale()

    def load_sa(self):
        image_folder = data_dir + "/" + self.patient_id + "/sa/images"
        contour_file = data_dir + "/" + self.patient_id + "/sa/contours.con"

        print(" SA: ", end="")

        # reading the contours
        if (os.path.isfile(contour_file) == False):
            print("Contour file does not exist")
            return
        cr = CONreaderVM(contour_file)
        contours = cr.get_hierarchical_contours()
                
        # reading the dicom files
        if len(os.listdir(image_folder)) == 0:
            print("Image folder is empty")
        else:
            dr = DCMreaderVM(image_folder)
            if (dr.broken):
                print("Dicom reader unsuccessful")
            else:
                for slice in contours:
                    maxFrame = None
                    maxArea = 0
                    for frame in contours[slice]:
                        if "ln" in contours[slice][frame]: # ha van vörös kontúr
                            area = polygonArea(contours[slice][frame]["ln"])
                            if (area > maxArea):
                                maxFrame = frame
                                maxArea = area
                    if maxFrame is not None:
                        self.images_sa.append(dr.get_image(slice, maxFrame))
                        self.contours_sa.append(contours[slice][maxFrame]["ln"])
                print("{} images and contours added".format(len(self.images_sa)))

    def load_sale(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/sale"
        
        print(" SALE: ", end="")

        # reading the dicom files
        if len(os.listdir(image_folder)) == 0:
            print("Image folder is empty")
        else:
            (images, sliceLocations) = loadDicom(image_folder)
            nullLocations = [i for i, sl in enumerate(sliceLocations) if sl == 0]
            if len(nullLocations) == 1:
                self.images_sale = images
            elif len(nullLocations) > 1:
                self.images_sale = images[:nullLocations[1]]
            print("OK")

    def load_la(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/la"

        # reading the dicom files
        if len(os.listdir(image_folder)) > 0:
            (images, sliceLocations) = loadDicom(image_folder)
            if (len(images) != 0):
                self.images_la = images[0]

    def load_lale(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/lale"

        # reading the dicom files
        if len(os.listdir(image_folder)) > 0:
            (images, sliceLocations) = loadDicom(image_folder)
            nullLocations = [i for i, sl in enumerate(sliceLocations) if sl == 0]
            if len(nullLocations) == 1:
                self.images_lale = images[nullLocations[1]:]
            elif len(nullLocations) > 2:
                self.images_lale = images[nullLocations[1]:nullLocations[2]]
                            

def polygonArea(pointArray):
    x = np.transpose(pointArray)[0]
    y = np.transpose(pointArray)[1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

if __name__=='__main__':
    data_dir = "../data"
    pickle_dir = "../bindata"

    patient_ids = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

    for pid in patient_ids:
        patient_data = PatientData(data_dir, pid)
        patient_data.load_all()

        outfile = open(pickle_dir + "/"+ pid, "wb")
        pickle.dump(patient_data, outfile)
        outfile.close()


