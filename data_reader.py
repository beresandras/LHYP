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

def loadDicomGeneric(folder_name):
    images = []
    sliceLocations = []
    acquisitionNumbers = []
    instanceNumbers = []

    dcm_files = sorted(os.listdir(folder_name))
    dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) < 4]
    if len(dcm_files) == 0:  # sometimes the order number is missing at the end
        dcm_files = sorted(os.listdir(folder_name))

    for file in dcm_files:
        if file.find('.dcm') != -1:
            try:
                temp_ds = pydicom.dcmread(os.path.join(folder_name, file))
                images.append(temp_ds.pixel_array)
                sliceLocations.append(int(temp_ds.SliceLocation))
                acquisitionNumbers.append(int(temp_ds.AcquisitionNumber))
                instanceNumbers.append(int(temp_ds.InstanceNumber))
            except Exception as e:
                #print(e)
                pass

    images = [img for img, acqNum, instNum in sorted(zip(images, acquisitionNumbers, instanceNumbers), key=lambda elem: elem[1] * (max(instanceNumbers) + 1) + elem[2])]
    sliceLocations = [sliceLoc for sliceLoc, acqNum, instNum in sorted(zip(sliceLocations, acquisitionNumbers, instanceNumbers), key=lambda elem: elem[1] * (max(instanceNumbers) + 1) + elem[2])]
            
    return (images, sliceLocations)

class PatientData:
    def __init__(self, data_dir, patient_id):

        self.data_dir = data_dir
        self.patient_id = patient_id

        self.images_sa = []
        self.images_sale = []
        self.images_la_2C = []
        self.images_la_4C = []
        self.images_la_3C = []
        self.images_lale = []

        self.contours_sa = []
        self.meta_str = ""

    def load_all(self):
        print("\n" + self.patient_id)
        self.load_sa()
        self.load_sale()
        self.load_la()
        self.load_lale()
        self.load_meta()

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
            (images, sliceLocations) = loadDicomGeneric(image_folder)
            if len(images) == 0:
                print("No image could be loaded successfully")
            else:
                nullLocations = [i for i, sl in enumerate(sliceLocations) if sl == 0]
                if len(nullLocations) <= 1:
                    self.images_sale = images
                elif len(nullLocations) > 1:
                    self.images_sale = images[:nullLocations[1]]
                print("{} images added".format(len(self.images_sale)))

    def load_la(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/la"

        print(" LA: ", end="")

        # reading the dicom files
        if len(os.listdir(image_folder)) == 0:
            print("Image folder is empty")
        else:
            (images, sliceLocations) = loadDicomGeneric(image_folder)
            if len(images) == 0:
                print("No image could be loaded successfully")
            else:
                endDiastoleLocations = [[] for i in range(3)] # 2C, 4C, 3C
                prevSl = None
                currentChamberIndex = 0
                for i, sl in enumerate(sliceLocations):
                    if sl != prevSl:
                        endDiastoleLocations[currentChamberIndex].append(i)
                        currentChamberIndex = (currentChamberIndex + 1) % 3
                    prevSl = sl
                if not (len(endDiastoleLocations[0]) == len(endDiastoleLocations[1]) and len(endDiastoleLocations[1]) == len(endDiastoleLocations[2])):
                    print("LA image distribution is not standard (2C-4C-3C)")
                else:
                    for i in endDiastoleLocations[0]:
                        self.images_la_2C.append(images[i])
                        self.images_la_4C.append(images[i])
                        self.images_la_3C.append(images[i])
                    print("{} + {} + {} images added".format(len(self.images_la_2C), len(self.images_la_4C), len(self.images_la_3C)))

    def load_lale(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/lale"

        print(" LALE: ", end="")

        # reading the dicom files
        if len(os.listdir(image_folder)) == 0:
            print("Image folder is empty")
        else:
            (images, sliceLocations) = loadDicomGeneric(image_folder)
            if len(images) == 0:
                print("No image could be loaded successfully")
            else:
                self.images_lale = images[(len(images) // 3) : (2 * len(images) // 3)]
                print("{} images added".format(len(self.images_lale)))

    def load_meta(self):
        meta_path = self.data_dir + "/" + self.patient_id + "/meta.txt"
        with open(meta_path, 'r') as meta_file:
            self.meta_str = meta_file.read()

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


