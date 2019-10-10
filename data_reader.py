from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw

import os
import pickle
import numpy as np

class PatientData:
    def __init__(self, data_dir, patient_id):

        self.data_dir = data_dir
        self.patient_id = patient_id

        self.images_la = []
        self.images_lale = []
        self.images_sa = []
        self.images_sale = []
        self.images_tra = []

        self.load_la()
        self.load_lale()
        self.load_sa()
        self.load_sale()
        self.load_tra()

    def load_la(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/la"

        # reading the dicom files
        dr = DCMreaderVM(image_folder)
        self.images_la = dr.dcm_images

    def load_lale(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/lale"

        # reading the dicom files
        dr = DCMreaderVM(image_folder)
        self.images_lale = dr.dcm_images

    def load_sa(self):
        image_folder = data_dir + "/" + self.patient_id + "/sa/images"
        con_file = data_dir + "/" + self.patient_id + "/sa/contours.con"

        # reading the contours
        cr = CONreaderVM(con_file)
        contours = cr.get_hierarchical_contours()
        self.contours = []

        for slice in contours:
            maxFrame = None
            maxArea = 0
            for frame in contours[slice]:
                for mode in contours[slice][frame]:
                    if mode == 'ln': # vörös
                        area = polygonArea(contours[slice][frame][mode])
                        if (area > maxArea):
                            maxFrame = frame
                            maxArea = area
                    if mode is not 'ln' or mode is not 'lp':
                        self.contours.append(contours[slice][frame][mode])
                
        # reading the dicom files
        dr = DCMreaderVM(image_folder)
        self.images_sa.append(dr.get_image(slice, maxFrame))

    def load_sale(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/sale"

        # reading the dicom files
        dr = DCMreaderVM(image_folder)
        self.images_sale = dr.dcm_images

    def load_tra(self):
        image_folder = self.data_dir + "/" + self.patient_id + "/tra"

        # reading the dicom files
        dr = DCMreaderVM(image_folder)
        self.images_tra = dr.dcm_images
                            

def polygonArea(pointArray):
    np.transpose(pointArray)
    x = np.transpose(pointArray)[0]
    y = np.transpose(pointArray)[1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

data_dir = "../data"
pickle_dir = "../bindata"

patient_ids = [subdir for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]

for pid in patient_ids:
    patient_data = PatientData(data_dir, pid)

    outfile = open(pickle_dir + "/"+ pid, "wb")
    pickle.dump(patient_data, outfile)
    outfile.close()


