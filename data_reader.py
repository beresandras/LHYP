# Reads the sa folder wiht dicom files and contours
# then draws the contours on the images.

from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw

import os
import pickle
import numpy as np

class PatientData:
    def __init__(self, data_dir, patient_id):

        folder_types = ["la", "lale", "sa", "sale", "tra"]
        self.images = {}
        
        for mode in folder_types:
            self.images[mode] = []

            if mode != "sa":
                image_folder = data_dir + "/" + patient_id + "/" + mode

                # reading the dicom files
                dr = DCMreaderVM(image_folder)
                self.images[mode] = dr.dcm_images

            else:
                image_folder = data_dir + "/" + patient_id + "/sa/images"
                con_file = data_dir + "/" + patient_id + "/sa/contours.con"

                # reading the contours
                cr = CONreaderVM(con_file)
                contours = cr.get_hierarchical_contours()
                self.contours = []

                for slice in contours:
                    maxFrame = None
                    maxArea = 0
                    for frame in contours[slice]:
                        for mode2 in contours[slice][frame]:
                            if mode2 == 'ln': # vörös
                                rgb = [1, 0, 0]
                                area = polygonArea(contours[slice][frame][mode2])
                                if (area > maxArea):
                                    maxFrame = frame
                                    maxArea = area
                            elif mode2 == 'lp': # zöld
                                rgb = [0, 1, 0]
                            else:
                                rgb = None
                            if rgb is not None:
                                self.contours.append(contours[slice][frame][mode2])
                    
                    # reading the dicom files
                    dr = DCMreaderVM(image_folder)
                    self.images[mode].append(dr.get_image(slice, maxFrame))
                            

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


