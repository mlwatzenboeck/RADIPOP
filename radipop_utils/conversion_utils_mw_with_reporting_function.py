from PIL import Image
from scipy import ndimage as ndi
from skimage import data, morphology, feature, filters
from skimage import io as skio
from skimage.filters import sobel, roberts, prewitt, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, opening, square, remove_small_holes, remove_small_objects
import numpy as np
import os
import pydicom
import sys
import pickle
import imageio as io
import matplotlib.pyplot as plt

#######note 20.9.2020
#since there are some errors with some images, try to implement conditional statement to reduce those
def create_images_for_display(name, input_dir=None, output_dir=None):
    '''
    Convert images from DICOM to png.
    '''
    name = str(name)
    if input_dir == None:
        input_dir = "data/ct_dicom/"
    path = input_dir  + str(name)

    if output_dir == None:
        output_dir = "assets/niftynet_raw_imagesz"
        os.makedirs(os.path.join(output_dir, str(name)), exist_ok=True)
    files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not files:
        print("no files found at ", os.path.join(path, f))
        return
    patient_position = files[0].PatientPosition
    # Patient scans are either "feet first" (FFS) or "head first" (HFS). If FFS, they must be reversed.
    reverse = patient_position == "FFS"
    files.sort(key=lambda x: float(x.SliceLocation), reverse=reverse)

    num_slides = len(files)
    
    
   ##########################here comes the 20/9/20 conditional statement
    #it tries to do the whole thing, but if the UnboundLocalError happens, it reduces the images#######################
    max_frame = [10000, 10000, 0, 0]

    try:
        for i, file in enumerate(files):
        #print(i)
            orig = extract_pixels_for_viewing(file)
            orig, frame = trim_background(orig)
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])
    except UnboundLocalError:
        ####TEST 24/7/21
        print("calling NEW altered trim background fun")
        #files = files[:i]
    #################################################################################################################     
    #####note: prior to implementing the conditioning, there was no indent here
        max_frame = [10000, 10000, 0, 0]
        for i, file in enumerate(files):
            orig = extract_pixels_for_viewing(file)
            ####TEST 24/7/21
            orig, frame = trim_background(orig, [0, 0, 512, 512])
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])


    for i, file in enumerate(files):

        ext = i
        orig = extract_pixels_for_viewing(file)
        aim = Image.fromarray(orig, mode='L')
        orig, _ = trim_background(orig, max_frame)
        aim = Image.fromarray(orig, mode='L')
        aim.save(os.path.join(output_dir, str(name), str(ext)+".png"), format='PNG')
    return "success"

##version with better input dir without the name stuff
def create_images_for_display_v2(input_dir=None, output_dir=None, name = "XX"):
    '''
    Convert images from DICOM to png.
    '''
    
    path = input_dir

    if output_dir == None:
        output_dir = "assets/niftynet_raw_imagesz"
    os.makedirs(os.path.join(output_dir, str(name)), exist_ok=True)
    files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not files:
        print("no files found at ", os.path.join(path, f))
        return
    patient_position = files[0].PatientPosition
    # Patient scans are either "feet first" (FFS) or "head first" (HFS). If FFS, they must be reversed.
    reverse = patient_position == "FFS"
    files.sort(key=lambda x: float(x.SliceLocation), reverse=reverse)

    num_slides = len(files)
    
    
   ##########################here comes the 20/9/20 conditional statement
    #it tries to do the whole thing, but if the UnboundLocalError happens, it reduces the images#######################
    max_frame = [10000, 10000, 0, 0]

    try:
        for i, file in enumerate(files):
        #print(i)
            orig = extract_pixels_for_viewing(file)
            orig, frame = trim_background(orig)
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])
    except UnboundLocalError:
        print(max_frame)
        ####TEST 24/7/21
        print("calling NEW altered trim background fun")
        #files = files[:i]
    #################################################################################################################     
    #####note: prior to implementing the conditioning, there was no indent here
        max_frame = [10000, 10000, 0, 0]
        for i, file in enumerate(files):
            orig = extract_pixels_for_viewing(file)
            ####TEST 24/7/21
            orig, frame = trim_background(orig, [0, 0, 512, 512])
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])
    
    print("Max frame = ", max_frame)


    for i, file in enumerate(files):

        ext = i
        orig = extract_pixels_for_viewing(file)
        aim = Image.fromarray(orig, mode='L')
        orig, _ = trim_background(orig, max_frame)
        print("Shape of orig after trim background:", orig.shape)
        aim = Image.fromarray(orig, mode='L')
        aim.save(os.path.join(output_dir, str(name), str(ext)+".png"), format='PNG')
    return "success"


#same function as above, but it does not save the images, only reports the coordinates of the reduced images

def report_coordinates(path):
    '''
    Convert images from DICOM to png.
    '''
    files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not files:
        print("no files found at ", os.path.join(path, f))
        return
    patient_position = files[0].PatientPosition
    # Patient scans are either "feet first" (FFS) or "head first" (HFS). If FFS, they must be reversed.
    reverse = patient_position == "FFS"
    files.sort(key=lambda x: float(x.SliceLocation), reverse=reverse)

    num_slides = len(files)
    
    
   ##########################here comes the 20/9/20 conditional statement
    #it tries to do the whole thing, but if the UnboundLocalError happens, it reduces the images#######################
    max_frame = [10000, 10000, 0, 0]

    try:
        for i, file in enumerate(files):
        #print(i)
            orig = extract_pixels_for_viewing(file)
            orig, frame = trim_background(orig)
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])
    except UnboundLocalError:
        ####TEST 24/7/21
        print("calling NEW altered trim background fun")
        #files = files[:i]
    #################################################################################################################     
    #####note: prior to implementing the conditioning, there was no indent here
        max_frame = [10000, 10000, 0, 0]
        for i, file in enumerate(files):
            orig = extract_pixels_for_viewing(file)
            ####TEST 24/7/21
            orig, frame = trim_background(orig, [0, 0, 512, 512])
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])


#    for i, file in enumerate(files):

#        ext = i
#        orig = extract_pixels_for_viewing(file)
#        aim = Image.fromarray(orig, mode='L')
#        orig, _ = trim_background(orig, max_frame)
#        aim = Image.fromarray(orig, mode='L')
#        aim.save(os.path.join(output_dir, str(name), str(ext)+".png"), format='PNG')
    return max_frame


def report_coordinates_for_dumb_and_the_deaf(path):
    '''
    Convert images from DICOM to png.
    '''
    files = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not files:
        print("no files found at ", os.path.join(path, f))
        return
    patient_position = files[0].PatientPosition
    # Patient scans are either "feet first" (FFS) or "head first" (HFS). If FFS, they must be reversed.
    reverse = patient_position == "FFS"
    files.sort(key=lambda x: float(x.SliceLocation), reverse=reverse)

    num_slides = len(files)
    
    
   ##########################here comes the 20/9/20 conditional statement
    #it tries to do the whole thing, but if the UnboundLocalError happens, it reduces the images#######################
    max_frame = [10000, 10000, 0, 0]

    try:
        for i, file in enumerate(files):
        #print(i)
            orig = extract_pixels_for_viewing(file)
            orig, frame = trim_background(orig)
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])
    except UnboundLocalError:
        return max_frame
        ####TEST 24/7/21
        print("calling NEW altered trim background fun")
        #files = files[:i]
    #################################################################################################################     
    #####note: prior to implementing the conditioning, there was no indent here
        max_frame = [10000, 10000, 0, 0]
        for i, file in enumerate(files):
            orig = extract_pixels_for_viewing(file)
            ####TEST 24/7/21
            orig, frame = trim_background(orig, [0, 0, 512, 512])
            l1, l2, u1, u2 = frame
            max_frame[:2] = min(max_frame[:2], frame[:2])
            max_frame[2:] = max(max_frame[2:], frame[2:])


#    for i, file in enumerate(files):

#        ext = i
#        orig = extract_pixels_for_viewing(file)
#        aim = Image.fromarray(orig, mode='L')
#        orig, _ = trim_background(orig, max_frame)
#        aim = Image.fromarray(orig, mode='L')
#        aim.save(os.path.join(output_dir, str(name), str(ext)+".png"), format='PNG')
    return max_frame



def save_partition(mask, path):
    mask = mask.astype(np.uint8)
    pickle.dump(mask, open( path + ".p", "wb" ))


def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)

    data_new[data <= (wl-ww/2.0)] = out_range[0]
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
         ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    data_new[data > (wl+ww/2.0)] = out_range[1]-1

    return data_new.astype(dtype)

def extract_pixels_for_viewing(dicom):
    pixels = dicom.pixel_array
    hu = pixels * dicom.RescaleSlope + dicom.RescaleIntercept
    return win_scale(hu, 60, 400, np.uint8, [0, 255]) 


def trim_background(img, dims = None):
    # Trim out background and table
    black = np.zeros_like(img)
    black[img < 10] = 1
    black[img >= 10] = 2
    black[:,0 ]= 1
    black[:,-1 ]= 1
    black_label, num_classes = label(black, background=0, connectivity=1, return_num=1)

    markers = np.zeros_like(img)
    markers[black_label != 1] = 1
    m_label, num_classes = label(markers, background=0, connectivity=1, return_num=1)
    m_label = m_label.astype(np.uint8) 
    remove_small_objects(m_label, min_size=50000, in_place=True)
    # remove everything that is not the body
    mask = np.where(m_label == 0)
    img2 = img.copy()
    img2[mask] = 0
    
    # Make a bounding box around the body
    if not dims:
        bbox = False
        for i, region in enumerate(regionprops(m_label)):
            minr, minc, maxr, maxc = region.bbox
            bbox = True
        if not bbox:
            print("nothing :/")
    else:
        minr, minc, maxr, maxc = dims

    return(img2[minr:maxr, minc:maxc], [minr, minc, maxr, maxc])


def partition_at_threshold(img, thresh, square_size, min_size, title=None, show_plot=True):
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, sharex=True)
        if title:
            fig.suptitle(title)

    # Now start looking for the border of the image. Filter out anything below threshold
    bw = ndi.gaussian_filter(img, sigma=(1), order=0) > thresh
    if show_plot:
        axes[0].imshow(bw, cmap=plt.cm.gray)
        axes[0].set_title('thresholded on intensity')
        axes[0].axis('off')

    # Smooth what remains
    remove_small_holes(bw, area_threshold=40, in_place=True)
    remove_small_objects(bw, min_size=min_size, in_place=True)
    cleared = closing(bw, square(square_size))
    distance = ndi.distance_transform_edt(np.logical_not(cleared))
    mask = np.zeros_like(distance)
    mask[distance <= 2] = 1
    distance = ndi.distance_transform_edt(mask)
    cleared = np.zeros_like(distance)
    cleared[distance > 2] = 1

    if show_plot:
        axes[1].imshow(cleared, cmap=plt.cm.gray)
        axes[1].set_title('cleaned up by removing small holes/objects')
        axes[1].axis('off')

    return cleared


def compute_preliminary_masks(patient_id, input_dir=None, output_dir=None):
    bones_thresh = [200, 2, 64]
    blood_vessels_thresh = [165, 5, 64]
    liver_thresh = [130, 1, 64]

    if not input_dir:
        input_dir = "assets/niftynet_raw_images"
    if not output_dir:
        output_dir = "assets/masks"

    file_dir = os.path.join(input_dir, str(patient_id))
    if os.path.exists(os.path.join(output_dir, str(patient_id))):
        print("This patient's masks seem to be already existing. Skipping!")
        return
    os.makedirs(os.path.join(output_dir, str(patient_id)), exist_ok = True)
    slices = [int(x.split(".")[0]) for x in os.listdir(file_dir)]

    for slice in slices:
        img_path = os.path.join(input_dir, "%d/%d.png" % (patient_id, slice))
        img = skio.imread(img_path)
        mask = partition_at_threshold(img, *bones_thresh, title="Bones", show_plot=False)
        imgb = img.copy() * (1 - mask)
        mask = partition_at_threshold(imgb, *blood_vessels_thresh, title="Blood vessels", show_plot=False)
        imgb = imgb * (1 - mask)
        liver = partition_at_threshold(imgb, *liver_thresh, title = "Organs/Liver", show_plot=False)
        edge_sobel = feature.canny(img, sigma=3)
        edge_sobel[liver == 0] = 0
        distance = ndi.distance_transform_edt(np.logical_not(edge_sobel))
        liver[distance <= 1] = 0
        mask = liver > 0
        remove_small_objects(mask, in_place=True)
        liver[mask==0] = 0
        newliver = label(liver)
        newliver[newliver>0] = newliver[newliver>0] + 2
        save_partition(newliver, os.path.join(output_dir, "%d/%d" % (patient_id, slice)))

    return "success" ####edit martin: removed indent here


