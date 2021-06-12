from PIL import Image
import pydicom
import os
import tqdm
import threading


def get_im_list(image_dir):
    im_files = []
    for root, _, files in os.walk(image_dir, topdown=True):
        for name in files:
            im_files.append(os.path.join(root, name))
    return im_files


def save_dicom(path, name_w_ext, dest):
    if os.path.exists(os.path.join(dest+"/")+name_w_ext):
        return
    try:
        os.makedirs(dest)
    except:
        pass
    im_dcm = pydicom.dcmread(path)
    im = Image.fromarray(im_dcm.pixel_array)
    im.save(os.path.join(dest+"/", name_w_ext))


def Worker(data):
    for im in tqdm.tqdm(data, ncols=120):
        name_dicom = '_'.join(im.split('/')[1:])
        name_png = name_dicom.split('.')[0] + '.png'
        save_dicom(im, name_png, f"png_{dir}")


if __name__ == '__main__':
    dirs = ['test/', 'train/']
    BATCH_SIZE = 800
    threads = []

    for dir in dirs:
        im_list = get_im_list(dir)
        for i in range(0, len(im_list), BATCH_SIZE):
            upper = i + BATCH_SIZE if i + \
                BATCH_SIZE < len(im_list) else len(im_list)
            lower = i
            t = threading.Thread(target=Worker, args=[
                im_list[lower:upper]]
            )
            threads.append(t)
            t.start()

# WARNINGS !!!
# UserWarning: The length of the pixel data in the dataset(13262360 bytes) indicates 
# it contains excess padding. 216296 bytes will be removed from the end of the data
