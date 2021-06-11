from PIL import Image
import pydicom

im_dcm = pydicom.dcmread('train/00086460a852/9e8302230c91/65761e66de9f.dcm')
im = Image.fromarray(im_dcm.pixel_array*10)
im.save("img.png")
