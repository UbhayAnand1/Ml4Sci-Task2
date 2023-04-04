import os
from astropy.io import fits
from PIL import Image

csv_file = "C:/Users/abhay/OneDrive/Desktop/task2/SpaceBasedTraining/classifications.csv"
fits_folder = "C:/Users/abhay/OneDrive/Desktop/task2/SpaceBasedTraining/files"
output_folder = "C:/Users/abhay/OneDrive/Desktop/task2/SpaceBasedTraining/output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(csv_file, 'r') as f:
    for line in f:
        values = line.strip().split(',')
        if values[0] != "ID":
            filename = values[0] + ".fits"
            fits_path = os.path.join(fits_folder, filename)
            jpg_path = os.path.join(output_folder, values[0] + ".jpg")
            hdul = fits.open(fits_path)
            data = hdul[0].data
            image = Image.fromarray(data)
            image = image.convert('L')
            image.save(jpg_path)
            hdul.close()
