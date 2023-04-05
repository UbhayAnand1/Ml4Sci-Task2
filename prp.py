from astropy.io import fits
from PIL import Image
import os
from torchvision import transforms
import numpy as np
from skimage import exposure

# Set the path to the directory containing the FITS files
fits_dir = r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\files'

# Set the path to the directory where you want to save the JPEG files
jpeg_dir = r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\jpeg_files'

# Create the JPEG directory if it doesn't exist
if not os.path.exists(jpeg_dir):
    os.makedirs(jpeg_dir)

# Define the transforms you want to apply to the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Loop through all the FITS files in the directory
for filename in os.listdir(fits_dir):
    if filename.endswith('.fits'):
        # Open the FITS file
        with fits.open(os.path.join(fits_dir, filename)) as hdul:
            # Get the image data from the first HDU
            image_data = hdul[0].data

            # Normalize the image data using rescale_intensity
            p2, p98 = np.percentile(image_data, (2, 98))
            image_data = exposure.rescale_intensity(image_data, in_range=(p2, p98))

            # Apply the transforms to the image data
            image_data = transform(image_data)

            # Reshape the image data to have 3 dimensions
            image_data = np.squeeze(image_data.numpy())

            # Clip the pixel values to be between 0 and 1
            image_data = np.clip(image_data, 0, 0.6)
            
            # Convert the image data to a NumPy array of type uint8
            image_data = (image_data * 255).astype(np.uint8)

            # Convert the image data to a PIL Image
            image = Image.fromarray(image_data)

            # Convert the image to mode RGB
            image = image.convert('RGB')

            # Save the image as a JPEG
            image.save(os.path.join(jpeg_dir, filename.replace('.fits', '.jpg')))