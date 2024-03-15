import cv2
from PIL import Image

# Read the image (replace 'path/to/img' with your image file path)  # Read as grayscale (optional)
img = Image.open('C:\\Users\\msaun\\Documents\\Cards\\test\\1-1.jpg')
img = img.convert('L')
desired_width = 224
aspect_ratio = img.width / img.height
desired_height = int(desired_width / aspect_ratio)
resized_image = img.resize((desired_width, desired_height), Image.ANTIALIAS)
# Get the dimensions (height and width)
width, height = resized_image.size

print(f"Image width: {width} pixels")
print(f"Image height: {height} pixels")