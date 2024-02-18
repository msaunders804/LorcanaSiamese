import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import os

def DistortColor(color_img):
    converter = ImageEnhance.Color(color_img)
    saturation = np.random.choice([.5,1.5])
    return converter.enhance(saturation)

def DistortBlur(blur_img):
    return blur_img.filter(filter=ImageFilter.BLUR)

def DistortContrast(contrast_img):
    converter = ImageEnhance.Contrast(contrast_img)
    saturation = np.random.choice([.5,1.5])
    return converter.enhance(saturation)

def DistortFlip(flip_img):
    initial_size = flip_img.size
    # determine at random how much or little we scale the image
    scale = 0.95 + np.random.random() * .1
    scaled_img_size = tuple([int(i * scale) for i in initial_size])
    # create a blank background with a random color and same size as intial image
    bg_color = tuple(np.random.choice(range(256), size=3))
    background = Image.new('RGB', initial_size, bg_color)
    # determine the center location to place our rotated card
    center_box = tuple((n - o) // 2 for n, o in zip(initial_size, scaled_img_size))
    # scale the image
    scaled_img = flip_img.resize(scaled_img_size)
    # randomly select an angle to skew the image
    max_angle = 5
    skew_angle = np.random.randint(-max_angle, max_angle)
    # add the scaled image to our color background
    background.paste(scaled_img.rotate(skew_angle, fillcolor=bg_color, expand=1).resize(scaled_img_size),
                     center_box)
    # potentially flip the image 180 degrees
    if np.random.choice([True, False]):
        background = background.rotate(180)

    return background

def DistortSharpness(sharpness_img):
    converter = ImageEnhance.Contrast(sharpness_img)
    contrast = np.random.choice([.5, 1.5])
    return converter.enhance(contrast)
def RandomDistortion(path, i):
    if i == 0 or i == 5:
        dis_image = DistortColor(path)
    elif i == 1 or i == 6:
        dis_image = DistortBlur(path)
    elif i == 2 or i == 7:
        dis_image = DistortContrast(path)
    elif i == 3 or i == 8:
        dis_image = DistortFlip(path)
    elif i == 4 or i == 9:
        dis_image = DistortSharpness(path)
    i += 1
    return dis_image

def GenerateDistorted(path, num_distorts):
    images_created = 0
    images_distorted = 0
    for file_name in os.listdir(path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            images_distorted += 1
            file_path = os.path.join(path, file_name)
            for i in range(num_distorts):
                distorted_path = os.path.join(path, f"{os.path.splitext(file_name)[0]}-{i}.jpg")
                if not os.path.exists(distorted_path):
                    img = Image.open(file_path).convert("RGB")
                    distorted = RandomDistortion(img,i)
                    distorted.save(distorted_path)
                    images_created += 1
                    print(f"Distorted image saved: {distorted_path}")
    print(f"\n{images_created} total unique distortions saved from {images_distorted} different images.")
