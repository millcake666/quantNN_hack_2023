import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image, ImageStat, ImageFilter


def binari_image(path_to_file: str) -> Image:
    # Open the image
    image = Image.open(path_to_file)
    
    # Smothing image
    moreSmoothenedImage = image.filter(ImageFilter.SMOOTH_MORE)
    
    # Convert the image to grayscale
    image = image.convert("L")
    
    # Get median value of image
    median_pixel = ImageStat.Stat(image).median
    
    # max_pixel = ImageStat.Stat(image).max
    # Set threshhold like 2 * median
    threshold = median_pixel[0]
    
    # Thansform image to FFFFFF and 000000
    return image.point(lambda x: 255 if x >= 1.8 * threshold else 0)
    
    
    
def split_qubits(image: Image) -> tuple[Image]:
    # Get image statistix
    w, h = image.size
    
    # Spliting qubits ((left, top, right, bottom))
    first_qubit = image.crop((0, 0 * h / 4, w, 1 * h / 4))
    second_qubit = image.crop((0, 1 * h / 4, w, 2 * h / 4))
    third_qubit = image.crop((0, 2 * h / 4, w, 3 * h / 4))
    four_qubit = image.crop((0, 3 * h / 4, w, 4 * h / 4))
    
    return first_qubit, second_qubit, third_qubit, four_qubit


def get_qubit_heat_map(path: str):

    # Get current path
    PATH = os.getcwd()
    
    # Fill list of list with zero values
    heat_map = [[0] * 56 for _ in range(164)]
    
    # For each picture
    for file in tqdm(os.listdir(path)):
        
        # We bring the picture to the black and white format
        image = binari_image(os.path.join(path, file))  
        
        # Get picture size  
        width, height = image.size
        
        # Add color of each pixel to heat_map
        for h in range(height):
            for w in range(width):
                heat_map[h][w] += image.getpixel((w, h)) 
        
    # Devide each pixel of heat_map to images_cout
    for h in range(height):
        for w in range(width):
            heat_map[h][w] /= len(os.listdir(path))
    
    # Convert list of lists to numpy.nd.array
    heat_map = np.asarray(heat_map, dtype=np.uint8)
    pic = Image.fromarray(heat_map)
    
    # Save result heat_map
    pic.save("heat_map.png")



def conv_image(image: Image, step: int) -> Image:
    # Getting image width and height
    width, height = image.size  
    
    # Creating / calculating result list
    result = [[0] * int(math.ceil(width / float(step))) for _ in range(int(math.ceil(height / float(step))))] 

    # For each step
    for h in range(0, height - (step - 1), step): 
        for w in range(0, width - (step - 1), step): 
            # Summarize values in rectangle step:step
            pixel_sum = 0 
            for i in range(step): 
                for j in range(step): 
                    pixel_sum += image.getpixel((w + j, h + i)) 

            # Write summarized value to resulting list
            result[int(math.ceil(h / float(step)))][int(math.ceil(w / float(step)))] = 255 if pixel_sum >= 255 else pixel_sum 
    
    # Convert list of lists to numpy.nd.array
    result = np.asarray(result, dtype=np.uint8)
    return Image.fromarray(result)


def conv_image_to_one_pixel(image: Image) -> Image:
    # image.save("image.png")
    
    # Convolution image
    image_1 = conv_image(image=image, step=2)
    # fimage_1.save("first_qubit_1.png")
    
    # One more convolution image
    image_2 = conv_image(image=image_1, step=2)
    # image_2.save("first_qubit_2.png") 
    
    # One more more convolution image
    image_3 = conv_image(image=image_2, step=2)
    # image_3.save("first_qubit_3.png")
        
    # One more more more convolution image
    image_4 = conv_image(image=image_3, step=2)
    # image_4.save("first_qubit_4.png")
    
    # One more more more more convolution image
    image_5 = conv_image(image=image_4, step=2)
    # image_5.save("first_qubit_5.png")
    
    # One more more more more more convolution image
    image_6 = conv_image(image=image_5, step=2)
    # fimage_6.save("first_qubit_6.png")
    
    return image_6



PATH = os.getcwd()    
columns = ["file number", "file name", "qubit 1 state", "qubit 2 state", "qubit 3 state", "qubit 4 state"]

index = 0
data = []
for file in tqdm(os.listdir(os.path.join(PATH, "Входные данные"))):
    qubits = split_qubits(binari_image(os.path.join(PATH, "Входные данные", file)))
    data.append([index,
                 file, 
                 1 if conv_image_to_one_pixel(qubits[0]).getpixel((0, 0)) > 0 else 0,
                 1 if conv_image_to_one_pixel(qubits[1]).getpixel((0, 0)) > 0 else 0,
                 1 if conv_image_to_one_pixel(qubits[2]).getpixel((0, 0)) > 0 else 0,
                 1 if conv_image_to_one_pixel(qubits[3]).getpixel((0, 0)) > 0 else 0,])
    index += 1

    

df = pd.DataFrame(data, columns = columns) 
df.to_csv("labeled_ions_team_number.csv", sep=';', encoding='utf-8', index=False)
# img.save("some.png", quality=95)


