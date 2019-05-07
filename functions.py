import numpy as np
from PIL import Image

# Function used to map a 1xWxH class tensor to a 3xWxH color image
def color_lanes(image, classes, i, color, HEIGHT, WIDTH):
    buffer_c1 = np.zeros((HEIGHT, WIDTH))
    buffer_c1[classes == i] = color[0]   
    image[:, :, 0] += buffer_c1

    buffer_c2 = np.zeros((HEIGHT, WIDTH))
    buffer_c2[classes == i] = color[1]   
    image[:, :, 1] += buffer_c2

    buffer_c3 = np.zeros((HEIGHT, WIDTH))
    buffer_c3[classes == i] = color[2]   
    image[:, :, 2] += buffer_c3
    return image

def blend(image_orig, image_classes): 
	image_classes = image_classes.astype(np.uint8)
	mask = np.zeros(image_classes.shape)

	mask[image_classes.nonzero()] = 255
	mask = mask[:, :, 0]
	mask = Image.fromarray(mask.astype(np.uint8))
	image_classes = Image.fromarray(image_classes)

	image_orig.paste(image_classes, None, mask)
	return image_orig


