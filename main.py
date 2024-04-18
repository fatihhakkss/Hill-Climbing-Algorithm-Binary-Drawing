# Required libraries imported
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import copy
import random

# Function to read an image and convert it into a binary matrix
def read_image(image_path, size=(224,224)):
    # Open and resize the image
    image = Image.open(image_path)
    image = image.resize(size, Image.LANCZOS)
    # Convert the image to black and white
    image_tf = image.convert('1')
    img = np.array(image_tf)

    # Convert pixel values to binary matrix
    original_matrix = []
    for row in img:
        new_row = []
        for pixel in row:
            if pixel:
                new_row.append(0)
            else:
                new_row.append(1)
        original_matrix.append(new_row)
    original_matrix = np.array(original_matrix)
    return original_matrix

# Function to pad the matrix for circle drawing
def pad_for_circle(matrix):
    diagonal = np.sqrt(2 * (len(matrix) ** 2))
    new_side = diagonal
    padding = int(np.ceil((new_side - len(matrix)) / 2))
    padded_matrix = np.pad(matrix, (padding,), 'constant', constant_values=0)
    final_size = max(padded_matrix.shape)
    final_matrix = np.pad(padded_matrix,
                          ((0, final_size - padded_matrix.shape[0]), (0, final_size - padded_matrix.shape[1])),
                          'constant', constant_values=0)
    return final_matrix

# Function to draw a line using Bresenham algorithm
def bresenham_line_draw(matris, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        # Plotting the line on the matrix
        if matris[y0][x0] == 0:
            matris[y0][x0] = 2
        else:
            matris[y0][x0] = 3
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    # Plotting the line (including the end point) on the matrix
    matris[y0][x0] = 2

# Function to control the line drawing and calculate a score
def bresenham_line_control(matris, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    score = 0

    while x0 != x1 or y0 != y1:
        if matris[y0][x0] == 1:
            score += 2
        elif matris[y0][x0] == 2:
            score -= 1
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return score

# Function to find the best line based on a score
def find_best_line(matris, points, x0, y0, x1, y1):
    score_prev = 0
    for i in range(360):
        x2, y2 = points[i]
        if x2 != x0 or y2 != y0:
            score_next = bresenham_line_control(matris, x1, y1, x2, y2)
            if score_next > score_prev:
                xs, ys = x2, y2
                score_prev = score_next
    return xs, ys

# Path of the image to be processed
image_path = 'Figure_1.jpg'

# Reading the image and displaying it
matrix = read_image(image_path)
#plt.imshow(matrix)
#plt.show()

# Padding the matrix for circle drawing and displaying it
final_matrix = pad_for_circle(matrix)
#plt.imshow(final_matrix)
#plt.show()

# Parameters for circle generation
N = len(final_matrix)
r = N / 2
center = (r, r)
points = []

# Generating points on a circle
for theta in np.linspace(0, 2 * np.pi, 360):
    x = int(round(r * np.cos(theta) + center[0] - 1))
    y = int(round(r * np.sin(theta) + center[1] - 1))
    points.append((x, y))

# Deep copy of the final matrix for drawing
matrix_draw = copy.deepcopy(final_matrix)
matrix_draw = np.zeros_like(matrix_draw)
# Parameters for genetic algorithm
population_size = 1000
mutation_rate = 40  #0-100


# Genetic algorithm for line drawing
x0, y0 = points[0]
x1, y1 = points[0]
for i in range(population_size):
    print(i)
    if random.randint(0, 100) < mutation_rate:
        x2, y2 = points[random.randint(0, 359)]
    else:
        x2, y2 = find_best_line(final_matrix, points, x0, y0, x1, y1)
    bresenham_line_draw(matrix_draw, x1, y1, x2, y2)
    x0, y0 = x1, y1
    x1, y1 = x2, y2
    #if i%100 == 0:
        #plt.imshow(matrix_draw)
        #plt.show()


# Displaying the final image with drawn lines
plt.imshow(matrix_draw)
plt.show()