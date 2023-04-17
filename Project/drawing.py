import numpy as np
import pyautogui
import pygame
from PIL import Image
from skimage import feature


# Procedure involved for this method:
# 1. Load image and convert it to grayscale, downsample it if it exceeds the size of the computer screen
# 2. Initialize pygame board to draw lines and set background color to white
# 3. Get the brightness of the image
def start():
    # Load image, change the image here to see different results
    image = Image.open("car.png")
    # Resize the image if it is too large for the computer screen to display
    my_screen_width, my_screen_height = pyautogui.size()[0], pyautogui.size()[1]
    if image.width > my_screen_width or image.height > my_screen_height:
        original_image_ratio = image.width / image.height
        image = image.resize((int(my_screen_width * 0.7), int(my_screen_width * 0.7 / original_image_ratio)))
    # Convert the image to gray for edge detection
    gray_image = image.convert('L')
    # Get image size
    image_size = gray_image.width * gray_image.height
    # Get bright sum for the entire image
    bright_sum = 0
    for x in range(gray_image.width - 1):
        for y in range(gray_image.height - 1):
            bright_sum += gray_image.getpixel((x, y))
    # Compute average brightness
    average_brightness = bright_sum / image_size

    # Initialize pygame for drawing lines
    pygame.init()
    # Create a screen
    screen = pygame.display.set_mode((gray_image.width, gray_image.height))
    screen.fill((255, 255, 255))
    # Draw all lines
    draw(gray_image, average_brightness, image_size, screen)
    # Update pygame screen to reflect all lines drawing
    pygame.display.update()
    # Run the game loop
    running = True
    while running:
        # Quit program only if user closes it
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Quit pygame
    pygame.quit()


# Procedure involved for this method:
# 1. Use existing canny edge detector to find the edge of the image
# 2. Loop through each pixel to draw lines
def draw(gray_image, average_brightness, image_size, screen):
    # Get edges using canny edge detector
    canny_edge = feature.canny(np.array(gray_image), sigma=0.1)
    # Flip the result of canny edge horizontally
    canny_edge = np.transpose(canny_edge)[::-1]
    # Flip the result of canny edge vertically
    canny_edge = canny_edge[::-1]
    canny_edge = (~canny_edge * 255).astype(np.uint8)
    # Compute max brightness
    max_bright = 255
    # Adjusts how much spacing should be implemented
    spacing_multiplier = 1

    # Loop through every pixel in the image
    for y in range(gray_image.height - 1):
        count_x = 0
        for x in range(gray_image.width - 1):
            # Index starts from the top left pixel and increments all the way to the bottom right corner
            index = x + y * gray_image.width
            # Use this builtin method to extract the row index and column index
            row, col = divmod(index, gray_image.width)
            pixel_value = gray_image.getpixel((col, row))
            # fix when the whole input image is bright
            if average_brightness > 200 and pixel_value < 225:
                pixel_value -= (average_brightness - 100)
            # Reset pixel value to 0 if it is negative
            if pixel_value < 0:
                pixel_value = 0
            # Horizontal density
            spacing_x = spacing_x_direction(pixel_value)
            spacing_x *= spacing_multiplier
            # Vertical density
            spacing_y, previous_spacing_y = spacing_y_direction(pixel_value)
            spacing_y *= spacing_multiplier
            # Make sure the lines drawing are smoothing enough
            correction_y = 30 % previous_spacing_y

            # Drawing horizontal lines
            brightness_sum_x = 0
            current_x = 0
            # Line starts from this pixel should not exist already
            if y % spacing_x == 0 and pixel_value < 240 and x >= count_x:
                row, col = divmod(y + x + current_x, gray_image.width)
                while brightness_sum_x <= max_bright and current_x < gray_image.width and canny_edge[col, row] != 0:
                    row, col = divmod(index + current_x, gray_image.width)
                    # Drawing should stop when it reaches the edge
                    if canny_edge[col, row] != 255:
                        break
                    brightness_sum_x += gray_image.getpixel((col, row))
                    current_x += 1
                # Draw the line
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x + current_x, y))
                count_x = x + current_x

            # Drawing vertical lines
            brightness_sum_y = 0
            current_y = 0
            # Line starts from this pixel should not exist already
            if (x - correction_y) % spacing_y == 0 and pixel_value < 210 and y >= -1:
                while brightness_sum_y <= max_bright and index + (current_y * gray_image.width) < gray_image.height:
                    row, col = divmod(index + (current_y * gray_image.width) + current_x, gray_image.width)
                    # Drawing should stop when it reaches the edge
                    if canny_edge[col, row] != 255:
                        break
                    brightness_sum_y += gray_image.getpixel((col, row))
                    current_y += 1
                # Draw the line
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x, y + current_y))

            # Drawing left diagonal line
            length_x = 0
            length_y = 0
            # Line starts from this pixel should not exist already
            if (x - correction_y) % spacing_y == 0 and y % spacing_x == 0 and pixel_value < 150:
                while (length_x < spacing_y or length_y < spacing_x) and (
                        index + length_y * gray_image.width - length_x) < image_size:
                    row, col = divmod(index + (length_y * gray_image.width - length_x), gray_image.width)
                    # Drawing should stop when it reaches the edge
                    if canny_edge[col, row] != 255:
                        break
                    # Increment length on both directions based on spacing, as this is a diagonal line
                    if length_x < spacing_y:
                        length_x += 1
                    if length_y < spacing_x:
                        length_y += 1
                # Draw the line
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x - length_x, y + length_y))

            # Drawing right diagonal line
            length_x = 0
            length_y = 0
            # Line starts from this pixel should not exist already
            if (x - correction_y) % spacing_y == 0 and y % spacing_x == 0 and pixel_value < 100:
                while (length_x < spacing_y or length_y < spacing_x) and (
                        index + length_y * gray_image.width + length_x) < image_size:
                    row, col = divmod(index + (length_y * gray_image.width + length_x), gray_image.width)
                    # Drawing should stop when it reaches the edge
                    if canny_edge[col, row] != 255:
                        break
                    # Increment length on both directions based on spacing, as this is a diagonal line
                    if length_x < spacing_y:
                        length_x += 1
                    if length_y < spacing_x:
                        length_y += 1
                # Draw the line
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x + length_x, y + length_y))


# Compute the density for horizontal lines, which will be used to drawing horizontal lines
def spacing_x_direction(pixel_value):
    density_x = 10
    # set the density of horizontal lines
    for i in range(0, 255, 40):
        if i <= pixel_value < i + 40:
            density_x = 4 * ((i // 40) + 1)
            break
    return density_x


# Compute the density for vertical lines, which will be used to drawing vertical lines
def spacing_y_direction(pixel_value):
    density_y = 10
    previous_density_y = 10
    # set the density of vertical lines
    for i in range(0, 255, 30):
        if i <= pixel_value < i + 30:
            previous_density_y = density_y
            density_y = 2 * ((i // 30) + 1)
            break
    return density_y, previous_density_y


if __name__ == '__main__':
    start()
