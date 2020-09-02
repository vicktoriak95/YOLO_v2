from PIL import ImageDraw, ImageFont
import colorsys
import random
import numpy as np

RED = (255, 0, 0)
GREEN = (34, 139, 34)


def generate_colors(num_of_colors):
    hsv_tuples = [(x / num_of_colors, 1., 1.) for x in range(num_of_colors)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def draw_object_on_image(image, box, score, predicted_class, color):
    draw = ImageDraw.Draw(image)

    label = '{} {:.2f}'.format(predicted_class, score)

    # Calculate drawing paramaters
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    # thickness = (image.size[0] + image.size[1]) // 300
    thickness = 5
    label_size = draw.textsize(label, font)

    # Calculate the box's domain
    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(label, (left, top), (right, bottom))

    # Calculate the text's location
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # Draw a detection (box & class) on the image
    # Draw thick box
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)
    # Draw text box
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
    # Draw text
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
