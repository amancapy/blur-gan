import os

import numpy
from PIL import Image, ImageDraw, ImageFilter

path = "test"
if not os.path.exists(f"animals_raw/raw-img/{path}"):
    os.mkdir(f"animals_raw/raw-img/{path}")

i = 0
for name in os.listdir(f"animals_raw/raw-img/{path}_pre"):
    if i % 100 == 0:
        print(i)
    i += 1

    image = Image.open(f"animals_raw/raw-img/{path}_pre/{name}")
    blimage = image.copy()
    if image.width < 256:
        image = image.resize((256, int(256 * image.height/image.width)))
        blimage = image.resize((256, int(256 * image.height/image.width)))
    elif image.height < 256:
        image = image.resize((int(256 * image.width/image.height), 256))
        blimage = blimage.resize((int(256 * image.width/image.height), 256))
    blimage = blimage.filter(ImageFilter.GaussianBlur(image.width / 200))

    w = 256
    left, top = int((image.width - w) / 2), int((image.height - w) / 2)
    right, bottom = left + w, top + w

    image = image.crop((left, top, right, bottom))
    blimage = blimage.crop((left, top, right, bottom))

    i1size, i2size = image.size, blimage.size

    jimage = Image.new('RGB', (i1size[0] * 2, i2size[1]), (0, 0, 0))
    jimage.paste(image, (0, 0))
    jimage.paste(blimage, (i1size[0], 0))

    jimage.save(f"animals_raw/raw-img/{path}/{name}")
    image.close()