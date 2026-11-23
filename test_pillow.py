from PIL import Image
import PIL

print("Pillow version:", PIL.__version__)

image = Image.new("RGB", (200, 200), "white")

print("Image created successfully!")
print("Image size:", image.size)