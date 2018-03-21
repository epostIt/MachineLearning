print("Running")
from PIL import Image


from PIL import Image, ImageFilter
try:
    original = Image.open("maggie.png")
    contour = original.filter(ImageFilter.CONTOUR)
    emboss = original.filter(ImageFilter.EMBOSS)
    findedges = original.filter(ImageFilter.FIND_EDGES)

except:
    print("Unable to load image")

print("The size of the Image is: ")
print(original.format, original.size, original.mode)
original.show()
contour.show()
emboss.show()
findedges.show()
