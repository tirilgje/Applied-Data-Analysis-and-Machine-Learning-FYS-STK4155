import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

'''
Code used to to downscale an image to the report 
'''

path = "..\Images\Apple\Apple D\Apple De06081.png"
# path = "..\Images\Pear\Pear 48.png"

image = Image.open(path)
# plt.imshow(image)
# plt.show()
dpi = 75
newsize = (75, 75)
im1 = image.resize(newsize)

fig = plt.figure(frameon=False)
fig.set_size_inches(7.5,7.5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(im1, aspect='auto')
fig.savefig('../Figs/Apple_Downsized.png')
plt.show()



path = "..\Images\Pear\Pear 48.png"

image = Image.open(path)
# plt.imshow(image)
# plt.show()

newsize = (75, 75)
im1 = image.resize(newsize)

fig = plt.figure(frameon=False)
fig.set_size_inches(7.5,7.5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(im1, aspect='auto')
fig.savefig('../Figs/Pear_Downsized.png')
# plt.imshow(im1)
plt.show()

# im1 = im1.save('../Figs/Apple_Downsized.png')
