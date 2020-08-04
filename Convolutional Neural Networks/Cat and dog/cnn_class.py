import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from keras.preprocessing import image


# dimensions of our images    -----   are these then grayscale (black and white)?
img_width, img_height = 64, 64

# load the model we saved
model = load_model('dog_cat_model.h5')

#Image source
img_src = 'dataset/training_set/cats/cat.100.jpg'

# Get test image ready
img = mpimg.imread(img_src)
test_image_main = image.load_img(img_src, target_size=(img_width, img_height))
test_image = image.img_to_array(test_image_main)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image, batch_size=1)

if result[0][0]==1.0:
  answer = 'Dog'
else:
  answer = 'cat'

plt.imshow(img)
plt.title(answer)
plt.show()
