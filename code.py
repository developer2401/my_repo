import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K
img_width, img_height = 224, 224
nb_train_samples = 8144
nb_validation_samples = 8041
epochs = 10
batch_size = 32
n_classes = 10


for dirname, _, filenames in os.walk(r'D:\tomato dataset'):   #train dataset address
    for filename in filenames:
        print(os.path.join(dirname, filename))

tomato_bacterial_spot_dir=r'D:\tomato dataset\Bacterial_spot227'  #address of all the folders inside the dataset
tomato_early_blight_dir=r'D:\tomato dataset\Early_blight227'
tomato_late_blight_dir=r'D:\tomato dataset\Late_blight227'
tomato_leaf_mold_dir=r'D:\tomato dataset\Leaf_Mold227'
tomato_septoria_leaf_spot_dir=r'D:\tomato dataset\Septoria_leaf_spot227'
tomato_spider_mite_dir=r'D:\tomato dataset\Two-spotted_spider_mite227'
tomato_target_spot_dir=r'D:\tomato dataset\Target_Spot227'
tomato_yellow_leaf_curl_dir=r'D:\tomato dataset\Tomato_Yellow_Leaf_Curl_Virus227'
tomato_mosaic_virus_dir=r'D:\tomato dataset\Tomato_mosaic_virus227'
tomato_healthy_dir=r'D:\tomato dataset\healthy227'

nrows=10
ncols=10
pic_index=0
fig=plt.gcf()
fig.set_size_inches(10,10)

tomato_bacterial_spot_names=os.listdir(tomato_bacterial_spot_dir)
tomato_early_blight_names=os.listdir(tomato_early_blight_dir)
tomato_late_blight_names=os.listdir(tomato_late_blight_dir)
tomato_leaf_mold_names=os.listdir(tomato_leaf_mold_dir)
tomato_septoria_leaf_spot_names=os.listdir(tomato_septoria_leaf_spot_dir)
tomato_spider_mite_names=os.listdir(tomato_spider_mite_dir)
tomato_target_spot_names=os.listdir(tomato_target_spot_dir)
tomato_yellow_leaf_curl_names=os.listdir(tomato_yellow_leaf_curl_dir)
tomato_mosaic_virus_names=os.listdir(tomato_mosaic_virus_dir)
tomato_healthy_names=os.listdir(tomato_healthy_dir)

pic_index += 10
next_tomato_bacterial_spot_pix = [os.path.join(tomato_bacterial_spot_dir, fname)
                for fname in tomato_bacterial_spot_names[pic_index-10:pic_index]]
next_tomato_early_blight_pix = [os.path.join(tomato_early_blight_dir, fname)
                for fname in tomato_early_blight_names[pic_index-10:pic_index]]
next_tomato_late_blight_pix = [os.path.join(tomato_late_blight_dir, fname)
                for fname in tomato_late_blight_names[pic_index-10:pic_index]]
next_tomato_leaf_mold_pix = [os.path.join(tomato_leaf_mold_dir, fname)
                for fname in tomato_leaf_mold_names[pic_index-10:pic_index]]
next_tomato_septoria_leaf_spot_pix = [os.path.join(tomato_septoria_leaf_spot_dir, fname)
                for fname in tomato_septoria_leaf_spot_names[pic_index-10:pic_index]]
next_tomato_spider_mite_pix = [os.path.join(tomato_spider_mite_dir, fname)
                for fname in tomato_spider_mite_names[pic_index-10:pic_index]]
next_tomato_target_spot_pix = [os.path.join(tomato_target_spot_dir, fname)
                for fname in tomato_target_spot_names[pic_index-10:pic_index]]
next_tomato_yellow_leaf_curl_pix = [os.path.join(tomato_yellow_leaf_curl_dir, fname)
                for fname in tomato_yellow_leaf_curl_names[pic_index-10:pic_index]]
next_tomato_mosaic_virus_pix = [os.path.join(tomato_mosaic_virus_dir, fname)
                for fname in tomato_mosaic_virus_names[pic_index-10:pic_index]]
next_tomato_healthy_pix = [os.path.join(tomato_healthy_dir, fname)
                for fname in tomato_healthy_names[pic_index-10:pic_index]]

for i, img_path in enumerate(
        next_tomato_bacterial_spot_pix + next_tomato_early_blight_pix + next_tomato_late_blight_pix + next_tomato_leaf_mold_pix + next_tomato_septoria_leaf_spot_pix + next_tomato_spider_mite_pix + next_tomato_target_spot_pix + next_tomato_yellow_leaf_curl_pix + next_tomato_mosaic_virus_pix + next_tomato_healthy_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

labels=['bacterial_spot','early_blight','late_blight','leaf_mold','septoria_leaf_spot','spider_mite','target_spot','yellow_leaf_curl','mosaic_virus','healthy']

validation_data_dir = 'tomato test set'

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')


model =tf.keras.models.load_model('plant_diseases_model.h5')
model.load_weights(r'C:\Users\com\PycharmProjects\PlantDisease\training\cp-299.ckpt.data-00000-of-00001')

pred = model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
predicted = np.argmax(pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, np.argmax(pred, axis=1))
plt.figure(figsize = (30,20))
sns.set(font_scale=1.4) #for label size
sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()
print()
print('Classification Report')
print(classification_report(validation_generator.classes, predicted, target_names=labels))

def predict_one(model):
    image_batch, classes_batch = next(validation_generator)
    predicted_batch = model.predict(image_batch)
    for k in range(0,image_batch.shape[0]):
      image = image_batch[k]
      pred = predicted_batch[k]
      the_pred = np.argmax(pred)
      predicted = labels[the_pred]
      val_pred = max(pred)
      the_class = np.argmax(classes_batch[k])
      value = labels[np.argmax(classes_batch[k])]
      plt.figure(k)
      isTrue = (the_pred == the_class)
      plt.title(str(isTrue) + ' - class: ' + value + ' - ' + 'predicted: ' + predicted + '[' + str(val_pred) + ']')
      plt.imshow(image)

predict_one(model)