import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import threading
from tqdm import tqdm
from skimage.data import imread
from collections import defaultdict
import os, sys, io, math, bson, struct

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator, ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

DATA_PATH = '/data/vision/fisher/data1/kaggle/cdiscount/'

def step_decay(epoch):
    lr_init = 0.001
    drop = 0.5
    epochs_drop = 2 
    lr_new = lr_init * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr_new

class LR_hist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON(item_data).decode()
            #item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df

def make_val_set(df, split_percentage=0.2, drop_percentage=0.):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()
                
    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)   
    return train_df, val_df

class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(224, 224),  #(180, 180)
                 with_labels=True, batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            #item = bson.BSON.decode(item_data)
            item = bson.BSON(item_data).decode()
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # Preprocess the image.  #check
            x = img_to_array(img)
            self.image_data_generator.fit(np.expand_dims(x, axis=0))
            x = self.image_data_generator.random_transform(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x) 
            #x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        #return self._get_batches_of_transformed_samples(index_array)
        return self._get_batches_of_transformed_samples(index_array[0])

#load example data
categories_df = pd.read_csv('./data/category_names.csv', index_col = 'category_id')
data_example = bson.decode_file_iter(open('./data/train_example.bson', 'rb'))

#data exploration
prod_to_category = dict()
rows, cols = 5, 5 
fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
ax = ax.ravel()

i=0
for c, d in enumerate(data_example):
    product_id = d['_id']
    category_id = d['category_id']
    prod_to_category[product_id] = category_id
    
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        if (i < rows * cols):
            ax[i].imshow(picture)
            ax[i].set_title(categories_df.loc[category_id, 'category_level3'][:12] + ' (' + str(e) + ')')
        i = i + 1

plt.tight_layout() 
plt.savefig('./figures/sample_products.png') 

print "unique level-1 categories: ", len(categories_df['category_level1'].unique())
print "unique level-2 categories: ", len(categories_df['category_level2'].unique())
print "unique level-3 categories: ", len(categories_df['category_level3'].unique())

plt.figure(figsize=(12,12))
sns.countplot(y=categories_df['category_level1'])
plt.tight_layout()
plt.savefig('./figures/level1_categories.png')

#TODO: plot number of images per level-3 accuracies
#https://www.kaggle.com/mihaskalic/keras-xception-model-0-68-on-pl-weights
#https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson

#create training and validation datasets 
num_train_products = 7069896
train_bson_path = DATA_PATH + '/data/train.bson'

num_test_products = 1768182
test_bson_path = DATA_PATH + '/data/test.bson'

"""
categories_df['category_idx'] = pd.Series(range(len(categories_df)), index=categories_df.index)
categories_df.to_csv('./data/categories.csv')
cat2idx, idx2cat = make_category_tables()

train_offset_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)
train_offset_df.to_csv("./data/train_offsets.csv")

train_images_df, val_images_df = make_val_set(train_offset_df, split_percentage=0.2, drop_percentage=0.0) 
train_images_df.to_csv("./data/train_images.csv")
val_images_df.to_csv("./data/val_images.csv")

print "number of training images: ", len(train_images_df)
print "number of validation images: ", len(val_images_df)
"""

train_bson_file = open(train_bson_path, 'rb')
lock = threading.Lock()

#load saved dataframes
print "loading saved dataframes..."
categories_df = pd.read_csv("./data/categories.csv", index_col=0)
cat2idx, idx2cat = make_category_tables()
train_offset_df = pd.read_csv("./data/train_offsets.csv", index_col=0)
train_images_df = pd.read_csv("./data/train_images.csv", index_col=0)
val_images_df = pd.read_csv("./data/val_images.csv", index_col=0)

#training parameters
num_classes = 5270
num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 64 
num_epochs = 5 #8 
steps_per_epoch = num_train_images / batch_size 
validation_steps = num_val_images / batch_size 

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_gen = BSONIterator(train_bson_file, train_images_df, train_offset_df, 
                        num_classes, train_datagen, lock, batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offset_df,
                       num_classes, val_datagen, lock, batch_size=batch_size, shuffle=True)

#check generator images
#bx, by = next(train_gen)
#plt.figure()
#plt.imshow(bx[-1].astype(np.uint8))
#plt.savefig('./figures/sample_batch_image.png')
#cat_idx = np.argmax(by[-1])
#cat_id = idx2cat[cat_idx]
#categories_df.loc[cat_id]

#instantiate pre-trained architecture
print "compiling the model..."

"""
#ARCH 1 (feature extraction: new classifier, same resnet base)
resnet = ResNet50(weights='imagenet', include_top=True)

inp = resnet.input
top_layer = Dense(num_classes, activation='softmax')
out = top_layer(resnet.layers[-2].output)
resnet_new = Model(inp, out)

#freeze all layers except for the last one
for l, layer in enumerate(resnet_new.layers[:-1]):
    layer.trainable = False

#ensure the last layer is trainable
for l, layer in enumerate(resnet_new.layers[-1:]):
    layer.trainable = True
"""

#ARCH 2 (fine tuning: new classifer and tune ResNet stage 5 using much lower learning rate)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

resnet_new = Sequential()
resnet_new.add(resnet_base)
resnet_new.add(Flatten())
resnet_new.add(Dense(num_classes, activation='softmax'))

#fine tuning
resnet_base.trainable = True

set_trainable = False
for layer in resnet_base.layers:
    if layer.name == 'res5a_branch2a':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    #end if
#end for

#TODO: ensemble of fine-tuned networks with ensemble weights tuned on validation data

#feature extraction (lr=1e-3)
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#fine tuning (lr=1e-4)
adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
resnet_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
resnet_new.summary()

#create callbacks
file_name = DATA_PATH + 'resnet-new-weights-checkpoint.hdf5'
checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
hist_lr = LR_hist()
reduce_lr = LearningRateScheduler(step_decay) 
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=64, verbose=1)
callbacks_list = [checkpoint, tensor_board, hist_lr, reduce_lr, early_stopping]

#train the model
print "model training..."
hist_resnet_new = resnet_new.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=2, validation_data=val_gen, validation_steps=validation_steps, callbacks=callbacks_list)

#save the model and weights
resnet_new.save(DATA_PATH + 'resnet_new_final_model.h5', overwrite=True)
resnet_new.save_weights(DATA_PATH + 'resnet_new_final_weights.h5', overwrite=True)

"""
#load saved model
resnet_new = load_model(DATA_PATH + '/trained_models/resnet_new_final_model.h5')
"""

#evaluate the model on test data
print "model prediction..."
submission_df = pd.read_csv('./data/sample_submission.csv')
test_datagen = ImageDataGenerator()

data_test = bson.decode_file_iter(open(test_bson_path, 'rb'))

with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data_test):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])

        batch_x = np.zeros((num_imgs, 224, 224, 3), dtype=K.floatx()) #(180, 180)

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(224, 224)) #(180, 180)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x

        prediction = resnet_new.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)

        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]        
        pbar.update()

submission_df.to_csv("./data/fourth_submission.csv.gz", compression="gzip", index=False)

#generate plots
#TODO: show both training and validation loss / acc on one plot
plt.figure()
plt.plot(hist_resnet_new.history['val_loss'], label='ResNet50 new')
plt.title('ResNet50 Transfer Learning')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.savefig('./figures/resnet50_new_val_loss.png')

plt.figure()
plt.plot(hist_resnet_new.history['val_acc'], label='ResNet50 new')
plt.title('ResNet50 Transfer Learning')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.savefig('./figures/resnet50_new_val_acc.png')

plt.figure()
plt.plot(hist_lr.lr, lw=2.0, label='learning_rate')
plt.title("ResNet50 model")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.legend()
plt.savefig('./figures/resnet50_learning_rate.png')

plot_model(resnet_new, to_file='./figures/resnet50_model.png')


