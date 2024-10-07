def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
    
train_path = r"Bactnet/Training data/stacks"

batch_size = 16
SIZE = 288
image_dataset = None
mask_dataset = None

def prepare_data(train_path, PATCH_SIZE, delete_empty=False, validation=False, seam=False):
    if validation:
        prefix = "validation"
    else:
        prefix = "training"
    
    stacks = os.listdir(os.path.join(train_path, prefix+"_source"))
    image_dataset = None
    mask_dataset = None
    for stack in stacks:
        if (stack.split(".")[-1]=="tif"):
            img = tiff.imread(os.path.join(train_path, prefix+"_source",stack))
            mask = tiff.imread(os.path.join(train_path, prefix+"_target", stack))
            
            if seam: #should the eges be cropped?
                half = int(PATCH_SIZE/2)
                img = patch_stack(img[1:-1, half:-half, half:-half], PATCH_SIZE, DEPTH = 1)
                mask =patch_stack(mask[: , half:-half, half:-half], PATCH_SIZE, DEPTH = 1)
            else:
                img = patch_stack(img[1:-1], PATCH_SIZE, DEPTH = 1)
                mask =patch_stack(mask, PATCH_SIZE, DEPTH = 1)
            
            print(stack, img.shape, mask.shape)
            mask = normalizeMinMax(mask)
            img = normalizePercentile(img, 0.1, 99.9, clip=True)
            
            if delete_empty:
                not_ok_idxs = checkEmptyMask(mask)
                mask = np.delete(mask, not_ok_idxs, axis=0)
                img = np.delete(img, not_ok_idxs, axis=0)
                print(stack, img.shape, mask.shape)

            

            if image_dataset is not None:
                image_dataset = np.concatenate((image_dataset, img))

            if mask_dataset is not None:
                mask_dataset = np.concatenate((mask_dataset, mask))

            if image_dataset is None:
                image_dataset = img

            if mask_dataset is None:
                mask_dataset = mask

           #print(image_dataset.shape, mask_dataset.shape)

    return image_dataset, mask_dataset

image_dataset, mask_dataset = prepare_data(train_path, SIZE, delete_empty=True, validation=False, seam=False)
seam_data = prepare_data(train_path, SIZE, delete_empty=True, validation=False, seam=False)
X_train = np.concatenate((image_dataset, seam_data[0]))
y_train = np.concatenate((mask_dataset, seam_data[1]))

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.15, random_state = 8)
X_test, y_test = prepare_data(train_path, SIZE, delete_empty=True, validation=True, seam=False)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Sanity check, view few mages
import random

image_number = random.randint(0, X_train.shape[0])
print(image_number)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number, 0, :, :])
plt.subplot(122)
plt.imshow(y_train[image_number, 0, :, :])
plt.show()

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import Activation, MaxPool2D, Concatenate


def conv_block(input, num_filters):
    t = Conv2D(num_filters, 3, padding="same",  data_format="channels_first", activation='relu')(input)
    t = BatchNormalization()(t)
    t = Dropout(0.1)(t)
    t = Conv2D(num_filters, 3, padding="same",  data_format="channels_first", activation='relu')(t)
    t = BatchNormalization()(t)
    t = Dropout(0.1)(t)
    return t

#Encoder block: Conv block followed by maxpooling

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2), data_format="channels_first")(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", data_format="channels_first")(input)
    x = Concatenate(axis=1)([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid", data_format="channels_first")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="BactUnet_single_frame_training")
    return model

keras.backend.clear_session() # Free up RAM in case the model definition cells were run multiple times


#from keras_unet_collection import models
from keras_unet_collection import losses


# Build model
def hybrid_loss(y_true, y_pred):

    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.3, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou #+loss_ssim

SIZE=288
input_shape = (1, SIZE, SIZE)
batch_size = 8

model = build_unet(input_shape)
model.compile(loss=hybrid_loss,
            #loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
                optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=[losses.dice_coef, losses.iou_seg])
model.summary()
print(get_model_memory_usage(batch_size, model))

#New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
#This gives a binary mask rather than a mask with interpolated values. 
seed=1337
from keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(rotation_range=180,
                     width_shift_range=0.25,
                     height_shift_range=0.25,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     data_format="channels_first")

mask_data_gen_args = dict(rotation_range=180,
                     width_shift_range=0.25,
                     height_shift_range=0.25,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     data_format="channels_first",
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

image_data_generator = ImageDataGenerator(**img_data_gen_args)
#image_data_generator.fit(X_train, augment=True, seed=seed)

image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(X_test, seed=seed, batch_size=batch_size) #Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
#mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed, batch_size=batch_size)  #Default batch size 32, if not specified here

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)


x = valid_img_generator.next()
y = valid_mask_generator.next()
print(x.shape, y.shape)
for i in range(len(x)):
    image = x[i]
    fig1 = plt.subplot(4, 4, i+1)
    fig1.set_xticks([])  #Turn off axis
    fig1.set_yticks([])
    plt.imshow(image[0, :,:], cmap='gray')
    #plt.subplot(4,4,2)
    #plt.imshow(mask[0,:,:])
plt.show()

for i in range(len(x)):
    image = y[i]
    fig1 = plt.subplot(4, 4, i+1)
    fig1.set_xticks([])  #Turn off axis
    fig1.set_yticks([])
    plt.imshow(image[0, :,:], cmap='gray')
    #plt.subplot(4,4,2)
    #plt.imshow(mask[0,:,:])
plt.show()


steps_per_epoch = 3*(len(X_train))//batch_size

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

epochs = 100
#model.load_weights(r"/models/bactunet_3frame_local.hdf5")
model_name = "bactunet_V4_single_frame"
#ModelCheckpoint callback saves a model at some interval.

filepath=r"models/"+model_name+".hdf5"
#Use Mode = max for accuracy and min for loss. 
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
#This callback will stop the training when there is no improvement in
# the validation loss for 10 consecutive epochs.

#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger(r'models/'+model_name+'.csv', separator=',', append=False)

callbacks_list = [checkpoint, early_stop, log_csv]

#We can now use these generators to train our model. 
#Give this a name so we can call it later for plotting loss, accuracy etc. as a function of epochs.

history = model.fit(
        my_generator,
        steps_per_epoch=steps_per_epoch,   
        epochs=epochs,
        validation_data=validation_datagen,
        validation_steps=steps_per_epoch,
        callbacks=callbacks_list)

model.save(r"models/"+model_name+".hdf5") 






    