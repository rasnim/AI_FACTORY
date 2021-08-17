import tensorflow as tf

IMAGE_DIMS = (224, 224, 3)


def base_augment():
    prep_func = tf.keras.applications.vgg16.preprocess_input
    vgg_model = tf.keras.applications.vgg16.VGG16(input_shape=IMAGE_DIMS, include_top=False, weights='imagenet')

    for layer in vgg_model.layers:
        layer.trainable = False
    return prep_func, vgg_model


def df_augment(prep_func, rot_range=None, zoom_range=None, width_shift=None, height_shift=None, hor_flip=None,
               ver_flip=None):
    dataset_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prep_func,
                                                                      rotation_range=rot_range,
                                                                      zoom_range=zoom_range,
                                                                      width_shift_range=width_shift,
                                                                      height_shift_range=height_shift,
                                                                      horizontal_flip=hor_flip,
                                                                      vertical_flip=ver_flip)
    return dataset_datagen


def dataset_generator(batch_size, dataset_datagen, dataset_df, cls_mod=None, shuf=True):
    dataset_batch = batch_size

    dataset_generator = dataset_datagen.flow_from_dataframe(dataframe=dataset_df, x_col='Path', y_col='Label',
                                                            target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]),
                                                            batch_size=batch_size, class_mode=cls_mod,
                                                            shuffle=shuf)

    return dataset_batch, dataset_generator
