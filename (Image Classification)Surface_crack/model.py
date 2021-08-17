import tensorflow as tf
import numpy as np


def model_structure(vgg_model):
    # defining a sequential model to learn
    clf_model = tf.keras.Sequential()

    # adding pretrained model
    clf_model.add(vgg_model)

    # using global average pooling instead of flatten and global max pooling
    clf_model.add(tf.keras.layers.GlobalAveragePooling2D())

    clf_model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    clf_model.add(tf.keras.layers.Dropout(0.3))

    clf_model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    clf_model.add(tf.keras.layers.Dropout(0.3))

    clf_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    clf_model.summary()
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return clf_model


def model_params(train_df, val_df, test_df, train_bs, val_bs, test_bs):
    steps_train = np.ceil(train_df.shape[0] / train_bs)
    steps_val = np.ceil(val_df.shape[0] / val_bs)
    steps_test = np.ceil(test_df.shape[0] / test_bs)
    print("Steps for training:", str(steps_train) + ',', "validation:", str(steps_val) + ',',
          "testing:", str(steps_test))
    return steps_train, steps_val, steps_test


def model_train(model, train_generator, steps_train, val_generator, steps_val):
    history = model.fit(train_generator, steps_per_epoch=steps_train, validation_data=val_generator, epochs=1,
                        validation_steps=steps_val, verbose=1)
    return history


def model_eval(model, test_generator, steps_test):
    model_val = model.evaluate_generator(test_generator, steps=steps_test, verbose=1)
    return model_val
