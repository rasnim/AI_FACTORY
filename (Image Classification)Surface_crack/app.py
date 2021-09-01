from preprocessing import *
from augmentation import *
from model import *

FILE_PATH = 'IMAGE_FILE_PATH'

if __name__ == '__main__':
    # data preprocessing
    img_paths, labels = image_load(FILE_PATH)
    train_df, val_df, test_df = dataset_split_main(img_paths, labels)

    # data augmentation
    # base augmentation : pretrained model & preprocessing input
    prep_func, vgg_model = base_augment()

    # dataset generator
    train_datagen = df_augment(prep_func, rot_range=10, zoom_range=0.1, width_shift=0.1, height_shift=0.1,
                               hor_flip=True, ver_flip=True)

    val_datagen = df_augment(prep_func, rot_range=0, zoom_range=0, width_shift=0, height_shift=0,
                             hor_flip=False, ver_flip=False)

    # dataset
    train_bs, train_generator = dataset_generator(batch_size=16, dataset_datagen=train_datagen, dataset_df=train_df,
                                                  cls_mod='binary', shuf=True)
    val_bs, val_generator = dataset_generator(batch_size=16, dataset_datagen=val_datagen, dataset_df=val_df,
                                              cls_mod='binary', shuf=False)
    test_bs, test_generator = dataset_generator(batch_size=16, dataset_datagen=val_datagen, dataset_df=test_df,
                                                cls_mod='binary', shuf=False)

    # model
    clf_model = model_structure(vgg_model)
    steps_train, steps_val, steps_test = model_params(train_df, val_df, test_df, 16, 16, 16)
    history = model_train(clf_model, train_generator, steps_train, val_generator, steps_val)
    model_val = model_eval(clf_model, test_generator, steps_test)
