from sklearn.model_selection import train_test_split

import os
import pandas as pd


def image_load(data_path):
    # initializing lists to store file paths for training and validation
    img_paths = []

    # importing libraries to store label references
    labels = []

    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            path = os.path.join(dirname, filename)

            if '.jpg' in path:
                img_paths.append(path)
                labels.append(path.split(os.path.sep)[-2])

    return img_paths, labels


def dataset_split_main(img_paths, labels):
    main_df = pd.DataFrame({'Path': img_paths, 'Label': labels}).sample(frac=1, random_state=10)

    oX_train, X_test, oy_train, y_test = train_test_split(main_df['Path'], main_df['Label'], test_size=0.2,
                                                          stratify=main_df['Label'], shuffle=True, random_state=20)

    X_train, X_val, y_train, y_val = train_test_split(oX_train, oy_train, test_size=0.2, stratify=oy_train,
                                                      shuffle=True, random_state=40)
    
    train_df = pd.DataFrame({'Path': X_train, 'Label': y_train})
    val_df = pd.DataFrame({'Path': X_val, 'Label': y_val})
    test_df = pd.DataFrame({'Path': X_test, 'Label': y_test})
    return train_df, val_df, test_df