import os
import re

import numpy as np
import cv2 as cv

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from CTCLayer import CTCLayer

DATASET_SIZE = -1
EPOCHS = 500
BATCH_SIZE = 16
WIDTH_INPUT = 100
HEIGHT_INPUT = 25
PATIENCE = 4

TRAIN_MODEL = False

max_len = -1
alphabet = None
alphabet_to_ind = None
ind_to_alphabet = None

# Referências:
# https://medium.com/@natsunoyuki/ocr-with-the-ctc-loss-efa62ebd8625
# https://keras.io/examples/vision/captcha_ocr
# https://www.tensorflow.org/tutorials/keras/save_and_load
def create_model(num_glyphs):
    last_filters = 64
    # Entrada
    labels = layers.Input(shape=(None,), name='label', dtype='float32')
    inputs = layers.Input(shape=(WIDTH_INPUT, HEIGHT_INPUT, 1), name='image', dtype='float32') 
    # Convoluções
    x = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal')(inputs)
    x = layers.Conv2D(last_filters, (3,3), activation='relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D((2,2))(x)
    # Reshape e dropout
    x = layers.Reshape(((WIDTH_INPUT-4)//2, (HEIGHT_INPUT-4)//2*last_filters))(x)
    x = layers.Dense(last_filters*2, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    # Saída
    x = layers.Dense(num_glyphs+1, activation='softmax', name='dense2')(x)
    output = CTCLayer()(labels, x)
    model = models.Model(inputs=[inputs, labels], outputs=output)
    model.compile(optimizer=optimizers.Adam())
    return model

dataset_dir = 'Word_Level_Training_Set'
train_model = os.path.isdir(dataset_dir) and TRAIN_MODEL
files = []
if train_model:
    with open(os.path.join(dataset_dir, 'train.txt'), 'r') as file:
        for line in file:
            path_and_word = line.strip().split('\t')
            files.append((os.path.join(dataset_dir, path_and_word[0]), path_and_word[1]))
            if len(files) == DATASET_SIZE:
                break

# Determina o comprimento da palavra mais longa e o total de símbolos no
# dataset.
#words = [file[1] for file in files]
#max_len = max([len(word) for word in words])
#alphabet = set(c for word in words for c in word)
#alphabet = list(sorted(list(alphabet)))
#print(alphabet)
#print(max_len)

alphabet = ['!', '"', '#', '$', '%', '&', "'", '(', ')', ',', '-', '.', '/',\
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '=', '?', 'A',\
        'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',\
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',\
        'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',\
        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
max_len = 18

alphabet_to_ind = layers.StringLookup(vocabulary=alphabet)
ind_to_alphabet = layers.StringLookup(vocabulary=alphabet_to_ind.get_vocabulary(), invert=True)

def encode_data(image, label):
    image = layers.Rescaling(1.0/255)(image)
    image = tf.transpose(image, perm=[1, 0])
    label = alphabet_to_ind(tf.strings.unicode_split(label, input_encoding='UTF-8'))
    return {"image": image, "label" : label}

if train_model:
    X_dataset = []
    y_dataset = []
    for file in files:
        word = file[1]+(max_len - len(file[1]))*' '
        y_dataset.append(word)

        gray = cv.imread(file[0], cv.IMREAD_GRAYSCALE)
        gray = 255 - cv.resize(gray, (WIDTH_INPUT, HEIGHT_INPUT))
        gray = np.array(gray.astype(np.uint8))
        X_dataset.append(gray)

    # Divide em conjunto de treino e de validação (0.7, 0.3)
    split = round(len(X_dataset)*0.7)
    X_train_set = np.array(X_dataset[:split])
    y_train_set = np.array(y_dataset[:split])
    X_valid_set = np.array(X_dataset[split:])
    y_valid_set = np.array(y_dataset[split:])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_set, y_train_set))
    train_ds = train_ds.map(encode_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid_set, y_valid_set))
    valid_ds = valid_ds.map(encode_data, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

model = create_model(len(alphabet_to_ind.get_vocabulary()))
checkpoint_path = 'training/cp-{epoch:03d}.weights.h5'
checkpoint_dir = 'training'
weights_dir = list(sorted(os.listdir(checkpoint_dir)))

if train_model:
    cp_callback =\
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,\
            save_weights_only=True, verbose=1, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",\
            patience=PATIENCE, restore_best_weights=True)
    
    last_epoch = 0
    if len(weights_dir) > 0:
        last_epoch_match = re.search(r'\d+', weights_dir[-1])
        if last_epoch_match:
            last_epoch = int(last_epoch_match.group(0))
            model.load_weights(os.path.join(checkpoint_dir, weights_dir[-1]))
    model.fit(train_ds, epochs=EPOCHS-last_epoch, validation_data=valid_ds,\
            callbacks=[early_stopping, cp_callback], verbose=2)
else:
    model.load_weights(os.path.join(checkpoint_dir, weights_dir[-1]))

model.summary()
prediction_model = models.Model(model.input[0],
                                model.get_layer(name='dense2').output)

def decode_batch_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred,
                                          input_length=input_len,
                                          greedy=True)[0][0][:, :max_len]
    output = []
    for res in results:
        # Convert the predicted indices to the corresponding chars.
        res = tf.strings.reduce_join(ind_to_alphabet(res)).numpy().decode("utf-8")
        output.append(res)
    return output

def recognize(gray, block, chars):
    X_test = []
    y_test = []
    for rec in chars:
        y = block[1]+rec[1]
        x = block[0]+rec[0]
        crop = gray[y:y+rec[3], x:x+rec[2]]
        crop = cv.resize(crop, (WIDTH_INPUT, HEIGHT_INPUT))
        crop = np.array(crop.astype(np.uint8))
        X_test.append(crop)
        y_test.append("")
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.map(encode_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    text = ""
    for batch in test_ds:
        batch_image = batch["image"]
        pred = prediction_model.predict(batch_image)
        words = decode_batch_prediction(pred)
        for word in words:
            text += word.strip("[UNK]") + ' '
    print(text)
    return text
