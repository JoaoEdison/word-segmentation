import numpy as np
import cv2 as cv
import os
import re
import math
import functools
from itertools import accumulate
import pyray as pr
from CTCLayer import CTCLayer

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Remove contornos muito pequenos e muito grandes.
def filter_recs_by_size(r, min_w, max_w, min_h, max_h):
    return min_w<r[2]<max_w and min_h<r[3]<max_h

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

# Assumindo que a coordenada x de a é menor que b.
def inside(a, b):
    return b[1] >= a[1] and b[1]+b[3] <= a[1]+a[3] and b[0]+b[2] <= a[0]+a[2]
def intersec(a, b):
    ay = a[1]+a[3]
    by = b[1]+b[3]
    overy = ay-b[1] if by >= ay else by-a[1]
    area = ((a[0]+a[2])*ay) + ((b[0]+b[2])*by)
    return max(a[0]+a[2]-b[0],0)*max(overy,0)/area

# Junta todos os contornos muito próximos no eixo x ou que estão completamente
# dentro de um maior.
def merge_contours(contours, dist_y=15, dist_x=7, overlap=0.25, by_overlap=False):
    contours.sort(key=lambda tup: tup[0])
    rectangles_merged = contours
    merge = True
    if by_overlap:
        while merge:
            merge = False
            prev = rectangles_merged
            rectangles_merged = []
            for i in range(len(prev)-1):
                j = i+1
                while j < len(prev):
                    if intersec(prev[i], prev[j]) > overlap:
                        rectangles_merged.append(union(prev[i], prev[j]))
                        merge = True
                        break
                    j += 1
                if merge:
                    for k in range(i+1, len(prev)):
                        if k == j:
                            continue
                        rectangles_merged.append(prev[k])
                    break
                else:
                    if j == len(prev):
                        rectangles_merged.append(prev[i])
                        if i == len(prev)-2:
                            rectangles_merged.append(prev[i+1])
        if len(rectangles_merged) == 0:
            rectangles_merged = prev
    else:
        while merge:
            merge = False
            prev = rectangles_merged
            rectangles_merged = []
            for i in range(len(prev)-1):
                j = i+1
                while j < len(prev):
                    if (abs(prev[i][1]-prev[j][1]) <= dist_y and prev[j][0]-(prev[i][0]+prev[i][2]) <= dist_x) \
                        or inside(prev[i], prev[j]):
                        rectangles_merged.append(union(prev[i], prev[j]))
                        merge = True
                        break
                    j += 1
                if merge:
                    for k in range(i+1, len(prev)):
                        if k == j:
                            continue
                        rectangles_merged.append(prev[k])
                    break
                else:
                    if j == len(prev):
                        rectangles_merged.append(prev[i])
                        if i == len(prev)-2:
                            rectangles_merged.append(prev[i+1])
        if len(rectangles_merged) == 0:
            rectangles_merged = prev
    return rectangles_merged

# Assume que a imagem de entrada está em 1024x1024
def get_blocks(rectangles, dist_y, dist_x):
    rectangles = merge_contours(rectangles, dist_y, dist_x)
    rectangles = merge_contours(rectangles, overlap=0.01, by_overlap=True)
    rectangles = list(filter(lambda r : filter_recs_by_size(r, 100, 1025, 100, 1025), rectangles))
    return rectangles

# Ordena as seleções de cima para baixo e da esquerda para a direita.
def sort_recs(a,b):
    if abs(a[1]-b[1])<=25:
        return a[0] - b[0]
    return a[1] - b[1]

def get_chars(box, rectangles, dist_y, dist_x):
    rectangles = list(filter(lambda r : inside(box, r), rectangles))
    rectangles = list(map(lambda r : (r[0]-box[0], r[1]-box[1], r[2], r[3]), rectangles))
    rectangles = merge_contours(rectangles, dist_y, dist_x)
    rectangles.sort(key=functools.cmp_to_key(sort_recs))
    return rectangles

## Envia para a engine de OCR e junta o texto.
#text = ""
#for r in div_rectangles:
#    crop_img = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
#    result = pytesseract.image_to_string(crop_img, config='--psm 8')
#    text = text + ' ' + result.strip()
#
## Mostra os resultados.
#print("Segmentado: ")
#print(text)
#print("Usando o Tesseract: ")
#print(pytesseract.image_to_string(img))

def draw_rectangles(origin, recs, color, border_size):
    out = origin.copy()
    for r in recs:
        cv.rectangle(out, (r[0],r[1]),(r[0]+r[2], r[1]+r[3]), color, border_size)
    return out

def get_pr_img(img, figure_size):
    img = cv.resize(img, (figure_size, figure_size))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return pr.Image(img.data, figure_size, figure_size, 1, 4)

def update_img(img, recs, color, border_size, figure_size):
    img = draw_rectangles(img, recs, color, border_size)
    return get_pr_img(img, figure_size)

def load_img(fname):
    img = cv.imread(fname)
    img = cv.resize(img, (1024, 1024))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = 255 - cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    closing = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(closing, 1, 2)
    rectangles = map(cv.boundingRect, contours)
    rectangles = list(filter(lambda r : filter_recs_by_size(r, 3, 500, 10, 500), rectangles))
    return img, closing, rectangles

BLOCK_DIST_Y = 39.0
BLOCK_DIST_X = 70.0
WORD_DIST_Y = 15.0
WORD_DIST_X = 21.0

block_dist_y = pr.ffi.new('float *', BLOCK_DIST_Y)
block_dist_x = pr.ffi.new('float *', BLOCK_DIST_X)
word_dist_y = pr.ffi.new('float *', WORD_DIST_Y)
word_dist_x = pr.ffi.new('float *', WORD_DIST_X)

# Referências:
# https://medium.com/@natsunoyuki/ocr-with-the-ctc-loss-efa62ebd8625
# https://keras.io/examples/vision/captcha_ocr
# https://www.tensorflow.org/tutorials/keras/save_and_load

DATASET_SIZE = -1
EPOCHS = 500
batch_size = 15
width_input = 100
height_input = 25

TRAIN_MODEL = False

max_len = -1
alphabet = None
alphabet_to_ind = None
ind_to_alphabet = None

def create_model(num_glyphs):
    print(num_glyphs)
    labels = layers.Input(shape=(None,), name='label', dtype='float32')
    inputs = layers.Input(shape=(width_input, height_input, 1), name='image', dtype='float32') 
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    # TODO: 32
    last_filters = 16
    x = layers.Conv2D(last_filters, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Reshape(((width_input-4)//2, (height_input-4)//2*last_filters))(x)
    x = layers.Dense(last_filters*10, activation='relu')(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(num_glyphs+1, activation='softmax', name='dense2')(x)
    output = CTCLayer()(labels, x)
    model = models.Model(inputs=[inputs, labels], outputs=output)
    model.compile(optimizer=optimizers.Adam())
    return model

test_image = 'test-images/81.jpg'

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

# Determina o comprimento da palavra mais longa e o total de símbolos no dataset:
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
    image = tf.transpose(image, perm = [1, 0])
    label = alphabet_to_ind(tf.strings.unicode_split(label, input_encoding='UTF-8'))
    return {"image": image, "label" : label}

if train_model:
    X_dataset = []
    y_dataset = []
    for file in files:
        word = file[1]+(max_len - len(file[1]))*' '
        y_dataset.append(word)

        gray = cv.imread(file[0], cv.IMREAD_GRAYSCALE)
        gray = cv.resize(gray, (width_input, height_input))
        gray = 255 - cv.GaussianBlur(gray, (3, 3), 0)
        canny = cv.Canny(gray, 50, 150, apertureSize=3)
        kernel = np.ones((3,3), np.uint8)
        closing = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
        closing = np.array(closing.astype(np.uint8))
        X_dataset.append(closing)

    # Divide em conjunto de treino e de validação (0.7, 0.3)
    split = round(len(X_dataset)*0.7)
    X_train_set = np.array(X_dataset[:split])
    y_train_set = np.array(y_dataset[:split])
    X_valid_set = np.array(X_dataset[split:])
    y_valid_set = np.array(y_dataset[split:])
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_set, y_train_set))
    train_ds = train_ds.map(encode_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size = tf.data.AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid_set, y_valid_set))
    valid_ds = valid_ds.map(encode_data, num_parallel_calls=tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = valid_ds.prefetch(buffer_size = tf.data.AUTOTUNE)

model = create_model(len(alphabet_to_ind.get_vocabulary()))
checkpoint_path = 'training/cp-{epoch:04d}.weights.h5'
checkpoint_dir = 'training'
weights_dir = list(sorted(os.listdir(checkpoint_dir)))

if train_model:
    cp_callback =\
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,\
            save_weights_only=True, verbose=1, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",\
            patience=10, restore_best_weights=True)
    
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
                                model.get_layer(name = 'dense2').output)

def decode_batch_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, 
                      input_length = input_len, 
                      greedy = True)[0][0][:, :max_len]
    output = []
    for res in results:
        # Convert the predicted indices to the corresponding chars.
        res = tf.strings.reduce_join(ind_to_alphabet(res)).numpy().decode("utf-8")
        output.append(res)
    return output

def recognize(closing, block, chars):
    X_test = []
    y_test = []
    for rec in chars:
        y = block[1]+rec[1]
        x = block[0]+rec[0]
        crop = closing[y:y+rec[3], x:x+rec[2]]
        crop = cv.resize(crop, (width_input, height_input))
        crop = np.array(crop.astype(np.uint8))
        X_test.append(crop)
        y_test.append("")
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.map(encode_data, num_parallel_calls= tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
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

#   raylib [text] example - Rectangle bounds
#
#   Example originally created with raylib 2.5, last time updated with raylib 4.0
#
#   Example contributed by Vlad Adrian (@demizdor) and reviewed by Ramon Santamaria (@raysan5)
#
#   Example licensed under an unmodified zlib/libpng license, which is an OSI-certified,
#   BSD-like license that allows static linking with closed source software
#
#   Copyright (c) 2018-2024 Vlad Adrian (@demizdor) and Ramon Santamaria (@raysan5)
def draw_text_boxed(font, text, x, width, y, height, font_size, color):
    spacing = 1.0

    length = pr.text_length(text)
    scale_factor = font_size/float(font.baseSize)
    draw = False
    start_line = -1
    end_line = -1
    text_offset_y = 0.0
    text_offset_x = 0.0

    codepointByteCount = pr.ffi.new('int *', 0)

    i = 0
    while i < length:
        codepointByteCount[0] = 0
        codepoint = pr.get_codepoint(text[i], codepointByteCount)
        index = pr.get_glyph_index(font, codepoint)

        if codepoint == 0x3f:
            codepointByteCount[0] = 1
        i += (codepointByteCount[0] - 1)

        glyph_width = 0.0
        if font.glyphs[index].advanceX == 0:
            glyph_width = font.recs[index].width*scale_factor
        else:
            glyph_width = font.glyphs[index].advanceX*scale_factor

        if i + 1 < length:
            glyph_width = glyph_width + spacing
        
        if draw:
            if text_offset_y + font.baseSize*scale_factor > height:
                break

            if codepoint != ' ':
                pr.draw_text_codepoint(font, codepoint,
                                       pr.Vector2(x + text_offset_x, y + text_offset_y),
                                       font_size, color)
            if i == end_line:
                text_offset_y += (float(font.baseSize) + float(font.baseSize)/2)*scale_factor
                text_offset_x = 0.0
                start_line = end_line
                end_line = -1
                glyph_width = 0.0

                draw = not draw
        else:
            if codepoint == ' ':
                end_line = i

            if text_offset_x + glyph_width > width:
                end_line = i if end_line < 1 else end_line
                if i == end_line:
                    end_line -= codepointByteCount[0]
                if start_line + codepointByteCount[0] == end_line:
                    end_line = (i - codepointByteCount[0])
                draw = not draw
            elif i+1 == length:
                end_line = i
                draw = not draw

            if draw:
                text_offset_x = 0.0
                i = start_line
                glyph_width = 0.0

        if text_offset_x != 0 or codepoint != ' ':
            text_offset_x += glyph_width
        i += 1

def main():
    figure_size = 320
    SLIDER_WIDTH = 320
    MARGIN = 800
    WIDTH_BUTTON = 128
    HEIGHT_BUTTON = 64
    DEFAULT_FONT_SIZE = 20

    X_COORDS = [MARGIN]
    Y_COORDS = [i//2*figure_size+64 if i%2 == 1 else i//2*figure_size for i in range(0,4)]
    Y_COORDS.append(figure_size+2*64)
    Y_COORDS.append(figure_size+3*64+20)

    pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE)
    pr.init_window(0, 0, "GUI")
    pr.gui_set_style(pr.DEFAULT, pr.TEXT_SIZE, DEFAULT_FONT_SIZE)
    
    FONT = pr.get_font_default()
    full_height = pr.get_screen_height()
    full_width = pr.get_screen_width()

    scale_y = full_height / (3.5*figure_size)
    Y_COORDS = list(map(lambda c : int(c*scale_y), Y_COORDS))
    figure_size = int(figure_size*scale_y)
    
    img, closing, rectangles = load_img(test_image)
    
    blocks = get_blocks(rectangles, block_dist_y[0], block_dist_x[0])
    chars = get_chars(blocks[0], rectangles, word_dist_y[0], word_dist_x[0])
    blocks_img = update_img(img, blocks, (0,255,0), 2, figure_size)
    crop = img[blocks[0][1]:blocks[0][1]+blocks[0][3], blocks[0][0]:blocks[0][0]+blocks[0][2]]
    chars_img = update_img(crop, chars, (0,255,0), 2, figure_size)
   
    texture_closing = pr.load_texture_from_image(get_pr_img(closing, figure_size))
    texture_blocks = pr.load_texture_from_image(blocks_img)
    texture_chars = pr.load_texture_from_image(chars_img)
     
    reset_button = pr.Rectangle(X_COORDS[0], Y_COORDS[4], WIDTH_BUTTON, HEIGHT_BUTTON)
    run_button = pr.Rectangle(X_COORDS[0]+WIDTH_BUTTON, Y_COORDS[4], WIDTH_BUTTON, HEIGHT_BUTTON)
    text = ""

    pr.set_target_fps(30)
    update_textures = False
    while not pr.window_should_close():
        if pr.is_file_dropped():
            dropped_files = pr.load_dropped_files()
            if pr.is_file_extension(dropped_files.paths[0], ".jpg") or\
                pr.is_file_extension(dropped_files.paths[0], ".jpeg") or\
                pr.is_file_extension(dropped_files.paths[0], ".png"):
                img, closing, rectangles = load_img(pr.ffi.string(dropped_files.paths[0]).decode('utf-8'))
                update_textures = True
            pr.unload_dropped_files(dropped_files)
 
        background_color_button_run = background_color_button_reset = pr.SKYBLUE
        border_color_run = border_color_reset = pr.DARKBLUE

        mouse_position = pr.get_mouse_position()
        pressed = pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT)
        if pr.check_collision_point_rec(mouse_position, reset_button):
            if pressed:
                word_dist_y[0] = WORD_DIST_Y
                word_dist_x[0] = WORD_DIST_X
                block_dist_y[0] = BLOCK_DIST_Y
                block_dist_x[0] = BLOCK_DIST_X
                update_textures = True
            else:
                background_color_button_reset = pr.WHITE
                border_color_reset = pr.BLUE
        elif pr.check_collision_point_rec(mouse_position, run_button):
            if pressed:
                blocks = get_blocks(rectangles, block_dist_y[0], block_dist_x[0])
                if len(blocks) > 0:
                    chars = get_chars(blocks[0], rectangles, word_dist_y[0], word_dist_x[0])
                    text = recognize(closing, blocks[0], chars) if len(blocks) > 0 else ""
            else:
                background_color_button_run = pr.WHITE
                border_color_run = pr.BLUE

        if update_textures:
            blocks = get_blocks(rectangles, block_dist_y[0], block_dist_x[0])
            blocks_img = update_img(img, blocks, (0, 255, 0), 2, figure_size)
            pr.update_texture(texture_blocks, blocks_img.data)
            pr.update_texture(texture_closing, get_pr_img(closing, figure_size).data)
            if len(blocks) > 0:
                chars = get_chars(blocks[0], rectangles, word_dist_y[0], word_dist_x[0])
                crop = img[blocks[0][1]:blocks[0][1]+blocks[0][3], blocks[0][0]:blocks[0][0]+blocks[0][2]]
                chars_img = update_img(crop, chars, (0, 255, 0), 2, figure_size)
                pr.update_texture(texture_chars, chars_img.data)
            else:
                chars_img = update_img(img, [], (0, 255, 0), 2, figure_size)
                pr.update_texture(texture_chars, chars_img.data)
            update_textures = False

        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        pr.draw_texture(texture_closing, 0, 0, pr.WHITE)
        pr.draw_texture(texture_blocks, 0, figure_size, pr.WHITE)
        pr.draw_texture(texture_chars, 0, figure_size*2, pr.WHITE)
        
        prev = block_dist_y[0]
        pr.gui_slider_bar(pr.Rectangle(MARGIN, Y_COORDS[0], SLIDER_WIDTH, HEIGHT_BUTTON),   "Distância limite de mesclagem no eixo Y (blocos)", f'{int(block_dist_y[0])}', block_dist_y, 0, 100)
        update_textures = update_textures or prev != block_dist_y[0]
        prev = block_dist_x[0]
        pr.gui_slider_bar(pr.Rectangle(MARGIN, Y_COORDS[1], SLIDER_WIDTH, HEIGHT_BUTTON),  "Distância limite de mesclagem no eixo X (blocos)", f'{int(block_dist_x[0])}', block_dist_x, 0, 100)
        update_textures = update_textures or prev != block_dist_x[0]
        prev = word_dist_y[0]
        pr.gui_slider_bar(pr.Rectangle(MARGIN, Y_COORDS[2], SLIDER_WIDTH, HEIGHT_BUTTON), "Distância limite de mesclagem no eixo Y (palavras)", f'{int(word_dist_y[0])}',  word_dist_y,  0, 100)
        update_textures = update_textures or prev != word_dist_y[0]
        prev = word_dist_x[0]
        pr.gui_slider_bar(pr.Rectangle(MARGIN, Y_COORDS[3], SLIDER_WIDTH, HEIGHT_BUTTON), "Distância limite de mesclagem no eixo X (palavras)", f'{int(word_dist_x[0])}',  word_dist_x,  0, 100)
        update_textures = update_textures or prev != word_dist_x[0]

        pr.draw_rectangle_rec(reset_button, background_color_button_reset)
        pr.draw_rectangle_rec(run_button, background_color_button_run)
        pr.draw_rectangle_lines(int(reset_button.x), int(reset_button.y),\
                int(reset_button.width), int(reset_button.height),\
                border_color_reset)
        pr.draw_rectangle_lines(int(run_button.x), int(run_button.y),\
                int(run_button.width), int(run_button.height),\
                border_color_run)
        pr.draw_text("Restaurar", int(int(reset_button.x) + int(reset_button.width)//2 -\
            pr.measure_text("Restaurar", DEFAULT_FONT_SIZE)/2), int(Y_COORDS[4] + 25), DEFAULT_FONT_SIZE,\
            border_color_reset)
        pr.draw_text("Executar", int(int(run_button.x) + int(run_button.width)//2 -\
            pr.measure_text("Executar", DEFAULT_FONT_SIZE)/2), int(Y_COORDS[4] + 25), DEFAULT_FONT_SIZE,\
            border_color_run)
        
        draw_text_boxed(FONT, text, X_COORDS[0], full_width-X_COORDS[0]-200,
                        Y_COORDS[5], full_height-Y_COORDS[5], 20, pr.BLUE)

        pr.end_drawing()
    
    pr.unload_texture(texture_blocks)
    pr.unload_texture(texture_chars)

    pr.close_window()

if __name__ == '__main__':
    main()
