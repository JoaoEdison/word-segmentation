# https://pylessons.com/ctc-text-recognition

import numpy as np
import cv2 as cv
import os
import math
import functools
from itertools import accumulate
import pytesseract
import pyray as pr

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
    
    # Divide em duas palavras retângulos muito altos.
    div_rectangles = []
    for r in rectangles:
        if r[3] > 60:
            div_rectangles.append((r[0], r[1],         r[2], r[3]//2))
            div_rectangles.append((r[0], r[1]+r[3]//2, r[2], r[3]//2))
        else:
            div_rectangles.append(r)

    div_rectangles.sort(key=functools.cmp_to_key(sort_recs))
    
    return div_rectangles

    ## Segmenta as palavras em caracteres.
    #min_w = 500
    #for r in div_rectangles:
    #    if r[2] < min_w:
    #        min_w = r[2]
    #if min_w > 25 or min_w < 17:
    #    mean = int(round(functools.reduce(lambda a, b: (0, 0, 0, a[3]+b[3]),\
    #        div_rectangles)[3] / len(div_rectangles)))
    #    min_w = min(mean, min_w)
    #
    #new_rectangles = []
    #for r in div_rectangles:
    #    for i in range(r[0], r[0]+r[2], min_w):
    #        new_rectangles.append((i, r[1], min_w, r[3]))
    #return new_rectangles
    
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

word_dist_y = pr.ffi.new('float *', 15.0)
word_dist_x = pr.ffi.new('float *', 7.0)
block_dist_y = pr.ffi.new('float *', 70.0)
block_dist_x = pr.ffi.new('float *', 70.0)

# Para cada um dos arquivos, carregar todos os retângulos encontrados e colocar
# junto do texto de referência.
# Dividir em conjunto de treino e de validação (0.7, 0.3)
# Limpar a rede
# Para cada retângulo, redimensionar para um multiplo do tamanho da entrada
# enviar cada fração para a rede
# Computar o erro
# Acompanhar o treinamento
# Salvar a rede
# Se a rede existir, carregá-la em vez de treinar
# Obter o texto.
def train_net(files):
    dataset = []
    for file in files:
        img, closing, rectangles = load_img(file[0])
        blocks = get_blocks(rectangles, block_dist_y[0], block_dist_x[0])
        chars = get_chars(blocks[0], rectangles, word_dist_y[0], word_dist_x[0])
        img_words = []
        for rec in chars:
            img_words.append(closing[blocks[0][1]+rec[1]: blocks[0][0]+rec[0]])
        dataset.append((img_words, file[1].split(' ')))
    

def main():
    figure_size = 320
    SLIDER_WIDTH = 256
    MARGIN = 800
    X_COORDS = [MARGIN]
    Y_COORDS = [i//2*figure_size+64 if i%2 == 1 else i//2*figure_size for i in range(0,4)]
    Y_COORDS.append(figure_size+2*64)

    pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE);
    pr.init_window(0, 0, "GUI")

    full_height = pr.get_screen_height()
    scale_y = full_height / (3.5*figure_size)
    Y_COORDS = list(map(lambda c : int(c*scale_y), Y_COORDS))
    figure_size = int(figure_size*scale_y)
    
    files = []
    directory = "test-images"
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".jpeg") or \
                file.endswith(".png"):
            file_name = os.path.join(directory, file)
            text_file = os.path.join(directory, file.split('.')[0] + ".txt")
            with open(text_file, "r") as f:
                content = f.read()
                files.append((file_name, content))
            if len(files) == 10:
                break

    train_net(files)

    img, closing, rectangles = load_img(files[0][0])
    
    blocks = get_blocks(rectangles, block_dist_y[0], block_dist_x[0])
    chars = get_chars(blocks[0], rectangles, word_dist_y[0], word_dist_x[0])
    blocks_img = update_img(img, blocks, (0,255,0), 2, figure_size)
    crop = img[blocks[0][1]:blocks[0][1]+blocks[0][3], blocks[0][0]:blocks[0][0]+blocks[0][2]]
    chars_img = update_img(crop, chars, (0,255,0), 2, figure_size)
   
    texture_closing = pr.load_texture_from_image(get_pr_img(closing, figure_size))
    texture_blocks = pr.load_texture_from_image(blocks_img)
    texture_chars = pr.load_texture_from_image(chars_img)
    
    run_button = pr.Rectangle(X_COORDS[0], Y_COORDS[4], 128, 64)

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
        
        background_color_button = pr.SKYBLUE
        border_color = pr.DARKBLUE
        if pr.check_collision_point_rec(pr.get_mouse_position(), run_button):
            if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
                word_dist_y[0] = 15.0
                word_dist_x[0] = 7.0
                block_dist_y[0] = 70.0
                block_dist_x[0] = 70.0
                update_textures = True
            else:
                background_color_button = pr.WHITE
                border_color = pr.BLUE

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
        pr.gui_slider_bar(pr.Rectangle(800, Y_COORDS[0], SLIDER_WIDTH, 64),   "Distância limite de mesclagem no eixo Y (blocos)", f'{int(block_dist_y[0])}', block_dist_y, 0, 100)
        update_textures = update_textures or prev != block_dist_y[0]
        prev = block_dist_x[0]
        pr.gui_slider_bar(pr.Rectangle(800, Y_COORDS[1], SLIDER_WIDTH, 64),  "Distância limite de mesclagem no eixo X (blocos)", f'{int(block_dist_x[0])}', block_dist_x, 0, 100)
        update_textures = update_textures or prev != block_dist_x[0]
        prev = word_dist_y[0]
        pr.gui_slider_bar(pr.Rectangle(800, Y_COORDS[2], SLIDER_WIDTH, 64), "Distância limite de mesclagem no eixo Y (palavras)", f'{int(word_dist_y[0])}',  word_dist_y,  0, 100)
        update_textures = update_textures or prev != word_dist_y[0]
        prev = word_dist_x[0]
        pr.gui_slider_bar(pr.Rectangle(800, Y_COORDS[3], SLIDER_WIDTH, 64), "Distância limite de mesclagem no eixo X (palavras)", f'{int(word_dist_x[0])}',  word_dist_x,  0, 100)
        update_textures = update_textures or prev != word_dist_x[0]

        pr.draw_rectangle_rec(run_button, background_color_button)
        pr.draw_rectangle_lines(int(run_button.x), int(run_button.y), int(run_button.width), int(run_button.height), border_color)
        pr.draw_text("Resetar", int(int(run_button.x) + int(run_button.width)//2 -\
            pr.measure_text("Resetar", 10)/2), int(Y_COORDS[4] + 30), 10,\
            border_color)

        pr.end_drawing()
    
    pr.unload_texture(texture_blocks)
    pr.unload_texture(texture_chars)

    pr.close_window()

if __name__ == '__main__':
    main()
