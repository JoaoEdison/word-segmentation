# Substituir todas as constantes por parâmetros.

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
def get_blocks(rectangles):
    rectangles = merge_contours(rectangles, 70, 70)
    rectangles = merge_contours(rectangles, overlap=0.01, by_overlap=True)
    rectangles = list(filter(lambda r : filter_recs_by_size(r, 300, 1025, 300, 1025), rectangles))
    return rectangles

# Ordena as seleções de cima para baixo e da esquerda para a direita.
def sort_recs(a,b):
    if abs(a[1]-b[1])<=25:
        return a[0] - b[0]
    return a[1] - b[1]

def get_chars(box, rectangles):
    rectangles = list(filter(lambda r : inside(box, r), rectangles))
    rectangles = list(map(lambda r : (r[0]-box[0], r[1]-box[1], r[2], r[3]), rectangles))
    rectangles = merge_contours(rectangles, 15, 7)
    
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
 

def main():
    pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE);
    pr.init_window(512, 512, "GUI")

    files = []
    directory = "Page_Level_Training_Set"
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".jpeg") or \
                file.endswith(".png"):
            files.append(file)
            if len(files) == 5:
                break
    fname = os.path.join(directory, files[0])
    print(fname)

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
    
    blocks = get_blocks(rectangles)
    blocks_img = draw_rectangles(img, blocks, (0, 255, 0), 2)
    blocks_img = cv.resize(blocks_img, (512, 512))
    blocks_img = cv.cvtColor(blocks_img, cv.COLOR_BGR2RGB)
    
    chars = get_chars(blocks[0], rectangles)
    crop = img[blocks[0][1]:blocks[0][1]+blocks[0][3], blocks[0][0]:blocks[0][0]+blocks[0][2]]
    chars_img = draw_rectangles(crop, chars, (0, 255, 0), 1)
    chars_img = cv.resize(chars_img, (512, 512))
    chars_img = cv.cvtColor(chars_img, cv.COLOR_BGR2RGB)
    
    ray_img_blocks = pr.Image(blocks_img.data, 512, 512, 1, 4)
    texture_blocks = pr.load_texture_from_image(ray_img_blocks)
    ray_img_chars = pr.Image(chars_img.data, 512, 512, 1, 4)
    texture_chars = pr.load_texture_from_image(ray_img_chars)

    pr.set_target_fps(10)
    action = True
    while not pr.window_should_close():
        if action:
            pixels = pr.load_image_colors(pr.Image())
            pr.update_texture(texture_blocks, pixels)
            pr.unload_image_colors(pixels)
            action = False
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.draw_texture(texture_blocks, 0, 0, pr.WHITE)
        pr.draw_texture(texture_chars, 0, 512, pr.WHITE)
        pr.end_drawing()
    
    pr.unload_texture(texture_blocks)
    pr.unload_texture(texture_chars)

    pr.close_window()

if __name__ == '__main__':
    main()
