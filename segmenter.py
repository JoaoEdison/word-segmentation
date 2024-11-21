# MIT License
# 
# Copyright (c) 2024 João Edison Roso Manica
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import cv2 as cv
import math
from functools import cmp_to_key, reduce
from pathlib import Path
import pyray as pr

from rectangles import *
from gui_elements import *
from ocr_model import recognize

# Assume que a imagem de entrada está em 1024x1024.
# Retorna os grandes blocos de texto encontrados na imagem.
def get_blocks(rectangles, dist_y, dist_x):
    rectangles = merge_recs(rectangles, dist_y, dist_x)
    rectangles = merge_recs(rectangles, overlap=0.01, by_overlap=True)
    rectangles = list(filter(lambda r: filter_recs_by_size(r, 100, 1025, 100, 1025), rectangles))
    return rectangles

# Segmenta as palavras dentro de um bloco e retorna elas ordenadas.
def get_chars(box, rectangles, dist_y, dist_x):
    rectangles = list(filter(lambda r: inside(box, r), rectangles))
    rectangles = list(map(lambda r: (r[0]-box[0], r[1]-box[1], r[2], r[3]), rectangles))
    rectangles = merge_recs(rectangles, dist_y, dist_x)
    rectangles.sort(key=cmp_to_key(sort_recs))
    return rectangles

def draw_rectangles(origin, recs, color, border_size):
    out = origin.copy()
    for i, r in enumerate(recs):
        cv.rectangle(out, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), color, border_size)
        cv.putText(out, f"{i+1}", (r[0]+4, r[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out

def load_img(fname):
    img = cv.imread(fname)
    img = cv.resize(img, (1024, 1024))
    gray = 255 - cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, 50, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(closing, 1, 2)
    rectangles = map(cv.boundingRect, contours)
    rectangles = list(filter(lambda r: filter_recs_by_size(r, 3, 500, 10, 500), rectangles))
    return img, gray, closing, rectangles

def main():
    PORTUGUESE = False
    if PORTUGUESE:
        gui_texts = ["Distância limite de mesclagem no eixo Y (blocos)",
                "Distância limite de mesclagem no eixo X (blocos)",
                "Distância limite de mesclagem no eixo Y (palavras)",
                "Distância limite de mesclagem no eixo X (palavras)",
                "Anterior", "Proximo",
                "Restaurar", "Ler", "Salvar"]
    else:
        gui_texts = ["Merge threshold distance on Y axis (blocks)",
                "Merge threshold distance on X axis (blocks)",
                "Merge threshold distance on Y axis (words)",
                "Merge threshold distance on X axis (words)",
                "Previous", "Next",
                "Reset", "Read", "Save"]

    BLOCK_DIST_Y = 39.0
    BLOCK_DIST_X = 70.0
    WORD_DIST_Y = 15.0
    WORD_DIST_X = 21.0
    SLIDER_WIDTH = 320
    PADDING = 100
    MARGIN = 20
    WIDTH_BUTTON = 128
    DEFAULT_FONT_SIZE = 20

    figure_size = 500
    height_button = 64

    SAVE_DIR = 'digitized'
    image_name = 'test-images/251.jpg'
    image_path = SAVE_DIR / Path(Path(image_name).name.split('.')[0]+'.txt')

    pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE)
    pr.init_window(0, 0, "GUI")
    pr.gui_set_style(pr.DEFAULT, pr.TEXT_SIZE, DEFAULT_FONT_SIZE)
    
    FONT = pr.get_font_default()
    full_height = pr.get_screen_height()
    full_width = pr.get_screen_width()

    Y_COORDS = [i//3*figure_size+height_button*(i%3) for i in range(0, 6)]
    Y_COORDS.append(figure_size+3*height_button+20)
    scale_y = full_height / (2.3*figure_size)
    figure_size = int(figure_size*scale_y)
    height_button = int(height_button*scale_y)
    Y_COORDS = list(map(lambda c: int(c*scale_y)+MARGIN, Y_COORDS))
    X_COORDS = [2*figure_size+PADDING]

    sliders = dict()
    padding_slider = pr.measure_text(gui_texts[0], DEFAULT_FONT_SIZE)-32
    sliders['block_y'] = Slider(pr.Rectangle(X_COORDS[0]+padding_slider, Y_COORDS[0], SLIDER_WIDTH, height_button),
                                gui_texts[0], 0, BLOCK_DIST_Y, 100)
    sliders['block_x'] = Slider(pr.Rectangle(X_COORDS[0]+padding_slider, Y_COORDS[1], SLIDER_WIDTH, height_button),
                                gui_texts[1], 0, BLOCK_DIST_X, 100)
    padding_slider = pr.measure_text(gui_texts[2], DEFAULT_FONT_SIZE)-32
    sliders['word_y'] = Slider(pr.Rectangle(X_COORDS[0]+padding_slider, Y_COORDS[3], SLIDER_WIDTH, height_button),
                               gui_texts[2], 0, WORD_DIST_Y, 100)
    sliders['word_x'] = Slider(pr.Rectangle(X_COORDS[0]+padding_slider, Y_COORDS[4], SLIDER_WIDTH, height_button),
                               gui_texts[3], 0, WORD_DIST_X, 100)

    img, gray, closing, rectangles = load_img(image_name)
    closing_img = cv.resize(closing, (figure_size, figure_size))
    blocks = get_blocks(rectangles, sliders['block_y'].get_value(), sliders['block_x'].get_value())
    chars = get_chars(blocks[0], rectangles, sliders['word_y'].get_value(), sliders['word_x'].get_value())
    blocks_img = draw_rectangles(img, blocks, (0, 255, 0), 2)
    blocks_img = cv.resize(blocks_img, (figure_size, figure_size))
    crop = img[blocks[0][1]:blocks[0][1]+blocks[0][3], blocks[0][0]:blocks[0][0]+blocks[0][2]]
    chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
    chars_img = cv.resize(chars_img, (figure_size, figure_size))

    textures = {
        'closing' : ZoomImage(pr.Rectangle(MARGIN, Y_COORDS[0], figure_size, figure_size),
                            [closing_img]),
        'blocks' : ZoomImage(pr.Rectangle(figure_size+MARGIN, Y_COORDS[0], figure_size, figure_size),
                            [blocks_img]),
        'chars' : ZoomImage(pr.Rectangle(MARGIN, Y_COORDS[3], figure_size, figure_size),
                            [chars_img])
    }

    prev_button = Button(pr.Rectangle(X_COORDS[0], Y_COORDS[2], WIDTH_BUTTON,\
        height_button), gui_texts[4], DEFAULT_FONT_SIZE)
    next_button = Button(pr.Rectangle(X_COORDS[0]+WIDTH_BUTTON, Y_COORDS[2],\
        WIDTH_BUTTON, height_button), gui_texts[5], DEFAULT_FONT_SIZE)
    reset_button = Button(pr.Rectangle(X_COORDS[0], Y_COORDS[5], WIDTH_BUTTON,\
        height_button), gui_texts[6], DEFAULT_FONT_SIZE)
    run_button = Button(pr.Rectangle(X_COORDS[0]+WIDTH_BUTTON, Y_COORDS[5],\
        WIDTH_BUTTON, height_button), gui_texts[7], DEFAULT_FONT_SIZE)
    save_button = Button(pr.Rectangle(X_COORDS[0]+WIDTH_BUTTON*2, Y_COORDS[5],\
        WIDTH_BUTTON, height_button), gui_texts[8], DEFAULT_FONT_SIZE)

    text_box = pr.Rectangle(X_COORDS[0], Y_COORDS[6],
                            full_width-X_COORDS[0]-200,
                            textures['chars'].rec.y+figure_size-Y_COORDS[6])

    pr.set_target_fps(30)

    text = ""
    block_idx = 0
    update_textures = False
    while not pr.window_should_close():
        if pr.is_file_dropped():
            dropped_files = pr.load_dropped_files()
            if pr.is_file_extension(dropped_files.paths[0], ".jpg") or\
                pr.is_file_extension(dropped_files.paths[0], ".jpeg") or\
                pr.is_file_extension(dropped_files.paths[0], ".png"):
                image_name = pr.ffi.string(dropped_files.paths[0]).decode('utf-8')
                img, gray, closing, rectangles = load_img(image_name)
                image_path = SAVE_DIR / Path(Path(image_name).name.split('.')[0]+'.txt')
                text = ""
                block_idx = 0
                update_textures = True
            pr.unload_dropped_files(dropped_files)

        mouse_position = pr.get_mouse_position()

        # Zoom
        for t in textures.values():
            if pr.check_collision_point_rec(mouse_position, t.rec):
                wheel_move = pr.get_mouse_wheel_move()
                if wheel_move > 0 and len(t.zoom_images) < 7:
                    resize_img = cv.resize(t.zoom_images[-1], (figure_size*2, figure_size*2))
                    x = int(pr.get_mouse_x() - t.rec.x)
                    y = int(pr.get_mouse_y() - t.rec.y)
                    resize_img = resize_img[y:y+figure_size, x:x+figure_size]
                    t.zoom_images.append(resize_img)
                    t.update_texture(resize_img)
                elif wheel_move < 0 and len(t.zoom_images) > 1:
                    t.zoom_images.pop()
                    resize_img = t.zoom_images[-1]
                    t.update_texture(resize_img)
                break

        pressed = pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT)
        prev_button.default()
        next_button.default()
        reset_button.default()
        run_button.default()
        save_button.default()
        if pr.check_collision_point_rec(mouse_position, reset_button.rec):
            if pressed:
                for s in sliders.values():
                    s.reset_var()
                update_textures = True
            else:
                reset_button.hover()
        elif pr.check_collision_point_rec(mouse_position, run_button.rec):
            if pressed:
                if len(blocks) > 0:
                    chars = get_chars(blocks[block_idx], rectangles, sliders['word_y'].get_value(), sliders['word_x'].get_value())
                    text = recognize(gray, blocks[block_idx], chars) if len(blocks) > 0 else ""
            else:
                run_button.hover()
        elif pr.check_collision_point_rec(mouse_position, save_button.rec):
            if text != "":
                if pressed:
                    Path(SAVE_DIR).mkdir(exist_ok=True)
                    image_path.write_text(text)
                else:
                    save_button.hover()

        if len(blocks) > 0:
            if pr.check_collision_point_rec(mouse_position, prev_button.rec):
                if pressed:
                    if block_idx > 0:
                        block_idx -= 1
                        chars = get_chars(blocks[block_idx], rectangles, sliders['word_y'].get_value(), sliders['word_x'].get_value())
                        crop =\
                        img[blocks[block_idx][1]:blocks[block_idx][1]+blocks[block_idx][3],
                                blocks[block_idx][0]:blocks[block_idx][0]+blocks[block_idx][2]]
                        chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
                        chars_img = cv.resize(chars_img, (figure_size, figure_size))
                        textures['chars'].zoom_images.clear()
                        textures['chars'].zoom_images.append(chars_img)
                        textures['chars'].update_texture(chars_img)
                else:
                    prev_button.hover()
            elif pr.check_collision_point_rec(mouse_position, next_button.rec):
                if pressed:
                    if block_idx < len(blocks)-1:
                        block_idx += 1
                        chars = get_chars(blocks[block_idx], rectangles, sliders['word_y'].get_value(), sliders['word_x'].get_value())
                        crop =\
                        img[blocks[block_idx][1]:blocks[block_idx][1]+blocks[block_idx][3],
                                blocks[block_idx][0]:blocks[block_idx][0]+blocks[block_idx][2]]
                        chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
                        chars_img = cv.resize(chars_img, (figure_size, figure_size))
                        textures['chars'].zoom_images.clear()
                        textures['chars'].zoom_images.append(chars_img)
                        textures['chars'].update_texture(chars_img)
                else:
                    next_button.hover()

        if update_textures:
            blocks = get_blocks(rectangles, sliders['block_y'].get_value(), sliders['block_x'].get_value())
            blocks_img = draw_rectangles(img, blocks, (0, 255, 0), 2)
            blocks_img = cv.resize(blocks_img, (figure_size, figure_size))
            closing_img = cv.resize(closing, (figure_size, figure_size))
            textures['closing'].zoom_images.clear()
            textures['closing'].zoom_images.append(closing_img)
            textures['closing'].update_texture(closing_img)
            textures['blocks'].zoom_images.clear()
            textures['blocks'].zoom_images.append(blocks_img)
            textures['blocks'].update_texture(blocks_img)
            if len(blocks) > 0:
                chars = get_chars(blocks[block_idx], rectangles, sliders['word_y'].get_value(), sliders['word_x'].get_value())
                crop =\
                img[blocks[block_idx][1]:blocks[block_idx][1]+blocks[block_idx][3],
                        blocks[block_idx][0]:blocks[block_idx][0]+blocks[block_idx][2]]
                chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
                chars_img = cv.resize(chars_img, (figure_size, figure_size))
            else:
                chars_img = cv.resize(img, (figure_size, figure_size))
            textures['chars'].zoom_images.clear()
            textures['chars'].zoom_images.append(chars_img)
            textures['chars'].update_texture(chars_img)
            update_textures = False

        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        for t in textures.values():
            pr.draw_texture(t.texture, int(t.rec.x), int(t.rec.y), pr.WHITE)
        
        for s in sliders:
            change = sliders[s].draw()
            if change and s == 'block_y' or s == 'block_x':
                block_idx = 0
            update_textures = update_textures or change
        
        if len(blocks) > 1:
            if block_idx > 0:
                prev_button.draw()
            if block_idx < len(blocks)-1:
                next_button.draw()
        reset_button.draw()
        run_button.draw()
        if text != "":
            save_button.draw()
        
        pr.draw_rectangle_rec(text_box, pr.LIGHTGRAY)
        draw_text_boxed(FONT, text, text_box, 20, pr.BLUE)

        pr.end_drawing()
    
    for t in textures:
        pr.unload_texture(t.texture)
    pr.close_window()

if __name__ == '__main__':
    main()
