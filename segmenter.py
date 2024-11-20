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
import functools
from itertools import accumulate
import pyray as pr

from rectangles import *
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
    rectangles.sort(key=functools.cmp_to_key(sort_recs))
    return rectangles

def draw_rectangles(origin, recs, color, border_size):
    out = origin.copy()
    for i, r in enumerate(recs):
        cv.rectangle(out, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), color, border_size)
        cv.putText(out, f"{i+1}", (r[0]+4, r[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out

def get_pr_img(img, figure_size):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return pr.Image(img.data, figure_size, figure_size, 1, 4)

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
def draw_text_boxed(font, text, box, font_size, color):
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
            if text_offset_y + font.baseSize*scale_factor > box.height:
                break

            if codepoint != ' ':
                pr.draw_text_codepoint(font, codepoint,
                                       pr.Vector2(box.x + text_offset_x, box.y + text_offset_y),
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

            if text_offset_x + glyph_width > box.width:
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
    BLOCK_DIST_Y = 39.0
    BLOCK_DIST_X = 70.0
    WORD_DIST_Y = 15.0
    WORD_DIST_X = 21.0
    block_dist_y = pr.ffi.new('float *', BLOCK_DIST_Y)
    block_dist_x = pr.ffi.new('float *', BLOCK_DIST_X)
    word_dist_y = pr.ffi.new('float *', WORD_DIST_Y)
    word_dist_x = pr.ffi.new('float *', WORD_DIST_X)

    figure_size = 320
    SLIDER_WIDTH = 320
    PADDING = 800
    MARGIN = 20
    WIDTH_BUTTON = 128
    HEIGHT_BUTTON = 64
    DEFAULT_FONT_SIZE = 20
    TEST_IMAGE = 'test-images/251.jpg'

    X_COORDS = [PADDING]
    Y_COORDS = [i//3*figure_size+64*(i%3) for i in range(0, 6)]
    Y_COORDS.append(figure_size+3*64+20)

    pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE)
    pr.init_window(0, 0, "GUI")
    pr.gui_set_style(pr.DEFAULT, pr.TEXT_SIZE, DEFAULT_FONT_SIZE)
    
    FONT = pr.get_font_default()
    full_height = pr.get_screen_height()
    full_width = pr.get_screen_width()

    scale_y = full_height / (3.5*figure_size)
    Y_COORDS = list(map(lambda c: int(c*scale_y)+MARGIN, Y_COORDS))
    figure_size = int(figure_size*scale_y)

    img, gray, closing, rectangles = load_img(TEST_IMAGE)
    closing_img = cv.resize(closing, (figure_size, figure_size))
    blocks = get_blocks(rectangles, block_dist_y[0], block_dist_x[0])
    chars = get_chars(blocks[0], rectangles, word_dist_y[0], word_dist_x[0])
    blocks_img = draw_rectangles(img, blocks, (0, 255, 0), 2)
    blocks_img = cv.resize(blocks_img, (figure_size, figure_size))
    crop = img[blocks[0][1]:blocks[0][1]+blocks[0][3], blocks[0][0]:blocks[0][0]+blocks[0][2]]
    chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
    chars_img = cv.resize(chars_img, (figure_size, figure_size))
   
    textures = [None]*3
    textures[0] = [pr.Rectangle(MARGIN, Y_COORDS[0], figure_size, figure_size),
                  pr.load_texture_from_image(get_pr_img(closing_img, figure_size)), [closing_img]]
    textures[1] = [pr.Rectangle(MARGIN, Y_COORDS[3], figure_size, figure_size),
                  pr.load_texture_from_image(get_pr_img(blocks_img, figure_size)), [blocks_img]]
    textures[2] = [pr.Rectangle(MARGIN, Y_COORDS[3]+figure_size, figure_size, figure_size),
                  pr.load_texture_from_image(get_pr_img(chars_img, figure_size)), [chars_img]]
    
    prev_button = pr.Rectangle(X_COORDS[0], Y_COORDS[2], WIDTH_BUTTON*2, HEIGHT_BUTTON)
    next_button = pr.Rectangle(X_COORDS[0]+WIDTH_BUTTON*2, Y_COORDS[2], WIDTH_BUTTON*2, HEIGHT_BUTTON)

    reset_button = pr.Rectangle(X_COORDS[0], Y_COORDS[5], WIDTH_BUTTON, HEIGHT_BUTTON)
    run_button = pr.Rectangle(X_COORDS[0]+WIDTH_BUTTON, Y_COORDS[5], WIDTH_BUTTON, HEIGHT_BUTTON)
    
    text_box = pr.Rectangle(X_COORDS[0], Y_COORDS[6],
                            full_width-X_COORDS[0]-200,
                            textures[2][0].y+figure_size-Y_COORDS[6])

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
                img, gray, closing, rectangles = load_img(pr.ffi.string(dropped_files.paths[0]).decode('utf-8'))
                block_idx = 0
                update_textures = True
            pr.unload_dropped_files(dropped_files)
 
        mouse_position = pr.get_mouse_position()
        
        # Zoom
        for t in textures:
            if pr.check_collision_point_rec(mouse_position, t[0]):
                wheel_move = pr.get_mouse_wheel_move()
                if wheel_move > 0 and len(t[2]) < 7:
                    resize_img = cv.resize(t[2][-1], (figure_size*2, figure_size*2))
                    x = int(pr.get_mouse_x() - t[0].x)
                    y = int(pr.get_mouse_y() - t[0].y)
                    resize_img = resize_img[y:y+figure_size, x:x+figure_size]
                    t[2].append(resize_img)
                    pr.update_texture(t[1], get_pr_img(resize_img, figure_size).data)
                elif wheel_move < 0 and len(t[2]) > 1:
                    t[2].pop()
                    resize_img = t[2][-1]
                    pr.update_texture(t[1], get_pr_img(resize_img, figure_size).data)
                break

        pressed = pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT)
        background_color_button_prev = background_color_button_next =\
        background_color_button_run = background_color_button_reset =\
        pr.SKYBLUE
        border_color_prev = border_color_next = border_color_run =\
        border_color_reset = pr.DARKBLUE
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
                if len(blocks) > 0:
                    chars = get_chars(blocks[block_idx], rectangles, word_dist_y[0], word_dist_x[0])
                    text = recognize(gray, blocks[block_idx], chars) if len(blocks) > 0 else ""
            else:
                background_color_button_run = pr.WHITE
                border_color_run = pr.BLUE

        if len(blocks) > 0:
            if pr.check_collision_point_rec(mouse_position, prev_button):
                if pressed:
                    if block_idx > 0:
                        block_idx -= 1
                        chars = get_chars(blocks[block_idx], rectangles, word_dist_y[0], word_dist_x[0])
                        crop =\
                        img[blocks[block_idx][1]:blocks[block_idx][1]+blocks[block_idx][3],
                                blocks[block_idx][0]:blocks[block_idx][0]+blocks[block_idx][2]]
                        chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
                        chars_img = cv.resize(chars_img, (figure_size, figure_size))
                        textures[2][2].clear()
                        textures[2][2].append(chars_img)
                        pr.update_texture(textures[2][1], get_pr_img(chars_img, figure_size).data)
                else:
                    background_color_button_prev = pr.WHITE
                    border_color_prev = pr.BLUE
            elif pr.check_collision_point_rec(mouse_position, next_button):
                if pressed:
                    if block_idx < len(blocks)-1:
                        block_idx += 1
                        chars = get_chars(blocks[block_idx], rectangles, word_dist_y[0], word_dist_x[0])
                        crop =\
                        img[blocks[block_idx][1]:blocks[block_idx][1]+blocks[block_idx][3],
                                blocks[block_idx][0]:blocks[block_idx][0]+blocks[block_idx][2]]
                        chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
                        chars_img = cv.resize(chars_img, (figure_size, figure_size))
                        textures[2][2].clear()
                        textures[2][2].append(chars_img)
                        pr.update_texture(textures[2][1], get_pr_img(chars_img, figure_size).data)
                else:
                    background_color_button_next = pr.WHITE
                    border_color_next = pr.BLUE

        if update_textures:
            blocks = get_blocks(rectangles, block_dist_y[0], block_dist_x[0])
            blocks_img = draw_rectangles(img, blocks, (0, 255, 0), 2)
            blocks_img = cv.resize(blocks_img, (figure_size, figure_size))
            closing_img = cv.resize(closing, (figure_size, figure_size))
            textures[0][2].clear()
            textures[0][2].append(closing_img)
            pr.update_texture(textures[0][1], get_pr_img(closing_img, figure_size).data)
            textures[1][2].clear()
            textures[1][2].append(blocks_img)
            pr.update_texture(textures[1][1], get_pr_img(blocks_img, figure_size).data)
            if len(blocks) > 0:
                chars = get_chars(blocks[block_idx], rectangles, word_dist_y[0], word_dist_x[0])
                crop =\
                img[blocks[block_idx][1]:blocks[block_idx][1]+blocks[block_idx][3],
                        blocks[block_idx][0]:blocks[block_idx][0]+blocks[block_idx][2]]
                chars_img = draw_rectangles(crop, chars, (0, 255, 0), 2)
                chars_img = cv.resize(chars_img, (figure_size, figure_size))
            else:
                chars_img = cv.resize(img, (figure_size, figure_size))
            textures[2][2].clear()
            textures[2][2].append(chars_img)
            pr.update_texture(textures[2][1], get_pr_img(chars_img, figure_size).data)
            update_textures = False

        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        for t in textures:
            pr.draw_texture(t[1], int(t[0].x), int(t[0].y), pr.WHITE)
        
        prev = block_dist_y[0]
        pr.gui_slider_bar(pr.Rectangle(PADDING, Y_COORDS[0], SLIDER_WIDTH, HEIGHT_BUTTON), "Distância limite de mesclagem no eixo Y (blocos)", f'{int(block_dist_y[0])}', block_dist_y, 0, 100)
        update_textures = update_textures or prev != block_dist_y[0]
        prev = block_dist_x[0]
        pr.gui_slider_bar(pr.Rectangle(PADDING, Y_COORDS[1], SLIDER_WIDTH, HEIGHT_BUTTON), "Distância limite de mesclagem no eixo X (blocos)", f'{int(block_dist_x[0])}', block_dist_x, 0, 100)
        update_textures = update_textures or prev != block_dist_x[0]
        prev = word_dist_y[0]
        pr.gui_slider_bar(pr.Rectangle(PADDING, Y_COORDS[3], SLIDER_WIDTH, HEIGHT_BUTTON), "Distância limite de mesclagem no eixo Y (palavras)", f'{int(word_dist_y[0])}',  word_dist_y, 0, 100)
        update_textures = update_textures or prev != word_dist_y[0]
        prev = word_dist_x[0]
        pr.gui_slider_bar(pr.Rectangle(PADDING, Y_COORDS[4], SLIDER_WIDTH, HEIGHT_BUTTON), "Distância limite de mesclagem no eixo X (palavras)", f'{int(word_dist_x[0])}',  word_dist_x, 0, 100)
        update_textures = update_textures or prev != word_dist_x[0]
        
        if len(blocks) > 1:
            if block_idx > 0:
                pr.draw_rectangle_rec(prev_button, background_color_button_prev)
                pr.draw_rectangle_lines(int(prev_button.x), int(prev_button.y),\
                        int(prev_button.width), int(prev_button.height),\
                        border_color_prev)
                pr.draw_text("Anterior", int(int(prev_button.x) + int(prev_button.width)//2 -\
                    pr.measure_text("Anterior", DEFAULT_FONT_SIZE)/2), int(Y_COORDS[2] + 25), DEFAULT_FONT_SIZE,\
                    border_color_prev)
            if block_idx < len(blocks)-1:
                pr.draw_rectangle_rec(next_button, background_color_button_next)
                pr.draw_rectangle_lines(int(next_button.x), int(next_button.y),\
                        int(next_button.width), int(next_button.height),\
                        border_color_next)
                pr.draw_text("Proximo", int(int(next_button.x) + int(next_button.width)//2 -\
                    pr.measure_text("Proximo", DEFAULT_FONT_SIZE)/2), int(Y_COORDS[2] + 25), DEFAULT_FONT_SIZE,\
                    border_color_next)

        pr.draw_rectangle_rec(reset_button, background_color_button_reset)
        pr.draw_rectangle_rec(run_button, background_color_button_run)
        pr.draw_rectangle_lines(int(reset_button.x), int(reset_button.y),\
                int(reset_button.width), int(reset_button.height),\
                border_color_reset)
        pr.draw_rectangle_lines(int(run_button.x), int(run_button.y),\
                int(run_button.width), int(run_button.height),\
                border_color_run)
        pr.draw_text("Restaurar", int(int(reset_button.x) + int(reset_button.width)//2 -\
            pr.measure_text("Restaurar", DEFAULT_FONT_SIZE)/2), int(Y_COORDS[5] + 25), DEFAULT_FONT_SIZE,\
            border_color_reset)
        pr.draw_text("Ler", int(int(run_button.x) + int(run_button.width)//2 -\
            pr.measure_text("Ler", DEFAULT_FONT_SIZE)/2), int(Y_COORDS[5] + 25), DEFAULT_FONT_SIZE,\
            border_color_run)
        
        pr.draw_rectangle_rec(text_box, pr.LIGHTGRAY)
        draw_text_boxed(FONT, text, text_box, 20, pr.BLUE)

        pr.end_drawing()
    
    for t in textures:
        pr.unload_texture(t[1])
    pr.close_window()

if __name__ == '__main__':
    main()
