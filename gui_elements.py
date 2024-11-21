'''
Componentes da interface gráfica de usuário.

MIT License

Copyright (c) 2024 João Edison Roso Manica

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import pyray as pr
import cv2 as cv

class ZoomImage:
    def __init__(self, rec, zoom_images):
        self.rec = rec
        self.zoom_images = zoom_images
        self.texture = pr.load_texture_from_image(self.get_pr_img(zoom_images[0]))

    def get_pr_img(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return pr.Image(img.data, int(self.rec.width), int(self.rec.height), 1, 4)

    def update_texture(self, img):
        pr.update_texture(self.texture, self.get_pr_img(img).data)

class Slider:
    def __init__(self, rec, text, min_v, init_v, max_v):
        self.rec = rec
        self.text = text
        self.var = pr.ffi.new('float *', init_v)
        self.min_v = min_v
        self.init_v = init_v
        self.max_v = max_v
    
    def get_value(self):
        return self.var[0]

    def reset_var(self):
        self.var[0] = self.init_v

    # Retorna True se houve mudança
    def draw(self):
        prev = self.var[0]
        pr.gui_slider_bar(self.rec, self.text, f'{int(self.var[0])}', self.var,
                          self.min_v, self.max_v)
        return prev != self.var[0]


class Button:
    def __init__(self, rec, text, font_size, background_color=pr.SKYBLUE,
            border_color=pr.DARKBLUE, hover_background_color=pr.WHITE,
            hover_border_color=pr.BLUE):
        self.rec = rec
        self.text = text
        self.font_size = font_size
        self.background_color = self.default_background_color = background_color
        self.border_color = self.default_border_color = border_color
        self.hover_background_color = hover_background_color
        self.hover_border_color = hover_border_color
        self.actual_background_color = self.background_color

    def hover(self):
        self.background_color = self.hover_background_color
        self.border_color = self.hover_border_color

    def default(self):
        self.background_color = self.default_background_color
        self.border_color = self.default_border_color
    
    def draw(self):
        pr.draw_rectangle_rec(self.rec, self.background_color)
        pr.draw_rectangle_lines(int(self.rec.x), int(self.rec.y),\
                int(self.rec.width), int(self.rec.height),\
                self.border_color)
        pr.draw_text(self.text, int(int(self.rec.x) + int(self.rec.width)//2 -\
            pr.measure_text(self.text, self.font_size)/2), int(self.rec.y + 20), self.font_size,\
            self.border_color)

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
