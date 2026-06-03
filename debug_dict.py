CHAR_MAP = {
    '0': ('0000011', '00011110', '00011110', '00000011', '00000001', '11000001', '11000001', '11000001', '01100011', '01111110'), 
    '1': ('11100', '00100', '00100', '00100', '00100', '00100', '00100', '00100', '11111', '11111'), # Box 14
    '2': ('01000011', '01111110', '00111110', '01000011', '11000001', '11000001', '11000001', '11000001', '01100011', '00111110'), # Box 5
    '3': ('0000011', '0000110', '0001100', '0011000', '0110000', '1100000', '1000000', '1000000', '1000000', '1111111'), # Box 3
    '4': ('1111', '1001', '1111', '1001', '1111', '1001', '1001', '0010'), # Box 8
    '5': ('01110', '10001', '00001', '00110', '00001', '10001', '01110'), # Box 28
    '6': ('01110', '10001', '10000', '11110', '10001', '10001', '01110'), # Box 27
    '7': ('0011110', '0001100', '0010100', '0110010', '0100011', '1111111', '0111111', '0011100'), # Box 22
    '8': ('0110110', '0100010', '1100011', '1100011', '1100011', '1100011', '1100011', '0100010', '0110110', '0011100'), # Box 15
    '9': ('00011000', '00011000', '00011000', '00011000', '10011001', '10011001', '10011001', '00011000', '00011000', '00111000', '11111110', '11111111'), # Box 23
    '[': ('1111101100100', '0000011111111'), # Box 6
    ']': ('11111111111111', '11111111111110'), # Box 25
    '%': ('00111111001100', '01111111011110', '01111111111110', '11111000011110', '11110000001111', '11110000001111', '11100000001110', '01100000001110', '01110000001100', '00111000011100', '00001111110000', '00000011000000'), # Box 16
    '.': ('010', '111', '010'), # Box 26
    ',': ('01', '11'), # Box 9
}

print('My Dictionary mapping:')
for k, v in CHAR_MAP.items():
    print(f"{k}: {len(v[0])}x{len(v)}")

import cv2
import numpy as np
from PIL import Image

def get_bin(char_img):
    h, w = char_img.shape
    r = []
    for row in range(h):
        r.append(''.join(['1' if char_img[row, c] > 128 else '0' for c in range(w)]))
    return r

img = Image.open('debug_images/test_full.png')
w, h = img.size
exp_bar = img.crop((0, h - 40, w, h))
img_cv = cv2.cvtColor(np.array(exp_bar), cv2.COLOR_RGB2BGR)

h, w = img_cv.shape[:2]
strip = img_cv[:, :int(w*0.55)]

hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])
mask = cv2.inRange(hsv, lower_white, upper_white)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
boxes = []
for i in range(1, num_labels):
    cx, cy, cw, ch, area = stats[i]
    if area >= 2 and ch >= 2 and ch <= 15 and cw <= 15:
        boxes.append((cx, cy, cw, ch))
boxes = sorted(boxes, key=lambda b: b[0])

for i, (cx, cy, cw, ch) in enumerate(boxes[:30]):
    char_img = mask[cy:cy+ch, cx:cx+cw]
    bins = get_bin(char_img)
    best_match = ""
    for k, v in CHAR_MAP.items():
        if len(v) == ch and len(v[0]) == cw:
            if list(v) == bins:
                best_match = k
    print(f'Char {i} [{cw}x{ch}] == {best_match if best_match else "?"}')
    if not best_match:
        for r in bins: print(r)
