#!/usr/bin/env python3
"""Show 20 gradient combinations using Rich (no yellow or green)"""
from rich.console import Console
from rich.text import Text

console = Console(force_terminal=True, color_system="truecolor")

gradients = [
    ("Orange → Cyan",       (255, 165, 0),   (0, 255, 255)),
    ("Red → Cyan",          (255, 80, 80),   (0, 255, 255)),
    ("Red → White",         (255, 80, 80),   (255, 255, 255)),
    ("Red → Light Blue",    (255, 80, 80),   (100, 180, 255)),
    ("Magenta → Cyan",      (255, 0, 255),   (0, 255, 255)),
    ("Magenta → White",     (255, 0, 255),   (255, 255, 255)),
    ("Pink → Cyan",         (255, 105, 180), (0, 255, 255)),
    ("Pink → Light Blue",   (255, 105, 180), (100, 200, 255)),
    ("Pink → White",        (255, 105, 180), (255, 255, 255)),
    ("Orange → White",      (255, 165, 0),   (255, 255, 255)),
    ("Orange → Light Blue", (255, 165, 0),   (100, 180, 255)),
    ("Coral → Sky Blue",    (255, 127, 80),  (135, 206, 235)),
    ("Coral → Cyan",        (255, 127, 80),  (0, 255, 255)),
    ("Salmon → Ice Blue",   (250, 128, 114), (173, 216, 230)),
    ("Tomato → Aqua",       (255, 99, 71),   (127, 255, 212)),
    ("Peach → Lavender",    (255, 180, 130), (200, 170, 255)),
    ("Rust → Teal",         (183, 65, 14),   (0, 206, 209)),
    ("Brick → Powder Blue", (203, 65, 84),   (176, 224, 230)),
    ("Copper → Periwinkle", (184, 115, 51),  (150, 150, 255)),
    ("Tangerine → Violet",  (255, 140, 0),   (180, 130, 255)),
]

for name, start, end in gradients:
    text = Text()
    text.append(f"{name:<25} ")
    for i in range(20):
        t = i / 19
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        text.append("█", style=f"rgb({r},{g},{b})")
    text.append(f"  {start} → {end}")
    console.print(text)
