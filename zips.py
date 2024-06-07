import os
from fontTools.ttLib import TTFont
import shutil

def has_cyrillic_support(font_path):
    try:
        font = TTFont(font_path)
        for table in font['cmap'].tables:
            if any(chr(charcode) >= '\u0400' and chr(charcode) <= '\u04FF' for charcode in table.cmap.keys()):
                return True
        return False
    except Exception as e:
        print(f"Ошибка проверки шрифта '{font_path}': {e}")
        return False

def has_empty_russian_A(font_path):
    try:
        font = TTFont(font_path)
        for table in font['cmap'].tables:
            if ord('А') not in table.cmap.keys():
                return True
        return False
    except Exception as e:
        print(f"Ошибка проверки шрифта '{font_path}': {e}")
        return False

def remove_fonts_without_cyrillic_support(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if file_name.lower().endswith('.ttf') or file_name.lower().endswith('.otf'):
                if not has_cyrillic_support(file_path) or has_empty_russian_A(file_path):
                    print(f"Удаление шрифта без поддержки кириллицы: {file_path}")
                    os.remove(file_path)

# Пример использования:
folder_path = "dafonts-free-v1"

remove_fonts_without_cyrillic_support(folder_path)
