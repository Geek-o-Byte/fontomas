from fontTools.ttLib import TTFont
import os

# Путь к папке с шрифтами
fonts_folder = "fonts"

# Путь к файлу, в который будем записывать имена шрифтов
output_file = "big_names.txt"

# Открываем файл для записи имен шрифтов
with open(output_file, 'w', encoding='utf-8') as f:
    # Проходим по всем файлам в папке с шрифтами
    for font_file in os.listdir(fonts_folder):
        # Проверяем, является ли файл файлом шрифта (ttf или otf)
        if font_file.lower().endswith(('.ttf', '.otf')):
            # Формируем полный путь к файлу шрифта
            font_path = os.path.join(fonts_folder, font_file)
            try:
                # Открываем файл шрифта с помощью fontTools
                font = TTFont(font_path)
                # Получаем имя шрифта и записываем его в файл
                font_name = font['name'].getName(1, 3, 1).toStr()
                f.write(font_name + '\n')
                print(f"Имя шрифта '{font_name}' записано в файл.")
            except Exception as e:
                print(f"Ошибка при обработке файла '{font_file}': {e}")

print("Процесс завершен. Имена шрифтов записаны в файл:", output_file)
