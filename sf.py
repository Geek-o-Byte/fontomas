import os

def write_filenames_to_txt(folder_path, output_txt):
    # Открываем файл для записи
    with open(output_txt, 'w', encoding="utf-8") as txtfile:
        # Проходим по всем файлам в указанной папке
        for filename in os.listdir(folder_path):
            # Проверяем, является ли текущий элемент файлом
            if os.path.isfile(os.path.join(folder_path, filename)):
                # Записываем имя файла в текстовый файл
                txtfile.write(filename + '\n')

# Путь к папке, которую нужно просканировать
folder_path = 'all_fonts'
# Имя для текстового файла, в который будут записаны имена файлов
output_txt = 'all_fonts.txt'

# Вызываем функцию для записи имен файлов в текстовый файл
write_filenames_to_txt(folder_path, output_txt)
