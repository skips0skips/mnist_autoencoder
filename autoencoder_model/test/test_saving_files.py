import os
import pickle
import unittest
from pathlib import Path
import sys

from ..file_handler import FileHandler

class FileHandlerTests(unittest.TestCase):
    folder_path = str(Path('autoencoder_model', 'file'))

    def test_save_and_load_file(self):
        data = [1, 2, 3, 4, 5]
        filename = 'data.pkl'

        # Сохраняем файл
        FileHandler.save_file(data, filename)

        # Проверяем, что файл был сохранен
        file_path = os.path.join(self.folder_path, filename)
        self.assertTrue(os.path.exists(file_path))

        # Загружаем файл
        loaded_data = FileHandler.load_file(filename)

        # Проверяем, что загруженные данные совпадают с исходными данными
        self.assertEqual(data, loaded_data)

        
        # Удаляем файл после проверки
        os.remove(file_path)


if __name__ == '__main__':
    unittest.main()
