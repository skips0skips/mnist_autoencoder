import os
import pickle
import unittest
from pathlib import Path

from autoencoder_model.saving_files import save


class TestSave(unittest.TestCase):

    def test_save(self):
        '''
        В этом тесте мы проверяем:
        Что файл сохраняется в ожидаемой папке.
        Что сохраненные данные соответствуют исходным данным.
        Удаляем созданный файл после выполнения теста (очистка).
          '''
        name = 'test_name'
        file_name = 'Vanilla.model'
        folder_path = 'my_folder'
        expected_path = os.path.join(folder_path, file_name)

        # Save the file
        save(name)

        # Check if the file exists
        self.assertTrue(os.path.exists(expected_path))

        # Load the saved data
        with open(expected_path, 'rb') as file:
            loaded_name = pickle.load(file)

        # Check if the loaded data matches the original data
        self.assertEqual(name, loaded_name)

        # Cleanup: delete the file
        os.remove(expected_path)


if __name__ == '__main__':
    unittest.main()