import os
import numpy as np

from ..config import Config

from ..data_preparation import Data

from ..model import CVAE

from ..train_pipeline import Trainer

# Тестирование метода train
def test_train():
    # Создаем фиктивные данные
    model, criterion, optimizer, device = Config.get_config_model()
    train_loader, test_loader = Data.data_loader()
    output_images_bool = False
    val_fit_bool = False
    n_epochs = 2

    # Создаем объект Trainer
    trainer = Trainer(model, criterion, optimizer, device, test_loader, train_loader)

    # Проверяем, что метод train возвращает модель
    trained_model = trainer.train(output_images_bool, val_fit_bool, n_epochs)
    assert isinstance(trained_model, CVAE), "Метод train не возвращает модель"

    # Проверяем, что train_losses и val_losses не пустые списки после обучения
    assert trainer.train_losses, "Список train_losses пустой"
    assert trainer.val_losses, "Список val_losses пустой"

    # Проверяем, что значения train_losses и val_losses являются списками numpy
    assert isinstance(trainer.train_losses[0], np.float64), "Значения в списке train_losses не являются типом np.float64"
    assert isinstance(trainer.val_losses[0], np.float64), "Значения в списке val_losses не являются типом np.float64"

    # Проверяем, что модель была сохранена в файл
    assert os.path.exists('self.model'), "Файл 'self.model' не найден"

    # Проверяем, что train_losses был сохранен в файл
    assert os.path.exists('train_losses'), "Файл 'train_losses' не найден"

if __name__ == '__main__':
    test_train()
