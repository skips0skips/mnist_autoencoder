
import random
import torch
import torch.nn.functional as F

from autoencoder_model.model import CVAE
from ..model import CVAE
#python -m autoencoder_model.test.test_model

def test_encode():
    '''Тестирование кодировщика (encode)'''
    model = CVAE()
    x = torch.randn(32, 1, 28, 28)  # Пример входных данных
    class_num = torch.tensor([random.randint(0, 9) for _ in range(32)])# Пример номеров классов
    mu, logvar, label = model.encode(x, class_num)
    
    # Проверяем размерность выходных тензоров
    assert mu.size() == (32, 4)
    assert logvar.size() == (32, 4)
    assert label.size() == (32, 10)
    
    # Проверяем значения выходных тензоров
    assert torch.all(torch.eq(label, F.one_hot(class_num, num_classes=10)))

def test_forward():
    '''Тестирование метода forward'''
    model = CVAE()
    x = torch.randn(32, 1, 28, 28)  # Пример входных данных
    class_num = torch.tensor([random.randint(0, 9) for _ in range(32)])# Пример номеров классов
    mu, logvar, reconstruction = model(x, class_num)
    
    # Проверяем размерность выходных тензоров
    assert mu.size() == (32, 4)
    assert logvar.size() == (32, 4)
    assert reconstruction.size() == (32, 1, 28, 28)
    
    # Проверяем значения выходного тензора
    assert torch.all(torch.ge(reconstruction, 0.0))  # Значения неотрицательные
    assert torch.all(torch.le(reconstruction, 1.0))  # Значения не превышают 1.0

def test_get_latent_var():
    '''Тестирование метода get_latent_var'''
    model = CVAE()
    x = torch.randn(32, 1, 28, 28)  # Пример входных данных
    class_num = torch.tensor([random.randint(0, 9) for _ in range(32)])# Пример номеров классов
    z = model.get_latent_var(x, class_num)
    
    # Проверяем размерность выходного тензора
    assert z.size() == (32, 4)
    
    # Проверяем значения выходного тензора
    assert torch.all(torch.isfinite(z))  # Значения не являются NaN или бесконечностью

def test_get_sample_var():
    '''Тестирование метода get_sample_var'''
    model = CVAE()
    z = torch.randn(32, 4)  # Пример латентного вектора
    class_num = torch.tensor([random.randint(0, 9) for _ in range(32)])# Пример номеров классов
    sample = model.get_sample_var(z, class_num)
    
    # Проверяем размерность выходного тензора
    assert sample.size() == (32, 1, 28, 28)
    
    # Проверяем значения выходного тензора
    assert torch.all(torch.ge(sample, 0.0))  # Значения неотрицательные
    assert torch.all(torch.le(sample, 1.0))  # Значения не превышают 1.0

if __name__ == '__main__':
    test_encode()
    test_forward()
    test_get_latent_var()
    test_get_sample_var()
