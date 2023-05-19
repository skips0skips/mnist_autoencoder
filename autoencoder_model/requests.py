import requests

# Пример запроса на запуск получения изображений
number = 5
response = requests.post('http://localhost:5000/start/{}'.format(number), headers={'Authorization': 'admin'})
if response.status_code == 200:
    image_data = response.json()
    # Обработка полученных данных
    print(image_data)
else:
    print('Ошибка при запросе:', response.text)

# Пример запроса на получение текущей конфигурации
response = requests.get('http://localhost:5000/get_config', headers={'Authorization': 'admin'})
if response.status_code == 200:
    config_data = response.json()
    # Обработка полученных данных
    print(config_data)
else:
    print('Ошибка при запросе:', response.text)

# Пример запроса на изменение конфигурации
new_config = {
    'train_fit_bool': True,
    'val_fit_bool': False,
    'n_epochs': 10,
    'output_images_bool': True
}
response = requests.post('http://localhost:5000/set_config', json=new_config, headers={'Authorization': 'admin'})
if response.status_code == 200:
    print('Конфигурация успешно изменена')
else:
    print('Ошибка при запросе:', response.text)

# Пример запроса на получение начальной фразы
response = requests.get('http://localhost:5000/set_initial_phrase', headers={'Authorization': 'admin'})
if response.status_code == 200:
    initial_phrase = response.json()['initial_phrase']
    # Обработка полученной фразы
    print(initial_phrase)
else:
    print('Ошибка при запросе:', response.text)
