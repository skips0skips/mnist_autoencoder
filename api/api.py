import base64
from autoencoder_model.main import Main
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import io
from torchvision.transforms.functional import to_pil_image

from autoencoder_model.model import CVAE

app = FastAPI()
main = Main()

@app.get("/")
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)



@app.get("/start/")
def start_form():
    """HTML form for entering a number."""
    form = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Enter a Number</h1>"
        "<form action='/start/' method='post'>"
        "<input type='number' name='number'>"
        "<input type='submit' value='Сгенерировать'>"
        "</form>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=form)

@app.post("/start/")
async def start_process(request: Request, number: int = Form(...)):
    """Process the submitted form."""
    
    main.start(number)
    return {"message": "Запущено получение изображений"}


@app.get("/get_image/{number}")
def get_image(number: int):
    images = main.get_image(number)
    images_html = ""

    for i, image in enumerate(images):
        pil_image = to_pil_image(image)
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            image_html = f'<img src="data:image/png;base64,{image_base64}" alt="Image {i}">'
            images_html += image_html

    html_content = f"""
    <html>
    <head>
        <title>Вывод изображений</title>
    </head>
    <body>
        {images_html}
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)



@app.get("/get_config", response_class=HTMLResponse)
def get_config():
    optimizer, device, output_images_bool, val_fit_bool, n_epochs = main.get_config()
    html_content = """
    <html>
        <head>
            <title>Configuration</title>
        </head>
        <body>
            <h1>Configuration</h1>
            <p>Optimizer: {optimizer}</p>
            <p>Device: {device}</p>
            <p>Output Images Bool: {output_images_bool}</p>
            <p>Validation Fit Bool: {val_fit_bool}</p>
            <p>Number of Epochs: {n_epochs}</p>
        </body>
    </html>
    """
    return html_content.format(
        optimizer=optimizer,
        device=device,
        output_images_bool=output_images_bool,
        val_fit_bool=val_fit_bool,
        n_epochs=n_epochs
    )

@app.put("/set_config")
def set_config(train_fit_bool: bool = None, val_fit_bool: bool = None, n_epochs: int = None, output_images_bool: bool = None):
    main.set_config(train_fit_bool, val_fit_bool, n_epochs, output_images_bool)
    return {"message": "Параметры конфигурации изменены"}


@app.get("/set_initial_phrase")
def set_initial_phrase():
    initial_phrase = main.set_initial_phrase()
    return {"initial_phrase": initial_phrase}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")
