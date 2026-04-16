import gradio as gr
import shutil
import zipfile
from pathlib import Path
from fastai.vision.all import *

learn = load_learner('foto_giuli.pkl')
classi = ['defocused_blurred', 'motion_blurred', 'sharp']

def classifica_foto(zip_input):
    # Estrai lo zip in input
    input_dir = Path('/tmp/input_foto')
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir()

    with zipfile.ZipFile(zip_input, 'r') as z:
        z.extractall(input_dir)

    # Crea cartelle output
    output = Path('/tmp/output')
    if output.exists():
        shutil.rmtree(output)
    for classe in classi:
        (output / classe).mkdir(parents=True, exist_ok=True)

    # Classifica ogni foto
    estensioni = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    for img_path in input_dir.rglob('*'):
        if img_path.suffix in estensioni:
            pred, idx, probs = learn.predict(PILImage.create(img_path))
            shutil.copy(img_path, output / pred / img_path.name)

    # Crea zip output
    zip_path = '/tmp/classificate.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in output.rglob('*'):
            if f.is_file():
                zipf.write(f, f.relative_to(output))

    return zip_path

demo = gr.Interface(
    fn=classifica_foto,
    inputs=gr.File(label='Carica lo zip con le tue foto'),
    outputs=gr.File(label='Scarica le foto classificate'),
    title='Classificatore foto',
    description='Carica uno .zip con le tue foto e scarica lo zip con le cartelle sharp, defocused_blurred e motion_blurred.'
)

demo.launch()