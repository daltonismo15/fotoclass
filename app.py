import gradio as gr
import shutil
import zipfile
from pathlib import Path
from fastai.vision.all import *

# Carica il modello
learn = load_learner('foto_giuli.pkl')
classi = ['defocused_blurred', 'motion_blurred', 'sharp']

def classifica_foto(files):
    # Crea cartelle output
    output = Path('output')
    if output.exists():
        shutil.rmtree(output)
    for classe in classi:
        (output/classe).mkdir(parents=True, exist_ok=True)

    # Classifica ogni foto
    for file in files:
        img_path = Path(file)  # <-- cambiato da file.name a file
        pred, idx, probs = learn.predict(PILImage.create(img_path))
        shutil.copy(img_path, output/pred/img_path.name)

    # Crea zip
    zip_path = '/tmp/classificate.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in output.rglob('*'):
            if f.is_file():
                zipf.write(f, f.relative_to(output))

    return zip_path

demo = gr.Interface(
    fn=classifica_foto,
    inputs=gr.File(file_count='multiple', label='Carica le tue foto'),
    outputs=gr.File(label='Scarica le foto classificate'),
    title='Classificatore foto',
    description='Carica le tue foto e scarica lo zip con le cartelle sharp, defocused_blurred e motion_blurred.'
)

demo.launch()