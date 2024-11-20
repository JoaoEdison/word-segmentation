## About

This program performs the digitization of handwritten texts. It features a graphical user interface (GUI) built with [Raylib](https://github.com/raysan5/raylib), allowing users to configure the main parameters used in segmentation and see the result.

The words are segmented from the main text using an image processing approach. They are then digitized through a neural network that combines a convolutional neural network with a recurrent neural network with long short-term memory. This model is not my creation; it is based on the tutorials: [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr) and [Japanese OCR with the CTC Loss](https://medium.com/@natsunoyuki/ocr-with-the-ctc-loss-efa62ebd8625).

The dataset used for training the neural network comes from the [ICDAR 2024 COMPETITION ON RECOGNITION AND VQA ON HANDWRITTEN DOCUMENTS](https://ilocr.iiit.ac.in/icdar_2024_hwd/index.html), Task A in English.

## Usage

This program was implemented in Python 3.10.15.

Install dependencies:
```console
pip install -r requirements.txt
```

Execute:
```console
python segmenter.py
```

### Advice

The neural network is not trained and tuned properly at present.

### License

MIT.

## Sobre

Este programa faz a digitalização de textos manuscritos. Possui um interface gráfica de usuário (GUI) feita em [Raylib](https://github.com/raysan5/raylib), na qual é possível configurar os parâmetros principais usados na segmentação e acompanhar o resultado final.

As palavras são segmentadas do texto principal usando-se uma abordagem de processamento de imagens. Depois, são digitalizadas por meio de uma rede neural formada pela combinação de uma rede neural convolucional e uma rede recorrente com memória de curto longo prazo. Este modelo não é de minha autoria; ele é baseado nos tutoriais: [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr) e [Japanese OCR with the CTC Loss](https://medium.com/@natsunoyuki/ocr-with-the-ctc-loss-efa62ebd8625).

O dataset utilizado para o treinamento da rede neural proveio da [ICDAR 2024 COMPETITION ON RECOGNITION AND VQA ON HANDWRITTEN DOCUMENTS](https://ilocr.iiit.ac.in/icdar_2024_hwd/index.html), Task A em Inglês.

## Uso

Este programa foi implementado em Python 3.10.15.

Instale as dependências:
```console
pip install -r requirements.txt
```

Execute:
```console
python segmenter.py
```

### Aviso

A rede neural não está treinada e ajustada adequadamente até o presente momento.

### Licença

MIT.
