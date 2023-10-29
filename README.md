# NLP Poem Generator

This repository contains a Python script for generating poems using Natural Language Processing (NLP) techniques. The code is implemented in a Jupyter notebook named `NLP_Poem.ipynb`, originally created in Google Colaboratory.

## Prerequisites

Make sure you have the following dependencies installed:

- TensorFlow
- gdown
- NumPy
- Matplotlib

You can install the required Python packages using the following:

```bash
pip install tensorflow gdown numpy matplotlib

Its made using tensorflow, so if you don't have it, install it using the code below
```
!pip install tensorflow
```

[Find the text used as the corpus here](https://drive.google.com/uc?id=15UqmiIm0xwh9mt0IYq2z3jHaauxQSTQT)

```

### Model Architecture

The NLP model architecture includes layers such as Embedding, Bidirectional LSTM, Bidirectional GRU, and Dense layers. Different configurations are commented out in the code, and you can experiment with them to observe different results.

```
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Bidirectional(GRU(64,return_sequences = True)))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(GRU(32)))
model.add(Dense(total_words, activation = 'leaky_relu'))
model.add(Dense(total_words,activation = 'sigmoid'))
adam = Adam(lr = 0.005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
```

*If you will un comment the code the number of parameters will rise to about 35Mn.*

![model image](https://github.com/sanidhaya/NLP-to-write-poem/blob/main/Capture1.PNG)
