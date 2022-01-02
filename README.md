# NLP-to-write-poem
Created a model to NLP model to write something beautiful, some poem maybe, covering up for my poor languages skills.  

Its made using tensorflow, so if you don't have it, install it using the code below
```
!pip install tensorflow
```

[Find the text used as the corpus here](https://drive.google.com/uc?id=15UqmiIm0xwh9mt0IYq2z3jHaauxQSTQT)

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
