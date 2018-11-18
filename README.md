# Simple LSTM classifier

A simple tool to train a character- or word-level classifier and predict a class based on an input string. Useful for tasks like detecting country by address line, nationality by full name, programming language by a few lines of code, etc. Prediction is printed into stdout.

## Usage

### Training

```
$ python simple_lstm_classifier.py train \
  -f data/male_female_names_corpus/male_female.txt -m male_female.h5 --num-epochs=50
Number of samples: 7943
Train...
Train on 6354 samples, validate on 1589 samples
... some time later...
Epoch 50/50
6354/6354 [==============================] - 31s 5ms/step - loss: 0.0356 - acc: 0.9104
                                                  - val_loss: 0.0788 - val_acc: 0.8991
```

### Prediction:

```
$ echo -e "Anna Karenina" | python simple_lstm_classifier.py predict \
  -f - -m male_female.h5 
Anna Karenina|female
```
