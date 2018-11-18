# Simple LSTM classifier

A simple tool to train a character or word level classifier and predict a class based on an input string. Useful for tasks like detecting country by address line, nationality by full name, programming language by a few lines of code, etc. Prediction is printed into stdout.

## Usage

### Training

$ python simple_lstm_classifier.py train -f data/male_female_names_corpus/male_female.txt -m male_female.h5 --num-epochs=1
Using TensorFlow backend.
Number of samples: 7943
tokenizer_x.word_index: {'a': 1, 'e': 2, 'i': 3, 'n': 4, 'r': 5, 'l': 6, 'o': 7, 't': 8, 's': 9, 'd': 10, 'm': 11, 'y': 12, 'h': 13, 'c': 14, 'b': 15, 'u': 16, 'g': 17, 'k': 18, 'j': 19, 'v': 20, 'f': 21, 'p': 22, 'w': 23, 'z': 24, 'x': 25, 'q': 26, '-': 27, ' ': 28, "'": 29}
tokenizer_y.word_index: {'female': 1, 'male': 2}
Number of classes: 2
Build model...
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 128)         3840      
_________________________________________________________________
lstm_1 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 128)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 387       
=================================================================
Total params: 168,835
Trainable params: 168,835
Non-trainable params: 0
_________________________________________________________________
Train...
Train on 6354 samples, validate on 1589 samples
... some time later...
Epoch 50/50
6354/6354 [==============================] - 31s 5ms/step - loss: 0.0356 - acc: 0.9104 - val_loss: 0.0788 - val_acc: 0.8991

## Prediction:

$ echo -e "Anna Karenina" | python simple_lstm_classifier.py predict -f - -m male_female.h5 
Using TensorFlow backend.
Anna Karenina|female
