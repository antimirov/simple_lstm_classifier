'''Simple class predictor'''
# coding: utf-8

import argparse
import os
import pickle
import sys

from sklearn.model_selection import train_test_split

# Suppress 'Your CPU supports instructions that this TensorFlow binary...'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#pylint: disable=wrong-import-position
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers import LSTM
from keras.utils import to_categorical
#pylint: enable=wrong-import-position

MAXWORDS = 10000
MAXLEN = 60

NUM_EPOCHS = 25
BATCH_SIZE = 10


def create_tokenizer(max_words, char_level=True):
    pass


def prepare_data(labeled_data_file, sep):

    X = []
    Y = []

    for line in open(labeled_data_file, encoding='utf8'):
        address, country = line.strip().split(sep)[:2]
        X.append(address)
        Y.append(country)

    tokenizer_x = Tokenizer(MAXWORDS, char_level=True)

    tokenizer_x.fit_on_texts(X)
    print('tokenizer_x.word_index:', tokenizer_x.word_index)
    reverse_word_index = {v:k for k, v in tokenizer_x.word_index.items()}

    X_tts_raw = [tokenizer_x.texts_to_sequences(a) for a in X]
    X_tts = []
    for row in X_tts_raw:
        X_tts.append([c[0] for c in row])

    del X

    print('len(X_tts):', len(X_tts))

    tokenizer_y = Tokenizer(MAXWORDS, char_level=False)
    tokenizer_y.fit_on_texts(Y)
    print('tokenizer_y.word_index:', tokenizer_y.word_index)

    num_classes = len(tokenizer_y.word_index.values())
    print('num_classes:', num_classes)

    Y_tts = [tokenizer_y.word_index[a.lower()] for a in Y]
    print('len Y_tts:', len(Y_tts))

    del Y

    X_train, X_test, Y_train, Y_test = train_test_split(X_tts, Y_tts, test_size=0.2)

    Y_train_one_hot_labels = to_categorical(Y_train, num_classes=num_classes+1)

    Y_test_one_hot_labels = to_categorical(Y_test, num_classes=num_classes+1)

    x_train = sequence.pad_sequences(X_train, maxlen=MAXLEN, padding='post', truncating='post')
    x_test = sequence.pad_sequences(X_test, maxlen=MAXLEN, padding='post', truncating='post')
    
    return (
        num_classes, x_train, Y_train_one_hot_labels,
        x_test, Y_test_one_hot_labels,
        tokenizer_x, tokenizer_y
    )

def train(labeled_data_file, num_epochs=NUM_EPOCHS, sep='|'):
    '''Train labeled data'''
    num_classes, x_train, Y_train_one_hot_labels, x_test, Y_test_one_hot_labels, tokenizer_x, tokenizer_y = \
        prepare_data(labeled_data_file, sep)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(MAXWORDS, 128, mask_zero=False))
    model.add(
        LSTM(
            128, input_shape=(MAXWORDS, MAXLEN), dropout=0.2,
            recurrent_dropout=0.2, return_sequences=True
        )
    )
    model.add(
        LSTM(
            128, input_shape=(MAXWORDS, MAXLEN), dropout=0.2,
            recurrent_dropout=0.2
        )
    )
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes+1, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    tb_callback = keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=0, batch_size=BATCH_SIZE,
        write_graph=True, write_grads=False, write_images=False,
        embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='batch'
    )

    print('Train...')
    history = model.fit(
        x_train, Y_train_one_hot_labels,
        batch_size=BATCH_SIZE,
        epochs=num_epochs,
        validation_data=(x_test, Y_test_one_hot_labels),
        callbacks=[tb_callback]
    )

    return model, history, tokenizer_x, tokenizer_y


def predict(lines_to_predict, model, tokenizer_x, tokenizer_y, num_classes=1, print_scores=False, sep='|', min_printable_score=0.001):
    '''Predict classes of the lines of text'''
    
    reverse_word_index_y = {v:k for k, v in tokenizer_y.word_index.items()}

    for line in lines_to_predict:
        line = line.strip()
        line_sequence = [x[0] for x in tokenizer_x.texts_to_sequences(line)]
        line_sequences_padded = sequence.pad_sequences(
            [line_sequence], maxlen=MAXLEN, padding='post', truncating='post'
        )
        prediction = model.predict(line_sequences_padded)[0]

        prediction_indices = []
        for p_idx in range(1, len(prediction)):
            prediction_indices.append((float(prediction[p_idx]), p_idx))
        prediction_indices_sorted = sorted(prediction_indices, reverse=True, key=lambda x: x[0])

        class_scores = []
        for p, p_idx in prediction_indices_sorted[:num_classes]:
            if p >= min_printable_score:
                if print_scores:
                    class_scores.append('{}={:.3f}'.format(reverse_word_index_y[p_idx], p))
                else:
                    class_scores.append(reverse_word_index_y[p_idx])

        print('{}{}{}'.format(line, sep, ','.join(class_scores)))


def save_model(model_file, model, tokenizer_x, tokenizer_y):
    root, _ = os.path.splitext(model_file)
    model.save(model_file)
    with open('{}_tokenizer_x.pickle'.format(root), 'wb') as handle:
        pickle.dump(tokenizer_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}_tokenizer_y.pickle'.format(root), 'wb') as handle:
        pickle.dump(tokenizer_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model_file):
    model = keras.models.load_model(model_file)

    root, _ = os.path.splitext(model_file)
    with open('{}_tokenizer_x.pickle'.format(root), 'rb') as handle:
        tokenizer_x = pickle.load(handle)
    with open('{}_tokenizer_y.pickle'.format(root), 'rb') as handle:
        tokenizer_y = pickle.load(handle)

    return model, tokenizer_x, tokenizer_y


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    required_group = parser.add_argument_group('Required')
    optonal_group = parser.add_argument_group('Optional')

    required_group.add_argument(
        'action', help='Action: train or predict?',
        choices=['train', 'predict']
    )
    required_group.add_argument(
        '-f', '--input-file',
        help='''A file with an input line and a prediction category, separated by SEP (--separator) symbol. If '-' then stdin is used''',
        required=True
    )
    required_group.add_argument(
        '-m', '--model-file',
        help='Model file. Overwriten after training or loaded for prediction. Class file will be stored along as modelname.cls',
        required=True
    )
    optonal_group.add_argument(
        '--num-epochs',
        help='Number of training epochs',
        type=int,
        default=NUM_EPOCHS
    )
    optonal_group.add_argument(
        '--num-classes',
        help='Print number of classes for each input line',
        type=int,
        default=1
    )
    optonal_group.add_argument(
        '--print-scores',
        help=r'Print scores for each class in such form: CLASS1=95.3%%,CLASS2=3.2%%,etc...',
        action='store_true'
    )
    parser.add_argument(
        '--separator',
        help='Output separator symbol: orig_lin<SEP><class1=0.98><class2=0.01>...',
        default='|'
    )
    optonal_group.add_argument(
        '--min-printable-score',
        help='''Don't print the class and its score if the probability is lower than this limit''',
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    if args.action == 'train':
        model, history, tokenizer_x, tokenizer_y = train(
            args.input_file, args.num_epochs, args.separator
        )
        save_model(args.model_file, model, tokenizer_x, tokenizer_y)
    elif args.action == 'predict':
        model, tokenizer_x, tokenizer_y = load_model(args.model_file)
        in_file = sys.stdin if args.input_file == '-' else open(args.input_file)
        predict(
            in_file,
            model,
            tokenizer_x,
            tokenizer_y,
            args.num_classes,
            args.print_scores,
            args.separator,
            args.min_printable_score
        )
    else:
        pass

if __name__ == '__main__':
    main()
