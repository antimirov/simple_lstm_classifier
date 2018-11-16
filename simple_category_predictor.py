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
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
#pylint: enable=wrong-import-position

MAXWORDS = 10000
MAXLEN = 60

NUM_EPOCHS = 25
BATCH_SIZE = 10


def prepare_data(labeled_data_file, sep):
    '''Read sep-separated data from the file, create train/test split, tokenize, categorize.'''

    x_raw = []
    y_raw = []

    for line in open(labeled_data_file, encoding='utf8'):
        address, country = line.strip().split(sep)[:2]
        x_raw.append(address)
        y_raw.append(country)

    tokenizer_x = Tokenizer(MAXWORDS, char_level=True)

    tokenizer_x.fit_on_texts(x_raw)
    print('tokenizer_x.word_index:', tokenizer_x.word_index)
    #reverse_word_index = {v:k for k, v in tokenizer_x.word_index.items()}

    x_tts_raw = [tokenizer_x.texts_to_sequences(a) for a in x_raw]
    x_tts = []
    for row in x_tts_raw:
        x_tts.append([c[0] for c in row])

    del x_raw

    print('len(x_tts):', len(x_tts))

    tokenizer_y = Tokenizer(MAXWORDS, char_level=False)
    tokenizer_y.fit_on_texts(y_raw)
    print('tokenizer_y.word_index:', tokenizer_y.word_index)

    num_classes = len(tokenizer_y.word_index.values())
    print('num_classes:', num_classes)

    y_tts = [tokenizer_y.word_index[a.lower()] for a in y_raw]
    print('len y_tts:', len(y_tts))

    del y_raw

    x_train, x_test, y_train, y_test = train_test_split(x_tts, y_tts, test_size=0.2)

    y_train_one_hot_labels = to_categorical(y_train, num_classes=num_classes+1)

    y_test_one_hot_labels = to_categorical(y_test, num_classes=num_classes+1)

    x_train = sequence.pad_sequences(x_train, maxlen=MAXLEN, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=MAXLEN, padding='post', truncating='post')

    return (
        num_classes, x_train, y_train_one_hot_labels,
        x_test, y_test_one_hot_labels,
        tokenizer_x, tokenizer_y
    )

def train(labeled_data_file, num_epochs=NUM_EPOCHS, sep='|'):
    '''Train labeled data'''
    num_classes, x_train, y_train_one_hot_labels, x_test, y_test_one_hot_labels,\
        tokenizer_x, tokenizer_y = prepare_data(labeled_data_file, sep)

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
        log_dir='./logz', histogram_freq=0, batch_size=BATCH_SIZE,
        write_graph=True, write_grads=False, write_images=False,
        embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='batch'
    )

    print('Train...')
    history = model.fit(
        x_train, y_train_one_hot_labels,
        batch_size=BATCH_SIZE,
        epochs=num_epochs,
        validation_data=(x_test, y_test_one_hot_labels),
        callbacks=[tb_callback]
    )

    return model, history, tokenizer_x, tokenizer_y


def predict(lines_to_predict, model, tokenizer_x, tokenizer_y, num_classes=1, print_scores=False, sep='|', min_printable_score=0.001):
    '''Predict classes of the lines of text'''

    reverse_word_index_y = {v:k for k, v in tokenizer_y.word_index.items()}

    for line in lines_to_predict:
        line = line.strip()
        # TODO implement as fallback when a character is not found, e.g. convert to ASCII.
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
        for pred, pred_idx in prediction_indices_sorted[:num_classes]:
            if pred >= min_printable_score:
                if print_scores:
                    class_scores.append('{}={:.3f}'.format(reverse_word_index_y[pred_idx], pred))
                else:
                    class_scores.append(reverse_word_index_y[pred_idx])

        print('{}{}{}'.format(line, sep, ','.join(class_scores)))


def save_model(model_file, model, tokenizer_x, tokenizer_y):
    '''Save model file plus x and y tokenizer data in separate files'''
    root, _ = os.path.splitext(model_file)
    model.save(model_file)
    with open('{}_tokenizer_x.pickle'.format(root), 'wb') as handle:
        pickle.dump(tokenizer_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}_tokenizer_y.pickle'.format(root), 'wb') as handle:
        pickle.dump(tokenizer_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model_file):
    '''Load model file plus x and y tokenizer data from separate files'''
    model = keras.models.load_model(model_file)

    root, _ = os.path.splitext(model_file)
    with open('{}_tokenizer_x.pickle'.format(root), 'rb') as handle:
        tokenizer_x = pickle.load(handle)
    with open('{}_tokenizer_y.pickle'.format(root), 'rb') as handle:
        tokenizer_y = pickle.load(handle)

    return model, tokenizer_x, tokenizer_y


def main():
    '''Main function: parses argiments, calls train or predict'''
    parser = argparse.ArgumentParser(
        description='A simple tool to train a character level classifier and predict a class based on an input string. Useful for tasks like detecting country by address line, nationality by name, etc. ',
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
