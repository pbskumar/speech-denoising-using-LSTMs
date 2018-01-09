import re
import glob
import pickle
import os
import shutil

import numpy as np

import scipy.io as sio
from scipy.io import wavfile as swav

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import load_model

import speech_preprocessing


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def generate_model_file_name(name_dict):
    """

    :param name_dict:
    :return:
    """
    key_elements = ['architecture', 'mask', 'gender', 'lookback',
                    'stateful', 'iter', 'batch', 'learnrate']
    file_name = []

    for el in key_elements:
        if el in name_dict and name_dict[el] is not None:
            file_name.append(el + '_' + str(name_dict[el]))

    return '_'.join(file_name)


def generate_sequence_data(input_data, target_data=None, lookback=40):

    """
    Return length is n-1 for n samples
    :param input_data:
    :param target_data:
    :param lookback:
    :return:
    """
    num_samples = input_data.shape[0]
    num_sequences = num_samples - lookback
    _X = []

    # For offsetting first lookback frames by repeating fist frame
    # 0,0,0,0 -> 1; 0,0,0,1 -> 2;
    # 0,0,1,2 -> 3 : lookback -1 frames for given lookback = 4
    _head_frames = np.repeat(input_data[0], lookback).reshape((-1, lookback)).T
    for i in range(1, lookback):
        _temp = []
        _head_frames = _head_frames[1:]
        _temp.extend(_head_frames)
        _temp.extend(input_data[:i])
        _X.append(np.array(_temp))

    for i in range(num_sequences):
        _X.append(input_data[i:i + lookback])

    # Generating sequence target only when target data is provided
    # In train phase
    if target_data is not None:
        _Y = target_data[1:num_samples]
        return np.array(_X), _Y
    else:
        return np.array(_X)


def model0(_model_params):
    # reading model parameters
    print('Reading model parameters')
    is_stateful = _model_params['stateful']
    timesteps = _model_params['lookback']
    # for 1024 window, frame size = 513
    input_dim = _model_params['frame_size'] // 2 + 1

    # creating a sequential model
    print('Creating sequential model')

    _model = Sequential()

    _model.add(
        LSTM(256, activation='tanh', input_shape=(timesteps, input_dim), return_sequences=True, stateful=is_stateful))
    _model.add(LSTM(512, activation='tanh', stateful=is_stateful))
    _model.add(Dense(1024, activation='tanh'))
    _model.add(Dropout(0.2))
    _model.add(Dense(2048, activation='tanh'))
    _model.add(Dropout(0.2))
    _model.add(Dense(1024, activation='tanh'))
    _model.add(Dropout(0.2))
    _model.add(Dense(513, activation='sigmoid'))

    print('Compiling model')
    _model.compile(loss='binary_crossentropy', metrics=['accuracy', 'mae'], optimizer='adam')

    return _model


def model1(_model_params):
    # reading model parameters
    print('Reading model parameters')
    is_stateful = _model_params['stateful']
    timesteps = _model_params['lookback']
    # for 1024 window, frame size = 513
    input_dim = _model_params['frame_size'] // 2 + 1

    # creating a sequential model
    print('Creating sequential model')

    _model = Sequential()

    _model.add(
        LSTM(1024, activation='tanh', input_shape=(timesteps, input_dim), return_sequences=True, stateful=is_stateful))
    _model.add(LSTM(2048, activation='tanh', return_sequences=True, stateful=is_stateful))
    _model.add(Dropout(0.2))
    _model.add(LSTM(1024, activation='tanh', stateful=is_stateful))
    _model.add(Dropout(0.2))
    _model.add(Dense(2048, activation='tanh'))
    _model.add(Dropout(0.2))
    _model.add(Dense(1024, activation='tanh'))
    _model.add(Dropout(0.2))
    _model.add(Dense(513, activation='sigmoid'))

    print('Compiling model')
    _model.compile(loss='binary_crossentropy', metrics=['accuracy', 'mae'], optimizer='adam')

    return _model


def train_model_batch(_signal_class, _train_data_path, _model_params, _model_file_name):
    # list of all types of models available
    _model_list = [model0, model1]

    print('Initializing selection filters')
    _target_mask = _model_params['mask']
    _gender_filter = _model_params['gender']
    _noise_filter = _model_params['noise']
    _scale_filter = _model_params['scale']

    # Training parameters
    _lookback = _model_params['lookback']
    _num_epochs = _model_params['iter']
    _batch_size = _model_params['batch']
    _architecture = _model_params['architecture']

    print('Fetching all relevant training files files\n')
    _train_file_paths_list = speech_preprocessing.get_filtered_sound_file_names_list(glob.glob(_train_data_path + '*'),
                                                                                     _gender=_gender_filter,
                                                                                     _noise=_noise_filter,
                                                                                     _scale=_scale_filter)
    # total number of training files
    f_len = len(_train_file_paths_list)

    if f_len == 0:
        print('No train files found')
        return

    # learning model
    print('Model learning initiated')
    _model = _model_list[_architecture](_model_params)

    print('Model fit phase started\n')
    try:
        # in each epoch read files one by one and train on batch
        for epoch in range(_num_epochs):
            print('Epoch: ', epoch + 1, '/', _num_epochs, sep='')

            # counter for files processed
            f_num = 0

            # shuffle paths

            # variable to hold loss and other metrics for each epoch
            losses = []

            for data_file in _train_file_paths_list:

                f_num += 1

                data_dict = pickle.load(file=open(data_file, 'rb'))
                _X_list = data_dict['X']
                _S_list = data_dict['S']
                _N_list = data_dict['N']

                num_spectrograms = len(_X_list)

                # shuffle spectrograms

                for i in range(num_spectrograms):
                    print('\rEpoch: {a:>4d} of {b:<4d} :: Processing file: {c:>3d} of {d:<3d} '
                          ':: Spectrogram: {e:>4d} of {f:<4d}'.format(a=epoch + 1, b=_num_epochs, c=f_num,
                                                                      d=f_len, e=i + 1, f=num_spectrograms), end='\r')
                    # print('\rProcessing file: {:0>4d} of {0:4d} :: Spectrogram: {:0>8d} of {:>8d}'
                    #       .format(f_num, f_len, i + 1, num_spectrograms), end='r')
                    # print('\rProcessing file:', f_num, 'of', f_len, ':: Spectrogram:', i+1, 'of', num_spectrograms, end='\r')

                    if _target_mask == 'IBM':
                        _Y = (np.absolute(_S_list[i]) > np.absolute(_N_list[i])).astype(int).T
                    else:
                        # Implemented only for IBM
                        _Y = []

                    _X_train, _Y_train = generate_sequence_data(np.absolute(_X_list[i].T), _Y, _lookback)

                    # training on batch of each file
                    losses = _model.train_on_batch(_X_train, _Y_train)

                    # After each file
                    # print()
            # After each epoch
            print(' ' * 80, end='\r')
            print('\rLoss:', losses[0], ':: Accuracy:', losses[1], ':: Mean abs. error:', losses[2])
            print()

    except KeyboardInterrupt:
        print('Training Interrupted')

    except Exception as e:
        # for all other exceptions raise the error
        raise e

    finally:
        # print(_model.summary())

        # writing model to file
        print('Writing model to file')
        _model.save(_model_file_name)
        print('Write to file:', _model_file_name.split('/')[-1], 'complete')

    print('Exiting train function')


def train_model(_signal_class, _train_data_path, _model_params, _model_file_name):
    """

    :param _signal_class:
    :param _train_data_path:
    :param _model_params:
    :param _model_file_name:
    """

    print('Initializing selection filters')
    _target_mask = _model_params['mask']
    _gender_filter = _model_params['gender']
    _noise_filter = _model_params['noise']
    _scale_filter = _model_params['scale']

    # Training parameters
    num_epochs = _model_params['iter']
    batch_size = _model_params['batch']

    print('Reading spectrograms')
    _X_train_complex, _Y_train_complex, _ = _signal_class.read_processed_data(_train_data_path, _target_mask,
                                                                              _gender_filter, _noise_filter,
                                                                              _scale_filter)

    # Taking absolute values and transposing data
    _X_train_abs = np.absolute(_X_train_complex).T
    _Y_train_abs = np.absolute(_Y_train_complex).T

    # creating sequences
    print('Generating time sequence data')
    _X_train, _Y_train = generate_sequence_data(_X_train_abs, _Y_train_abs, _model_params['lookback'])

    # list of all types of models available
    _model_list = [model0]

    # learning model
    print('Model learning initiated')
    _model = _model_list[_model_params['architecture']](_model_params)

    print('Model fit phase started')
    try:
        _model.fit(_X_train, _Y_train, epochs=num_epochs, batch_size=batch_size, shuffle=False, verbose=2,
                   validation_split=0.2)
        # Problem with stateful LSTMs
        # ValueError: If a RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors:
        # 1. If using a Sequential model, specify the batch size by passing a `batch_input_shape` argument to your first layer.
        # 2. If using the functional API, specify the time dimension by passing a `batch_shape` argument to your Input layer.

        # for i in range(num_epochs):
        #     print('Iteration: ', i + 1, '/', num_epochs, sep='')
        #     # , batch_size=batch_size
        #     _model.fit(_input, target, epochs=1, shuffle=False, verbose=2, validation_split=0.2)
        #
        #     # Reset memory after each epoch if the model is stateful
        #     if is_stateful:
        #         _model.reset_states()

    except KeyboardInterrupt:
        print('Training Interrupted')

    finally:
        print(_model.output)
        # writing model to file
        print('Writing model to file')
        _model.save(_model_file_name)
        print('Write to file:', _model_file_name.split('/')[-1], 'complete')

    print('Exiting train function')


def test_model(_signal_class, _test_data_path, _result_path, _model_params, _model_file_name):
    print('Initializing test filters and relevant parameters')
    _target_mask = _model_params['mask']
    _gender_filter = _model_params['gender']
    _noise_filter = _model_params['noise']
    _scale_filter = _model_params['scale']
    _lookback = _model_params['lookback']

    print('Loading model for test phase')
    _model = load_model(_model_file_name)
    print('Model load complete.\n')

    # Creating directories
    _model_name = (_model_file_name.split('/')[-1]).split('.')[0]
    print('Creating results folder for the model:', _model_name)
    _model_result_path = _result_path + '/' + _model_name + '/'
    os.makedirs(os.path.dirname(_model_result_path), exist_ok=True)
    print('Results directory created\n')

    print('Fetching all relevant test files\n')
    _test_file_paths_list = speech_preprocessing.get_filtered_sound_file_names_list(glob.glob(_test_data_path + '*'),
                                                                                    _gender=_gender_filter,
                                                                                    _noise=_noise_filter,
                                                                                    _scale=_scale_filter)

    # counting the total number of files
    _num_test_files = len(_test_file_paths_list)

    f_num = 1
    for _test_file_path in _test_file_paths_list:

        print('Processing file:', f_num, 'of', _num_test_files, ' :: filename:', _test_file_path)
        f_num += 1

        # Creating directory to store the wave files generated by testing the selected file
        _test_file_name = _test_file_path.split('/')[-1]
        print('Creating directory for ', _test_file_name, ' wave files')
        _wave_file_path = _model_result_path + _test_file_name + '/'
        os.makedirs(os.path.dirname(_wave_file_path), exist_ok=True)

        data_dict = pickle.load(open(_test_file_path, 'rb'))
        X_test_complex_list = data_dict['X']
        S_test_complex_list = data_dict['S']

        print_progress_bar(0, len(X_test_complex_list), prefix='Progress:', suffix='Complete', length=50)
        for num_speech in range(len(X_test_complex_list)):

            # Setting output wave file name
            _wav_filename = _wave_file_path + 's' + str(num_speech)

            # Spectrograms for each sample
            X_test_complex = X_test_complex_list[num_speech]

            # removing first frame because sequence generation removes first frame
            X_test_complex_for_prediction = X_test_complex[:, 1:]
            S_test_complex = S_test_complex_list[num_speech][:, 1:]

            # Generating sequence data with lookback
            X_test_sequence = generate_sequence_data(np.absolute(X_test_complex.T), lookback=_lookback)

            # Preidicting target
            Y_predicted = _model.predict(X_test_sequence)

            # Regenerating speech signal using appropriate technique for a given target
            if _target_mask == 'IBM':
                S_predicted = np.multiply(X_test_complex_for_prediction, np.round(Y_predicted.T))
            elif _target_mask == 'IRM':
                S_predicted = np.multiply(X_test_complex_for_prediction, Y_predicted.T)
            else:
                print("Incorrect target name. Should be 'IRM' or 'IBM' ")
                return ValueError

            # Writing predicted wave file to disk
            s_regenerated = _signal_class.get_wave(S_predicted)
            swav.write(data=s_regenerated, rate=16000, filename=_wav_filename + '_predicted.wav')

            # Writing original wave file to disk
            s_original = _signal_class.get_wave(S_test_complex)
            swav.write(data=s_original, rate=16000, filename=_wav_filename + '_original.wav')

            # Writing noisy wave file to disk
            s_noisy = _signal_class.get_wave(X_test_complex_for_prediction)
            swav.write(data=s_noisy, rate=16000, filename=_wav_filename + '_noisy.wav')

            print_progress_bar(num_speech + 1, len(X_test_complex_list), prefix='Progress:', suffix='Complete',
                               length=50)

        print()

    print('Testing for model:', _model_name, 'is complete')


if __name__ == '__main__':

    # Fixing seed for reproducibility
    np.random.seed(345)

    # Model settings
    model_dict = {  # serial number of various models
        # starts from 0. Allows us to create a list of different model functions and call by index
        'architecture': 0,

        # target mask
        'mask': 'IBM',

        # Type of gender: both, male, female
        'gender': 'both',

        # Number of timesteps to lookback in LSTM sequence
        'lookback': 20,

        # Keras: control tp allow longterm dependency in LSTM training
        # Always false until stateful issue is resolved
        'stateful': False,

        # number of iterations
        'iter': 500,

        # batch size
        'batch': 100,

        # learning rate (may be redundant in adam optimizer)
        'learnrate': None,

        # Noise filter: list with any combination of the following noise types
        #   Possible Training noises: birds, casino, jungle, motorcycle, ocean
        #   Possible Testing noises: birds, casino, jungle, motorcycle, ocean, computerkeyboard
        'noise': [],

        # Scale filter:
        #   Possible entries in the list (any combination): '0dB', 'neg5dB', 'pos5dB'
        'scale': [],

        # window size for STFT
        'frame_size': 1024
    }

    # Binary switch to select test or train mode
    run_in_train_mode = False

    # instantiating signal class
    print('Instantiating Signal class')
    signal = speech_preprocessing.SignalProcessing(_frame_len=model_dict['frame_size'])

    # initializing file paths and names
    model_file_path = 'trained_models/'
    model_file_name = model_file_path + generate_model_file_name(model_dict) + '.h5'

    # Creates model file path if it does not exist
    os.makedirs(os.path.dirname(model_file_name), exist_ok=True)

    if run_in_train_mode:

        print('Initializing Training parameters')
        train_data_path = 'data/train/'

        # Training Data
        print('Training phase initiated')
        train_model_batch(_signal_class=signal, _train_data_path=train_data_path, _model_params=model_dict,
                          _model_file_name=model_file_name)

        print('Training phase complete')

    else:

        print('Initializing Test parameters')
        test_data_path = 'data/test/'
        test_result_path = 'test_results/'

        model_file_name = model_file_path + \
                          'architecture_0_mask_IBM_gender_male_lookback_20_stateful_False_iter_500_batch_100.h5'

        print('Testing phase initiated')
        test_model(signal, test_data_path, test_result_path, model_dict, model_file_name)
        # test_model(_signal_class=signal, _test_data_path=test_data_path, _model_file_name=model_file_name,
        #            _target_mask=model_dict['mask'], _output_wavefile_path=test_output_wave_file_path,
        #            _snr_path=test_snr_path, _gender=test_gender_filter, _noise=test_noise_filter,
        #            _scale=test_scale_filter)
        print('Testing phase completed')
