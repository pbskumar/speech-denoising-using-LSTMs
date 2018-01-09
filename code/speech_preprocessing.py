import numpy as np
import scipy.io.wavfile as swav
import scipy.signal

import os
import glob
import re
import pickle


def get_filtered_sound_file_names_list(_file_names_list, _gender='both', _noise=None, _scale=None):
    """
    Expecting each filename to be in folder1/folder2/gender/noise/scale format
    :param _file_names_list: List of filenames to be filtered
    :param _gender: Filter to load files with specific genders
    :param _noise: Filter to load files with specific noises
    :param _scale: Filter to load files with specific scales
    :return: Filenames that match the filters
    """
    _file_names_split_list = [re.split('[/_]+', fname) for fname in _file_names_list]

    if _gender != 'both':
        _file_names_split_list = [f_name for f_name in _file_names_split_list if f_name[-3] == _gender]

    if _noise:
        if type(_noise) == str:
            _noise = [_noise]
        _file_names_split_list = [f_name for f_name in _file_names_split_list if f_name[-2] in _noise]

    if _scale:
        if type(_scale) == str:
            _scale = [_scale]
        _file_names_split_list = [f_name for f_name in _file_names_split_list if f_name[-1] in _scale]

    _file_names_list = ['_'.join(['/'.join(fname_split[:3]), fname_split[-2], fname_split[-1]])
                        for fname_split in _file_names_split_list]

    return _file_names_list


class SignalProcessing:
    def __init__(self, _frame_len=1024, _fs=16000):
        self._N = _frame_len
        self._fs = _fs

    def get_stft(self, _signal):
        _, _, _stft = scipy.signal.stft(_signal, self._fs, nperseg=self._N)
        return _stft

    def get_wave(self, _stft):
        _, _signal = scipy.signal.istft(_stft, self._fs)
        _signal = (_signal - _signal.mean()) / _signal.std()
        return _signal

    @staticmethod
    def get_stack_spectrogram(_spectrogram_list):
        stacked_spectrogram = _spectrogram_list[0]

        for i in range(len(_spectrogram_list) - 1):
            stacked_spectrogram = np.hstack((stacked_spectrogram, _spectrogram_list[i + 1]))
        return stacked_spectrogram

    def read_processed_data(self, _folder_path, _target='IBM', _gender=None, _noise=None, _scale=None):

        # validating target
        if _target not in ['IBM', 'IRM', 'FFT']:
            print('Enter valid target keyword')
            return

        # generic spectrogram shape for any fame size
        num_features = (self._N // 2 + 1)

        # initializing output data
        stacked_dict = {'X': np.empty((num_features, 0)),
                        'S': np.empty((num_features, 0)),
                        'N': np.empty((num_features, 0))}
        _Y = None

        # Implement gender, noise and scale filters
        file_names_list = get_filtered_sound_file_names_list(glob.glob(_folder_path + '*'),
                                                             _gender=_gender, _noise=_noise, _scale=_scale)

        # iterate over each file,
        # read,
        # each file has a dict of data
        # append data to S, N, X
        file_names_len = len(file_names_list)
        i = 1
        for fname in file_names_list:
            print('Processing file', i, 'of', file_names_len)
            i += 1
            data_dict = pickle.load(open(fname, 'rb'))

            for key in ['X', 'S', 'N']:
                stacked_dict[key] = np.hstack((stacked_dict[key], self.get_stack_spectrogram(data_dict[key])))

        # Creating target
        if _target == 'IBM':
            _Y = (np.absolute(stacked_dict['S']) > np.absolute(stacked_dict['N'])).astype(int)
        elif _target == 'IRM':
            _S_square = np.square(np.absolute(stacked_dict['S']))
            _N_square = np.square(np.absolute(stacked_dict['N']))
            _Y = np.sqrt(np.divide(_S_square, (_S_square + _N_square)))
        elif _target == 'FFT':
            stacked_dict['X'] = np.log(1 + stacked_dict['X'])
            stacked_dict['S'] = np.log(1 + stacked_dict['S'])
            _Y = np.divide(stacked_dict['S'], stacked_dict['X'])

        # return features, target, original signal spectrogram
        return stacked_dict['X'], _Y, stacked_dict['S']

    @staticmethod
    def get_snr(original_signal, noisy_signal):

        noise = noisy_signal - original_signal
        return 10 * np.log10(original_signal.var() / noise.var())


if __name__ == '__main__':
    # setting seed
    np.random.seed(122)

    print('Initial set up started')

    # Setting Paths
    input_data_path = 'TIMIT/'
    input_noise_path = input_data_path + 'noise/'

    input_male_train_path = input_data_path + 'train/*/m*/'
    input_female_train_path = input_data_path + 'train/*/f*/'

    input_male_test_path = input_data_path + 'test/*/m*/'
    input_female_test_path = input_data_path + 'test/*/f*/'

    output_folder_path = 'data/'

    print('File paths set up complete')

    # setting frame size
    dft_frame_size = 1024

    # Instantiating Signal class
    signal_proc = SignalProcessing(dft_frame_size)

    print('Instantiated Signal class object')

    # Paths of all noise files
    noise_files = glob.glob(input_noise_path + '*wav', recursive=True)

    # list of all paths to be processed
    file_paths = [input_male_train_path, input_female_train_path,
                  input_male_test_path, input_female_test_path]

    print('Speech processing about to start')

    # iterating over each path
    for path in file_paths:

        speech_folder_path_split = re.split('\W+', path)

        print('Started processing:', speech_folder_path_split[1], speech_folder_path_split[2], 'files')

        speech_files = glob.glob(path + '*wav', recursive=True)

        random_speech_files = np.random.choice(speech_files, size=150)

        for noise_file in noise_files:
            # extracting name of each noise file
            noise_name = re.split('\W+', noise_file)[-2]
            # reading noise wav file
            _, noise_raw = swav.read(noise_file)

            print()
            print('Started processing', noise_name, 'noise:: ',
                  speech_folder_path_split[1], speech_folder_path_split[2], 'files')

            noise_raw_len = len(noise_raw)

            data_0dB = {'X': [], 'S': [], 'N': []}
            data_pos5dB = {'X': [], 'S': [], 'N': []}
            data_neg5dB = {'X': [], 'S': [], 'N': []}

            i = 0
            for speech_file in random_speech_files:
                if i % 10 == 0:
                    print('Status:', i, 'speech files processed')
                i += 1

                # reading speech wav file
                _, speech = swav.read(speech_file)

                speech_len = len(speech)

                # making random cut of noise signal
                max_idx = noise_raw_len - speech_len
                idx = np.random.randint(low=0, high=max_idx)
                noise = noise_raw[idx:idx + speech_len]
                assert len(noise) == speech_len

                # normalizing using mean and standard deviation
                noise_0dB = (noise - noise.mean()) / noise.std()

                # normalizing using mean and standard deviation
                speech_0dB = (speech - speech.mean()) / speech.std()

                # 0dB mixture
                x_0dB = speech_0dB + noise_0dB

                # 5dB mixture
                scale_pos5dB = 10 ** (-5 / 20)
                x_pos5dB = speech_0dB + scale_pos5dB * noise_0dB

                # -5dB mixture
                scale_neg5dB = 10 ** (5 / 20)
                x_neg5dB = speech_0dB + scale_neg5dB * noise_0dB

                # Generating spectrograms
                S = signal_proc.get_stft(speech_0dB)

                N_0dB = signal_proc.get_stft(noise_0dB)
                N_pos5dB = signal_proc.get_stft(scale_pos5dB * noise_0dB)
                N_neg5dB = signal_proc.get_stft(scale_neg5dB * noise_0dB)

                X_0dB = signal_proc.get_stft(x_0dB)
                X_pos5dB = signal_proc.get_stft(x_pos5dB)
                X_neg5dB = signal_proc.get_stft(x_neg5dB)

                data_0dB['X'].append(X_0dB)
                data_0dB['S'].append(S)
                data_0dB['N'].append(N_0dB)

                data_pos5dB['X'].append(X_pos5dB)
                data_pos5dB['S'].append(S)
                data_pos5dB['N'].append(N_pos5dB)

                data_neg5dB['X'].append(X_neg5dB)
                data_neg5dB['S'].append(S)
                data_neg5dB['N'].append(N_neg5dB)

                # --- processing for a signal file ends ---

            speech_file_path_split = re.split('\W+', random_speech_files[0])
            # test or train
            speech_type = speech_file_path_split[1]
            # male or female
            speaker_gender = 'male' if speech_file_path_split[-3][0] == 'm' else 'female'

            output_file_path_name = output_folder_path + speech_type + '/' + speaker_gender + '_'

            # Creating directory if it does not exist
            temp_file_name = output_file_path_name + noise_name + '_0dB'
            os.makedirs(os.path.dirname(temp_file_name), exist_ok=True)

            pickle.dump(file=open(output_file_path_name + noise_name + '_0dB', 'wb'), obj=data_0dB)
            pickle.dump(file=open(output_file_path_name + noise_name + '_pos5dB', 'wb'), obj=data_pos5dB)
            pickle.dump(file=open(output_file_path_name + noise_name + '_neg5dB', 'wb'), obj=data_neg5dB)

            print('Processed', noise_name, 'noise\n')
