"""
File : ArrayFromFeatures.py
Author : Dominik Boboš
Description : This script load and parse input files - load feature vectors of various extentions,
"""
import os
import pickle

import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile

from third_party_scripts.htk.HTK import HTKFile


def GetMFCC(file, verbose=False):
    """
    Extract MFCC features from .wav file
    :param file: .wav file
    :param verbose:
    :return: MFCC feature vector
    """
    sample_f, signal = wavfile.read(file)
    mfcc_arr = mfcc(signal, sample_f, 0.025, 0.01, 13, 512)
    if verbose:
        print("FILE | SAMPLE | NUMPY ARRAY SHAPE")
        print(file, sample_f, mfcc_arr.shape, sep=" | ")
    return mfcc_arr


def LoadHTK(file, verbose=False):
    """
    Load feature vector from htk file
    :param file: file in htk format
    :param verbose:
    :return: Numoy array of loaded htk feature vector
    """
    htk_reader = HTKFile()
    htk_reader.load(file)
    feature = np.array(htk_reader.data)

    if verbose:
        print("FILE | SAMPLES NUM | FEATURES NUM | SAMPLE PERIOD | QUALIFIERS")
        print(file, htk_reader.nSamples, htk_reader.nFeatures, htk_reader.sampPeriod, htk_reader.qualifiers, sep=" | ")
        print("FILE | NUMPY ARRAY SHAPE")
        print(file, feature.shape, sep=" | ")
    return feature


def LoadString(file):
    """
    Load and parse phone string
    :param file: phone string in .txt with info about duration of phoneme, its framing and the phoneme itself
    :return: parsed phone string
    """
    f = open(file, "r")
    lines = f.readlines()
    frames = []
    lengths = []
    index_map = {}
    string = ''
    for idx, line in enumerate(lines):
        # [0] - start 
        # [1] - end
        # [2] - phoneme
        # [3] - length # in hundredths of a second
        split_line = line.split(' ')
        start = int(split_line[0][:-5] if split_line[0] != '0' else 0)
        end = int(split_line[1][:-5])
        phoneme = split_line[2] if split_line[2] != 'pau' else " "  # convert pause to space
        index_map[idx] = phoneme
        length = abs(float(split_line[3][:-1]))
        frames.append([start, end])
        string += phoneme
        lengths.append(length)
    f.close()
    return [frames, lengths, index_map, string]


def ReduceDimension(feature):
    """
    Reduce dimension of phone posteriors by 3 - reduce triple- classes of each phoneme to one
    :param feature: phone posteriors numpy array
    :return: reduced numpy array
    """
    reduced = np.empty([feature.shape[0], feature.shape[1] // 3])
    for frame in range(feature.shape[0]):
        for i in range(0, feature.shape[1], 3):
            reduced[frame, i // 3] = (feature[frame, i] + feature[frame, i + 1] + feature[frame, i + 2])
    return reduced


def ReduceFrames(feature, size=5):
    """
    Reduce size of given feature vector by given size - average the count of "size" frames into one
    :param feature: feature vector in numpy array
    :param size: the count of frames to averaging
    :return: reduced numpy array
    """
    reduced = np.empty([feature.shape[0] // size, feature.shape[1]])
    for frame in range(0, reduced.shape[0]):
        for i in range(0, reduced.shape[1]):
            reduced[frame, i] = np.mean(feature[frame*size:frame*size + size, i])
    return reduced


def CompressFrames(feature, size=5):
    """
    Reduce size of given feature vector by given size - only each "size"-th frame is used
    :param feature: feature vector in numpy array
    :param size: each "size"-th frame is preserved
    :return: reduced numpy array
    """
    return np.compress([True if i % size == 0 else False for i in range(feature.shape[0])], feature, axis=0)


def GetOneFolderUp(file):
    """
    Get one folder up while preserving the filename
    :param file: filename path
    :return: the same file in parent folder
    """
    processed = '' 
    folders = file.split('/')
    for idx, folder in enumerate(folders):
        if idx == len(folders)-2: #the last but one item (folder to get away from)
            continue
        if idx == len(folders)-1: #the last one (file)
            processed += folder #filename
            continue
        processed += folder + '/'
    return processed

def GetOneFolderDeep(file, folder_extension):
    """
    Get to the corresponding folder of needed feature vector
    :param file: file path
    :param folder_extension: extension of folder (lat, phn, bnf, str)
    :return: file located to the corresponding filder
    """
    processed = '' 
    folders = file.split('/')
    for idx, folder in enumerate(folders):
        if idx == len(folders)-2: #the last but one item (folder to append deeper folder)
            processed += folder + '/' + folder + folder_extension + '/'
            continue
        if idx == len(folders)-1: #the last one (file)
            processed += folder #filename
            continue
        processed += folder + '/'
    return processed


def Get2RigthFolder(file, folder_extension):
    """
    Get to right folder of needed extension
    :param file:
    :param folder_extension:
    :return:
    """
    processed = '' 
    folders = file.split('/')
    for idx, folder in enumerate(folders):
        if idx == len(folders)-2: #the last but one item (folder to rename to right folder)
            processed += folder[:-4] + folder_extension + '/' 
            continue
        if idx == len(folders)-1: #the last one (file)
            processed += folder #filename
            continue
        processed += folder + '/'
    return processed


def RightExtensionFile(file, feature):
    """
    Get corresponding folder extension based on feature vector type
    :param file:
    :param feature:
    :return: folder extension
    """
    extension = ''
    folder_extension = ''
    if feature.lower() == 'mfcc':
        extension = '.wav'
        folder_extension = ''
    elif feature.lower() == 'posteriors':
        extension = '.lin'
        folder_extension = '_phn' 
    elif feature.lower() == 'bottleneck':
        extension = '.fea'
        folder_extension = '_bnf'
    elif feature.lower() == 'string':
        extension = '.txt'
        folder_extension = '_str'
    else:
        raise Exception("Unsupported feature '{}'".format(feature))

    if file[-4:] == extension:
        result = file
    elif file[-4:] == '.wav':
        result = GetOneFolderDeep(file, folder_extension)[:-4] + extension 
    elif extension == '.wav':
        result = GetOneFolderUp(file)[:-4] + extension  # from wav folder to the extension folder
    else:
        result = Get2RigthFolder(file, folder_extension)[:-4] + extension  # renaming file folder
    if os.path.exists(result):
        return result
    else: 
        return None


def OpenPickle(path):
    """
    open pickle file and load it
    :param path: file path to pkl file
    :return: opened and loaded python object
    """
    pkl_list = []
    try:
        open_file = open(path, "rb")
        pkl_list = pickle.load(open_file)
        open_file.close()
        print("{} file exist. Loading data from pickle file.".format(path))
    except IOError:
        raise("{} file not found. No data will be retrieved.".format(path))
    return pkl_list


def SavePickle(path, pkl_list):
    """
    Save python object to pkl file of name path
    :param path: file path of pkl file
    :param pkl_list: python object to save
    """
    open_file = open(path, "wb")
    pickle.dump(pkl_list, open_file)
    open_file.close()
    print(path, "file saved successfully.")


def Parse(filename):
    """
    Parse target and non-target files based on their filename
    :param filename:
    :return: list of information about the given file
    """
    parsed_file = []
    del_path = ''
    temp = ''
    try:
        del_path = filename.split('/')[-1]
    except IndexError:
        del_path = filename
    temp = del_path.split('_')
    parsed_file.append(temp[0])  # sample ID name
    parsed_file.append(int(temp[1]))  # part of cut sample
    # <20 -> sample without message
    if len(del_path) < 25:
        parsed_file.append(float(temp[2][:-4]))  # sample duration in seconds
        parsed_file.append(temp[2][-3:])  # extension format (wav, lin etc.)
        parsed_file.append(float(temp[2][:-4]))  # total duration
        # [sampleID, cutPart, sampleDuration, extension, totalDuration] -> no pre-recorded message 
        return parsed_file
    else:
        parsed_file.append(float(temp[2]))  # sample duration in seconds
    temp = del_path.split('__')[-1]  # part with operator message info
    parsed_file.append(temp.split('_')[0])  # operator message ID
    temp = temp.split('_')[-1].split('(')
    parsed_file.append(float(temp[1].split(')')[0]))  # message start point
    parsed_file.append(float(temp[2].split(')')[0]))  # message duration in seconds
    parsed_file.append(float(temp[3].split(')')[0]))  # volume gain
    parsed_file.append(float(temp[4].split(')')[0]))  # repeat count
    parsed_file.append(float(temp[5].split(')')[0]))  # speed
    parsed_file.append(temp[5].split('.')[-1])  # extension format (wav, lin etc.)
    parsed_file.append(parsed_file[2] + parsed_file[5])  # total duration of mixed file
    # [sampleID, cutPart, sampleDuration, messageID, messStart, messDuration, volume, repeat, speed, extension, totalDuration]
    return parsed_file


def GetArray(file, feature, reduce_dimension):
    """
    Get corresponding feature vector numpy array of given feature
    :param file: file path
    :param feature:
    :param reduce_dimension: to reduce phoneme posteriors 3 times
    :return: numpy array of a feature vector
    """
    try:
        if feature == 'mfcc':
            file_arr = GetMFCC(file)
        if feature == 'posteriors':
            file_arr = LoadHTK(file)
            if reduce_dimension:
                file_arr = ReduceDimension(file_arr)
        if feature == 'bottleneck':
            file_arr = LoadHTK(file)
        return file_arr
    except NotImplementedError:
        return []

if __name__ == '__main__':
    # parse phoneme string when running as script itself
    import sys
    print(LoadString(sys.argv[1]))