"""
File : FuzzyStringSystem.py
Author : Dominik Boboš
Description : This script performs Fuzzy phone string match based systems
"""
import time

import numpy as np
from fuzzywuzzy import fuzz
from sklearn.utils import shuffle

import ArrayFromFeatures
import DTWsystem


def ReturnIndexOccurrences(string_dict, char):
    """
    returns indices of all occurrences of character ch
    :param string_dict: string dictionary of phonemes
    :param char: phoneme to perform searching
    :return: returns indices of all occurrences of character ch, returns [] when ch is not found
    """
    return [index for index, value in string_dict.items() if value == char]


def AssemblyString(string_dict, start, end):
    """
    Assembles string from phoneme string dictionary to perform fuzzy match
    :param string_dict: string dictionary of phonemes
    :param start: index to start assembling
    :param end: index to stop assembling
    :return: assembled string
    """
    assembled_string = ""
    for i in range(start, end):
        assembled_string += string_dict[i]
    return assembled_string


def GetFramesFromIndices(file, start, end):
    """
    Get element frames from indices
    :param file: loaded phoneme string
    :param start: start index
    :param end: end index
    :return: the starting and end frames of the phonemes
    """
    return file[0][start][0], file[0][end][1]


def GetIndicesFromFrames(file, start, end):
    """
    Get indices of a phoneme string
    :param file: loaded phoneme string
    :param start: starting frame
    :param end: ending frame
    :return: the starting and end substring of the phoneme string
    """
    start_idx = 0
    end_idx = 0
    if len(file[0]) < 1:
        return start_idx, end_idx
    for idx, start_frame in enumerate(file[0]):
        if start_frame[0] > start:
            start_idx = idx - 1  # the previous one is the one we want
            break
        elif len(file[0]) == idx + 1:
            start_idx = idx  # the last frame is the one we want (but this should not happen)
    for idx, end_frame in enumerate(file[0]):
        if end_frame[1] > end:
            end_idx = idx - 1  # the previous one is the one we want
            break
        elif len(file[0]) == idx + 1:
            end_idx = idx  # the last frame is the one we want
    return start_idx, end_idx


def FindLongPau(file, char=' '):
    """
    Pause analysis - searching for long silent parts
    :param file: parsed string
    :param char: phoneme for finding long parts,
    :return: segmented string
    """
    pause_indices = ReturnIndexOccurrences(file[2], char)
    trimmed_string = []
    long_pause_indices = []
    if len(pause_indices) == 0:
        trimmed_string.append(((0, len(file[2]) - 1), file[-1]))
    for i, index in enumerate(pause_indices):
        if file[1][index] > 200.0:  # pause longer than 2 seconds
            long_pause_indices.append(index)
    if long_pause_indices:
        for i, index in enumerate(long_pause_indices):
            if i == 0:
                start = 0
            else:
                start = long_pause_indices[i - 1]
            if i == len(long_pause_indices) - 1:  # the last one:
                end = len(file[2]) - 1  # the last index in the original string
            else:
                end = index
            if (end - start) > 50:  # string of more interests
                assembled_string = AssemblyString(file[2], start, end)
                trimmed_string.append(((start, end), assembled_string))  # tuple (index of the start from the original
                # string and end, trimmed string)
    if len(trimmed_string) == 0:
        trimmed_string.append(((0, len(file[2]) - 1), file[-1]))
    return trimmed_string


def FuzzyCompareBase(data, hit_threshold, parallelize=[0, 0]):
    """
    Baseline fuzzy phone string matching
    :param data: list of data to evaluate
    :param hit_threshold: threshold for hard decision
    :param parallelize: paralellization - will evaluate files from parallelize[0] to parallelize[1]
    :return: evaluation results
    """
    fuzzy_time = []
    result_list = []
    score_list = []
    cache = {}
    have_hit = False
    data_nested = data.copy()
    if len(data[0]) > 2000:
        data_nested[0], data_nested[1] = shuffle(data_nested[0], data_nested[1], random_state=10)
    parallelize_string = ''
    if parallelize[0] != 0 or parallelize[1] != len(data[0]):
        parallelize_string = str(parallelize[0]) + '-' + str(parallelize[1])
    for idx, file in enumerate(data[0]):
        if idx < parallelize[0]:
            continue
        if idx >= parallelize[1]:
            continue
        start_time = time.time()
        if file.split('/')[-1] in cache:
            parsed_file, file_array = cache[file.split('/')[-1]]
        else:
            parsed_file = ArrayFromFeatures.Parse(file)
            file_array = ArrayFromFeatures.LoadString(file)
            cache[file.split('/')[-1]] = [parsed_file, file_array]
        if len(data[0]) > 2000:
            data_nested[0], data_nested[1] = shuffle(data_nested[0], data_nested[1], random_state=10)
        for idx_nested, file_nested in enumerate(data_nested[0]):
            if file.split('/')[-1] == file_nested.split('/')[-1]:
                continue
            if file_nested.split('/')[-1] in cache:
                parsed_file_nested, file_nested_array = cache[file_nested.split('/')[-1]]
            else:
                parsed_file_nested = ArrayFromFeatures.Parse(file_nested)
                file_nested_array = ArrayFromFeatures.LoadString(file_nested)
                cache[file_nested.split('/')[-1]] = [parsed_file_nested, file_nested_array]
            score = fuzz.partial_ratio(file_array[-1], file_nested_array[-1])
            print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}). -> score={}".format(
                parsed_file[0], "0" if data[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(data[0]),
                parsed_file_nested[0], "0" if len(parsed_file_nested) < 10 else "1:" + parsed_file_nested[3],
                idx_nested + 1, len(data_nested[0]), score))
            if score > hit_threshold and score != 100 and score != 75 and score != 67:
                have_hit = True
                f = open("{}evalBaseFuzzy_unknown.txt".format(parallelize_string), "a")
                append_string = "{}\t{}\t{}\t{}\n".format(file.split('/')[-1], data[1][idx], "1", score)
                result_list.append(append_string)  # train[1] == label
                f.write(append_string)
                f.close()
                break
            else:
                if score != 100 and score != 75 and score != 67:
                    score_list.append(score)
            if len(data[0]) > 2000 and idx_nested > (len(data[0]) // 4):
                break
        if not have_hit:
            f = open("{}evalBaseFuzzy_unknown.txt".format(parallelize_string), "a")
            if score_list == []:
                score_list.append(1)
            append_string = "{}\t{}\t{}\t{}\n".format(file.split('/')[-1], data[1][idx], "0", np.max(score_list))
            result_list.append(append_string)  # train[1] == label
            f.write(append_string)
            f.close()
        fuzzy_time.append(time.time() - start_time)
        print('Next File. Time: {}s'.format(fuzzy_time[-1]))
        score_list = []
        have_hit = False
    ft = open("{}timeBaseFuzzy_unknown.txt".format(parallelize_string), "a")
    ft.write("MEAN:{}\tTOTAL:{}\n".format(np.mean(fuzzy_time), np.sum(fuzzy_time)))
    ft.close()
    return result_list


def RQAclusterFuzzyMatch(data, clust_list, hit_threshold, known, cluster_feature, parallelize=[0, 0]):
    """
    Fuzzy matching system using clusters and pause analysis
    :param data: list of data to evaluate
    :param clust_list: clusters
    :param hit_threshold: threshold for hard decision
    :param known: usage of referenced cluster
    :param cluster_feature: feature vector of used clusters
    :param parallelize: paralellization - will evaluate files from parallelize[0] to parallelize[1]
    :return:
    """
    fuzzy_time = []
    result_list = []
    score_list = []
    cache = {}
    cache_cluster = {}
    have_hit = False
    parallelize_string = ''
    if parallelize[0] != 0 or parallelize[1] != len(data[0]):
        parallelize_string = str(parallelize[0]) + '-' + str(parallelize[1])
    for idx, file in enumerate(data[0]):
        if idx < parallelize[0]:
            continue
        if idx >= parallelize[1]:
            continue
        start_time = time.time()
        if file.split('/')[-1] in cache:
            parsed_file, file_array = cache[file.split('/')[-1]]
        else:
            parsed_file = ArrayFromFeatures.Parse(file)
            file_array = ArrayFromFeatures.LoadString(file)
            file_array[-1] = FindLongPau(file_array)
            cache[file.split('/')[-1]] = [parsed_file, file_array]

        # loop through candidates-> queries to search in 'file' 
        for idx_nested, cluster_nested in enumerate(clust_list):

            for idx_file, file_nested in enumerate(cluster_nested):
                if idx_file > 2:  # go to next when first 3 are compared
                    break
                if file[-4:] != file_nested[0][-4:]:  # extension comparision
                    file_nested[0] = ArrayFromFeatures.RightExtensionFile(file_nested[0], "string")
                if not file_nested[0]:  # does not exist .txt file for the given file in the path
                    continue
                if file.split('/')[-1] == file_nested[0].split('/')[-1]:
                    continue
                if file_nested[0].split('/')[-1] in cache_cluster:
                    parsed_file_nested, file_nested_array = cache_cluster[file_nested[0].split('/')[-1]]
                else:
                    parsed_file_nested = ArrayFromFeatures.Parse(file_nested[0])
                    file_nested_array = ArrayFromFeatures.LoadString(file_nested[0])
                    start = int(file_nested[1][0][0]) * file_nested[2]
                    end = int(file_nested[1][-1][0]) * file_nested[2]
                    start, end = GetIndicesFromFrames(file_nested_array, start, end)
                    assembled_string = AssemblyString(file_nested_array[2], start, end)
                    file_nested_array[-1] = [((start, end), assembled_string)]
                    cache_cluster[file_nested[0].split('/')[-1]] = [parsed_file_nested, file_nested_array]
                for idx_split, file_string in enumerate(file_array[-1]):
                    score = fuzz.partial_ratio(file_string[-1], file_nested_array[-1])
                    print(
                        "Processing {}[{}] ({}/{})[{}/{}] with {}[{}] ({}/{}) cluster number [{}]. -> score={}".format(
                            parsed_file[0], "0" if data[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(data[0]),
                                                                                                 idx_split + 1,
                            len(file_array[-1]),
                            parsed_file_nested[0],
                            "0" if len(parsed_file_nested) < 10 else "1:" + parsed_file_nested[3], idx_file + 1,
                            len(cluster_nested) if len(cluster_nested) < 3 else 3, idx_nested,
                            score))
                    if score > hit_threshold and score != 100 and score != 75:
                        have_hit = True
                        f = open("{}evalRQAclusteredFuzzyMatch_{}_{}.txt".format(parallelize_string,
                                                                                 "known" if known else "unknown",
                                                                                 cluster_feature), "a")
                        hit_frames = GetFramesFromIndices(file_array, file_string[0][0], file_string[0][1])
                        append_string = "{}\t{}\t{}\t{}\t{}-{}\t{}\n".format(file.split('/')[-1], data[1][idx], "1",
                                                                             idx_nested, hit_frames[0], hit_frames[1],
                                                                             score)
                        result_list.append(append_string)  # train[1] == label
                        f.write(append_string)
                        f.close()
                        break
                    else:
                        if score != 100 and score != 75:
                            score_list.append(score)
                if have_hit:
                    break
            if have_hit:
                break
        if not have_hit:
            f = open(
                "{}evalRQAclusteredFuzzyMatch_{}_{}.txt".format(parallelize_string, "known" if known else "unknown",
                                                                cluster_feature), "a")
            if score_list == []:
                score_list.append(1)
            append_string = "{}\t{}\t{}\t{}\n".format(file.split('/')[-1], data[1][idx], "0", np.max(score_list))
            result_list.append(append_string)  # train[1] == label
            f.write(append_string)
            f.close()
        fuzzy_time.append(time.time() - start_time)
        print('Next File. Time: {}s'.format(fuzzy_time[-1]))
        score_list = []
        have_hit = False
    print(np.sum(fuzzy_time), np.mean(fuzzy_time))
    ft = open("{}timeRQAclusteredFuzzyMatch_{}_{}.txt".format(parallelize_string, "known" if known else "unknown",
                                                              cluster_feature), "a")
    ft.write("MEAN:{}\tTOTAL:{}\n".format(np.mean(fuzzy_time), np.sum(fuzzy_time)))
    ft.close()
    return result_list


def FuzzyMatchSystem(train=None, test=None, system='fuzzy_match_base', clust_feature="bottleneck",
                     parallelize=[None, None]):
    """
    Switcher preparing the classification process of the wanted system.
    :param train: list of datafiles
    :param test: list of datafiles
    :param system: specified system
    :param clust_feature: feature vector of used clusters
    :param parallelize: paralellization - will evaluate files from parallelize[0] to parallelize[1]
    :return: evaluation results
    """
    train, train_nested, test, test_nested = DTWsystem.InitCheckShuffle(train, test)
    clust_list = []
    result_list = []
    data = test
    if parallelize == [None, None]:
        parallelize = [0, len(data[0])]
    if system == 'fuzzy_match_base':
        result_list = FuzzyCompareBase(data=data, hit_threshold=60, parallelize=parallelize)
    if system == 'fuzzy_match_pau_analysis':
        result_list = FuzzyComparePauAnalysis(data=data, hit_threshold=67, parallelize=parallelize)
    if system == 'rqacluster_fuzzy_match_unknown':
        try:
            clust_list = ArrayFromFeatures.OpenPickle(
                "evaluations/objects/cluster_rqa_list_{}.pkl".format(clust_feature))
        except:
            raise Exception("Could not provide RQAclusterFuzzyMatch, no cluster found. Create a cluster first")
        result_list = RQAclusterFuzzyMatch(data=data, clust_list=clust_list, hit_threshold=51, known=False,
                                           cluster_feature=clust_feature, parallelize=parallelize)
    if system == 'rqacluster_fuzzy_match_known':
        try:
            clust_list = ArrayFromFeatures.OpenPickle(
                "evaluations/objects/cluster_known_list_{}.pkl".format(clust_feature))
        except:
            raise Exception("Could not provide RQAclusterFuzzyMatch, no cluster found. Create a cluster first")
        result_list = RQAclusterFuzzyMatch(data=data, clust_list=clust_list, hit_threshold=51, known=True,
                                           cluster_feature=clust_feature, parallelize=parallelize)
    return result_list


if __name__ == "__main__":
    import sys

    file_array = ArrayFromFeatures.LoadString(sys.argv[1])
    string = FindLongPau(file_array)
    real_frames = GetFramesFromIndices(file_array, string[0][0][0], string[0][0][1])
    indices = GetIndicesFromFrames(file_array, real_frames[0], real_frames[1])
    print(string, real_frames, indices)
