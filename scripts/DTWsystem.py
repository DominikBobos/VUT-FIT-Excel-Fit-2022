"""
File : DTWsystem.py
Author : Dominik Boboš
Description : This script performs DTW/SDTW based systems
"""
import librosa
import numpy as np
from numpy.core._multiarray_umath import ndarray
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import ArrayFromFeatures
from fastdtw import fastdtw
from sklearn.utils import shuffle
import time

from third_party_scripts.seg_dtw.sdtw import segmental_dtw as SegmentalDTW


def MyDTW(o, r):
    """
    My DTW umplementation
    :param o: First sequence
    :param r: Seconds sequence
    :return: cost matrix, normalised DTW distance, DTW Alignment path
    """
    cost_matrix: ndarray = cdist(o, r, metric='euclidean')
    m, n = np.shape(cost_matrix)
    for i in range(m):
        for j in range(n):
            if (i == 0) & (j == 0):
                cost_matrix[i, j] = cost_matrix[i, j]  # inf
            elif i == 0:
                cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i, j - 1]  # inf
            elif j == 0:
                cost_matrix[i, j] = cost_matrix[i, j] + cost_matrix[i - 1, j]  # inf
            else:
                cost_matrix[i, j] = np.min([cost_matrix[i, j] + cost_matrix[i - 1, j],
                                            cost_matrix[i, j] + cost_matrix[i, j - 1],
                                            cost_matrix[i, j] * np.sqrt(2) + cost_matrix[i - 1, j - 1]])
    # backtracking
    path = [m - 1], [n - 1]
    i, j = m - 1, n - 1
    while i != 0 or j != 0:
        backtrack = np.argmin([cost_matrix[i - 1, j - 1], cost_matrix[i - 1, j], cost_matrix[i, j - 1]])
        if backtrack == 1:
            path[0].append(i - 1)
            path[1].append(j)
            i -= 1
        elif backtrack == 2:
            path[0].append(i)
            path[1].append(j - 1)
            j -= 1
        else:
            path[0].append(i - 1)
            path[1].append(j - 1)
            i -= 1
            j -= 1
    np_path = np.array(path, dtype=np.int64)
    return cost_matrix, cost_matrix[-1, -1] / (cost_matrix.shape[0] + cost_matrix.shape[1]), np_path.T


def Similarity(wp, interval=None):
    """
    Function for analysing DTW warping path and searching for diagonal lines
    :param wp: warping path
    :param interval: threshold for ratio function
    :return: list of candidates
    """
    if interval is None:
        interval = [0.98, 1.02]
    sim_list = []
    tmp_list = []
    score_ratio = 0
    ratio_list = []
    good_trend = 0
    false_trend = 0
    constant_false = 0
    have_hit = False
    RIGHT = 0
    UP = 1
    DIAG = 2
    direction = -1  # unknown at first
    prev_direction = -1
    prev_point = None
    for point in wp:  # np.flip(wp, 0):
        if prev_point is None:
            prev_point = point
            continue
        # going RIGHT ->
        if prev_point[0] == point[0] and prev_point[1] < point[1]:
            direction = RIGHT
        # going UP ^
        elif prev_point[0] < point[0] and prev_point[1] == point[1]:
            direction = UP
        # going DIAG ⟋`
        else:
            direction = DIAG
        # print("PREVIOUS DIRECTION:", prev_direction, "    DIRECTION:", direction)
        if tmp_list:
            if (direction == RIGHT and prev_direction == UP) or (direction == UP and prev_direction == RIGHT):
                constant_false = 0
            # false_trend += 1
            elif (direction == RIGHT and prev_direction == RIGHT) or (direction == UP and prev_direction == UP):
                # good_trend -= 1
                false_trend += 1
                constant_false += 1
            elif direction == DIAG:
                good_trend += 1
                constant_false = 0
            else:
                constant_false = 0
            tmp_list.append(point)

        if tmp_list == [] and direction == DIAG:
            if prev_direction == -1 or (prev_direction == DIAG and good_trend >= 5):
                tmp_list.append(point)
            good_trend += 1
            false_trend = 0
            constant_false = 0
        if constant_false >= 25:
            del tmp_list[-1]
            false_trend -= 1
            for i in range(constant_false):  # cleaning the list of the constant false points
                if len(tmp_list) > 0:
                    del tmp_list[-1]
                    false_trend -= 1
            if len(tmp_list) >= 200:
                ratio = (tmp_list[-1][0] - tmp_list[0][0]) / (tmp_list[-1][1] - tmp_list[0][1])
                if score_ratio < (1 - np.absolute(ratio - 1)) and have_hit is False:
                    score_ratio = 1 - np.absolute(ratio - 1)  # finding maximum
                if interval[0] < ratio < interval[1]:
                    if not have_hit:
                        score_ratio = 1 - np.absolute(ratio - 1)
                        have_hit = True
                    if score_ratio > ratio:
                        score_ratio = 1 - np.absolute(ratio - 1)
                    sim_list.append(np.array([tmp_list[0], tmp_list[-1]]))
                    ratio_list.append("{:.4f}".format(ratio))
            constant_false = 0
            false_trend = 0
            good_trend = 0
            tmp_list.clear()
        prev_point = point
        prev_direction = direction

    if len(tmp_list) >= 200:
        del tmp_list[-1]
        false_trend -= 1
        for i in range(constant_false):
            if len(tmp_list) > 0:
                del tmp_list[-1]
                false_trend -= 1
        ratio = (tmp_list[-1][0] - tmp_list[0][0]) / (tmp_list[-1][1] - tmp_list[0][1])
        if interval[0] < ratio < interval[1]:
            sim_list.append(np.array([tmp_list[0], tmp_list[-1]]))
            ratio_list.append("{:.4f}".format(ratio))
    sim_list = np.array(sim_list, dtype=object)
    return tuple(sim_list), tuple(ratio_list), score_ratio, have_hit


def InitCheckShuffle(train, test):
    """
    initialize and shuffles given list of files
    :param train: training dataset
    :param test: test dataset
    :return: shuffled datasets
    """
    if test is None or test[0] is None:
        test = []
        test_nested = []
    else:
        test[0], test[1] = shuffle(test[0], test[1], random_state=10)
        test_nested = test.copy()
    if train is None or train[0] is None:
        train = []
        train_nested = []
    else:
        train[0], train[1] = shuffle(train[0], train[1], random_state=10)
        train_nested = train.copy()
    return train, train_nested, test, test_nested


def GetThreshold(system, feature, metric):
    """
    Gets thresholds depending on the used parameters
    :param system: current system
    :param feature: used feature
    :param metric: distance mestric
    :return: correct threshold for the given system
    """
    if system == 'basedtw':
        hit_threshold = 0.0
        loop_count = 0
        if feature not in ['mfcc', 'posteriors', 'bottleneck']:
            raise Exception(
                "For BaseDTW system use only mfcc, posteriors or bottleneck feature vectors, not {}".format(feature))
        if feature == 'mfcc':
            hit_threshold = 35.0
            loop_count = 49
        if feature == 'posteriors':
            hit_threshold = 4.0
            loop_count = 49
        if feature == 'bottleneck':
            hit_threshold = 1.25
            loop_count = 99
        return hit_threshold, loop_count

    if system == 'arenjansen' or system == 'rqa_unknown':
        if feature not in ['mfcc', 'posteriors', 'bottleneck']:
            raise Exception(
                "For BaseDTW system use only mfcc, posteriors or bottleneck feature vectors, not {}".format(feature))
        rqa_threshold = 500.0
        hit_threshold = 50  # don't care random value
        loop_count = 50  # don't care random value
        return rqa_threshold, hit_threshold, loop_count

    if system == 'rqa_dtw_unknown' or system == '2pass_dtw_unknown':
        if feature not in ['mfcc', 'posteriors', 'bottleneck']:
            raise Exception(
                "For BaseDTW system use only mfcc, posteriors or bottleneck feature vectors, not {}".format(feature))
        rqa_threshold = 500.0
        loop_count = 50
        if feature == 'mfcc':
            hit_threshold = 60.0
        if feature == 'posteriors':
            hit_threshold = 9.0
        if feature == 'bottleneck':
            hit_threshold = 1.25
        return rqa_threshold, hit_threshold, loop_count

    if system == 'rqa_sdtw_unknown' or system == '2pass_sdtw_unknown':
        if feature not in ['mfcc', 'posteriors', 'bottleneck']:
            raise Exception(
                "For RQA/DTW system use only mfcc, posteriors or bottleneck feature vectors, not {}".format(feature))
        rqa_threshold = 500.0
        loop_count = 50
        if feature == 'mfcc':
            hit_threshold = 0.004
        if feature == 'posteriors':
            hit_threshold = 0.29
        if feature == 'bottleneck':
            hit_threshold = 0.28
        return rqa_threshold, hit_threshold, loop_count

    if system == 'rqa_cluster':
        if feature not in ['mfcc', 'bottleneck', 'posteriors']:
            raise Exception('{} is unsupported feature for clustering. Use MFCC or bottleneck or posteriors'.format(
                feature.upper()))
        if feature == 'mfcc':
            if metric == 'euclidean':
                hit_threshold = 35.0
            else:
                hit_threshold = 0.004
        if feature == 'posteriors':
            if metric == 'euclidean':
                hit_threshold = 10.0
            else:
                hit_threshold = 0.29
        if feature == 'bottleneck':
            if metric == 'euclidean':
                hit_threshold = 1.25
            else:
                hit_threshold = 0.29
        return hit_threshold
    if system == 'rqa_cluster_system':
        if feature not in ['mfcc', 'bottleneck', 'posteriors']:
            raise Exception('{} is unsupported feature for clustering. Use MFCC or bottleneck or posteriors'.format(
                feature.upper()))
        if feature == 'mfcc':
            if metric == 'euclidean':
                hit_threshold = 35.0
            else:
                hit_threshold = 0.0025
        if feature == 'posteriors':
            if metric == 'euclidean':
                hit_threshold = 10.0
            else:
                hit_threshold = 0.29
        if feature == 'bottleneck':
            if metric == 'euclidean':
                hit_threshold = 1.25
            else:
                hit_threshold = 0.20
        return hit_threshold


def BaseDtwUnknown(train=None, test=None, feature='mfcc', reduce_dimension=True, parallelize=[None, None]):
    """
    Baseline DTW system
    :param train: list of train files
    :param test: list of test files
    :param feature: type od feature vectors
    :param reduce_dimension: to reduce size of phoneme posteriors by 3
    :param parallelize: paralellization - will evaluate files from parallelize[0] to parallelize[1]
    :return: list of evaluated files and their dtw distance
    """
    train, train_nested, test, test_nested = InitCheckShuffle(train, test)
    result_list = []
    hits_count = 0
    threshold = [0.9, 1.1]
    if parallelize == [None, None]:
        parallelize = [0, len(test[0])]
    cache = {}
    hit_threshold, loop_count = GetThreshold('basedtw', feature, 'euclidean')
    loop_count = len(test[0]) // 8
    one_round = []
    parallelize_string = ''
    if parallelize[0] != 0 or parallelize[1] != len(test[0]):
        parallelize_string = str(parallelize[0]) + '-' + str(parallelize[1])
    for idx, file in enumerate(test[0]):  # test[0] == list of files
        if idx < parallelize[0]:
            continue
        if idx >= parallelize[1]:
            continue
        start_time = time.time()
        if file.split('/')[-1] in cache:
            parsed_file, file_array = cache[file.split('/')[-1]]
        else:
            parsed_file = ArrayFromFeatures.Parse(file)
            file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
            if len(file_array) == 0:
                continue
            cache[file.split('/')[-1]] = [parsed_file, file_array]
        score_list = []
        dist_list = []
        hit_dist = []
        test_nested[0], test_nested[1] = shuffle(test_nested[0], test_nested[1], random_state=10)
        for idx_nested, file_nested in enumerate(test_nested[0]):
            if file.split('/')[-1] == file_nested.split('/')[-1]:  # same file
                continue
            if file_nested.split('/')[-1] in cache:
                parsed_file_nested, file_nested_array = cache[file_nested.split('/')[-1]]
            else:
                parsed_file_nested = ArrayFromFeatures.Parse(file_nested)
                file_nested_array = ArrayFromFeatures.GetArray(file_nested, feature, reduce_dimension)
                if len(file_nested_array) == 0:
                    continue
                cache[file_nested.split('/')[-1]] = [parsed_file_nested, file_nested_array]
            final_dist, wp = fastdtw(file_array, file_nested_array, dist=euclidean)
            sim_list, ratio_list, score, hit = Similarity(np.asarray(wp), threshold)
            final_dist = final_dist / (file_array.shape[0] + file_nested_array.shape[0])
            if sim_list:
                distances = []
                for point in sim_list:
                    dist, wp = fastdtw(file_array[point[0][0]:point[1][0]], file_nested_array[point[0][1]:point[1][1]],
                                       dist=euclidean)
                    distances.append(dist / (file_array[point[0][0]:point[1][0]].shape[0] +
                                             file_nested_array[point[0][1]:point[1][1]].shape[0]))
                final_dist = min(distances)
            print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}). -> Distance={:.4f}".format(
                parsed_file[0], "0" if test[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(test[0]),
                parsed_file_nested[0], "0" if test_nested[1][idx_nested] == 0 else "1:" + parsed_file_nested[3],
                                                                                     idx_nested + 1, len(test[0]),
                final_dist))

            score_list.append(score)
            dist_list.append(final_dist)

            if final_dist < hit_threshold:
                hits_count += 1
                hit_dist.append(final_dist)
            # "surely" got hit, going to the next sample / or counting too much
            if hits_count >= 1 or idx_nested > loop_count:
                break
        f = open("{}evalBaseDTW_{}.txt".format(parallelize_string, feature), "a")

        # if len(result_list_nested) > 0:
        if hits_count > 0:
            hits_count = 0
            score_list = list(filter(lambda x: threshold[0] < x < threshold[1], score_list))
            print(score_list)
            append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "1",
                                                          min(score_list), max(hit_dist))
            result_list.append(append_string)  # train[1] == label
            f.write(append_string)
        else:
            # if not result_list_nested:
            if not score_list:
                append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0", "0.0",
                                                              min(dist_list))
                result_list.append(append_string)
                f.write(append_string)
            else:
                append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], test[1][idx], "0",
                                                              max(score_list), min(dist_list))
                result_list.append(append_string)  # train[1] == label
                f.write(append_string)
        f.close()
        one_round.append(time.time() - start_time)
        print('Next File. Time: {}s'.format(one_round[-1]))
    ft = open("{}timeBaseDTW_{}.txt".format(parallelize_string, feature), "a")
    ft.write("MEAN:{}\tTOTAL:{}\n".format(np.mean(one_round), np.sum(one_round)))
    ft.close()
    return result_list


def SecondPassCluster(data, clust_list, hit_threshold, frame_reduction, feature, reduce_dimension, rqa_time, metric,
                      sdtw, known, parallelize=[0, 0]):
    """
    Second pass approach using clusters
    :param data: list of files to evaluate
    :param clust_list: clusters
    :param hit_threshold: threshold to make hard decision
    :param frame_reduction: size of frame averaging
    :param feature: type of feature vector
    :param reduce_dimension: to reduce size of phoneme posteriors by 3
    :param rqa_time: time of RQA analysis if performed before
    :param metric: used metric to compute DTW/SDTW
    :param sdtw: use SDTW instead of DTW
    :param known: using reference cluster
    :param parallelize: paralellization - will evaluate files from parallelize[0] to parallelize[1]
    :return: list of evaluated files and their dtw distance and cluster class
    """
    dtw_time = []
    result_list = []
    cache = {}
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
            file_array = ArrayFromFeatures.GetArray(file, feature, reduce_dimension)
            if len(file_array) == 0:
                continue
            file_array = ArrayFromFeatures.ReduceFrames(file_array, size=frame_reduction)
            # file_array = ArrayFromFeatures.CompressFrames(file_array, size=frame_reduction)
            cache[file.split('/')[-1]] = [parsed_file, file_array]
        if float(parsed_file[-1]) < 3.0:  # skipping samples of total duration shorter than 3seconds
            print("Skipping", file.split('/')[-1], data[1][idx],
                  "because the total duration {} is less than 3.0 seconds".format(parsed_file[-1]))
            continue
        dist_list = []
        have_hit = False
        # loop through candidates-> queries to search in 'file' 
        for idx_nested, cluster_nested in enumerate(clust_list):
            have_hit = False
            for idx_file, file_nested in enumerate(cluster_nested):
                if idx_file > 1:  # go to next when first 2 are compared
                    break
                if file[-4:] != file_nested[0][-4:]:  # extension comparision -> could return None
                    file_nested[0] = ArrayFromFeatures.RightExtensionFile(file_nested[0], feature)
                if not file_nested[0]:  # the wanted file does not exist in the given path
                    continue
                if file.split('/')[-1] == file_nested[0].split('/')[-1]:
                    continue
                if file_nested[0].split('/')[-1] in cache:
                    parsed_file_nested, file_nested_array = cache[file_nested[0].split('/')[-1]]
                else:
                    parsed_file_nested = ArrayFromFeatures.Parse(file_nested[0])
                    file_nested_array = ArrayFromFeatures.GetArray(file_nested[0], feature, reduce_dimension)
                    if len(file_nested_array) == 0:
                        continue
                    file_nested_array = ArrayFromFeatures.ReduceFrames(file_nested_array, size=frame_reduction)
                    cache[file_nested[0].split('/')[-1]] = [parsed_file_nested, file_nested_array]
                start = int(file_nested[1][0][0]) * file_nested[2] // frame_reduction
                end = int(file_nested[1][-1][0]) * file_nested[2] // frame_reduction
                if sdtw:
                    path = SegmentalDTW(file_nested_array[start:end],
                                        file_array,
                                        R=5,
                                        L=200 / frame_reduction,
                                        dist=metric)
                    wp = np.asarray(path[1][3]) * frame_reduction
                    occurence_frames = [wp[-1][1], wp[0][1]]
                    dtw_distance = path[0]
                    if feature == 'posteriors' and dtw_distance < 0.01:
                        dtw_distance *= 100  # normalization for weird cases
                else:
                    cost_matrix, wp = librosa.sequence.dtw(X=file_array.T,
                                                           Y=file_nested_array[start:end].T,
                                                           metric='euclidean',
                                                           weights_mul=np.array([np.sqrt([2]), 1, 1],
                                                                                dtype=np.float64))  # cosine rychlejsie
                    dtw_distance = cost_matrix[wp[-1, 0], wp[-1, 1]]
                if dtw_distance is None:
                    continue
                print("Processing {}[{}] ({}/{}) with {}[{}] ({}/{}) cluster number [{}]. -> Distance={:.4f}".format(
                    parsed_file[0], "0" if data[1][idx] == 0 else "1:" + parsed_file[3], idx + 1, len(data[0]),
                    parsed_file_nested[0], "0" if len(parsed_file_nested) < 10 else "1:" + parsed_file_nested[3],
                                                                                         idx_file + 1,
                    len(cluster_nested) if len(cluster_nested) < 2 else 2, idx_nested, dtw_distance))
                if dtw_distance < hit_threshold:
                    have_hit = True
                    f = open(
                        "{}evalRQAclustered_{}_{}_{}.txt".format(parallelize_string, "SDTW" if sdtw == True else "DTW",
                                                                 "known" if known else "unknown", feature), "a")
                    if sdtw:
                        append_string = "{}\t{}\t{}\t{}\t{}-{}\t{}\n".format(file.split('/')[-1], data[1][idx], "1",
                                                                             idx_nested, occurence_frames[0],
                                                                             occurence_frames[1], dtw_distance)
                    else:
                        append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], data[1][idx], "1",
                                                                      idx_nested, dtw_distance)
                    result_list.append(append_string)  # train[1] == label
                    f.write(append_string)
                    f.close()
                    # if it is wanted to add to cluster
                    # clust_list[idx_nested].append([cluster[1][0], cluster[1][1], cluster[1][2], 
                    #                     dtw_distance, len(cluster[1][1])*cluster[1][2]])
                    # clust_list[idx_nested] = sorted(clust_list[idx_nested], key = lambda x: x[-1]) # sort with new item
                    break
                dist_list.append(dtw_distance)
            if have_hit:
                break
        if not have_hit:
            f = open("{}evalRQAclustered_{}_{}_{}.txt".format(parallelize_string, "SDTW" if sdtw is True else "DTW",
                                                              "known" if known else "unknown", feature), "a")
            append_string = "{}\t{}\t{}\t{}\t{}\n".format(file.split('/')[-1], data[1][idx], "0", "XX", min(dist_list))
            result_list.append(append_string)  # train[1] == label
            f.write(append_string)
            f.close()
        dtw_time.append(time.time() - start_time)
        print('Next File. Time: {}s'.format(dtw_time[-1]))
    print(np.sum(dtw_time), np.mean(dtw_time))
    ft = open("{}timeRQAclustered_{}_{}_{}.txt".format(parallelize_string, "SDTW" if sdtw is True else "DTW",
                                                       "known" if known else "unknown", feature), "a")
    ft.write("RQA_MEAN:{}\tRQA_TOTAL:{}\tDTW_MEAN:{}\tDTW_TOTAL:{}\tTOTAL_MEAN:{}\tTOTAL_TIME:{}\n".format(
        np.mean(rqa_time), np.sum(rqa_time),
        np.mean(dtw_time), np.sum(dtw_time),
        np.mean(rqa_time) + np.mean(dtw_time), np.sum(rqa_time) + np.sum(dtw_time)))
    ft.close()
    return result_list
