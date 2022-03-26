"""
File : main.py
Author : Dominik Boboš
Description : Main script to start evaluation of the wanted system on the given files.
"""
import argparse
import glob
import os
import sys

import numpy as np

import DTWsystem
import RQAsystem
import FuzzyStringSystem


def CheckPositive(value):
    """
    Checks if input argument is bigger than zero.
    :param value: value from command line
    :return: checked value converted to int
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "{} is an invalid positive int value for argument --frame-reduction or --parallelization".format(value))
    return ivalue

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--src")
parser.add_argument("--feature", help="Features to use. (mfcc, posteriors, bottleneck, string)")
parser.add_argument("--cluster-feature", help="Features to use. (mfcc, posteriors, bottleneck)")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-d", "--dev", action="store_true", help="Development mode")
parser.add_argument("--system", required=True,
                    help="Available systems: basedtw, arenjansen/rqa_unknown, rqa_dtw_unknown/2pass_dtw_unknown, rqa_sdtw_unknown/2pass_sdtw_unknown,")
parser.add_argument("--frame-reduction", type=CheckPositive,
                    help="Downsampling the the feature vector, averaging given N-frames")
parser.add_argument("--parallelize-from", type=CheckPositive, help="Evaluating samples from the given number")
parser.add_argument("--parallelize-to", type=CheckPositive, help="Evaluating samples to the given number")
arguments = parser.parse_args()

features = ['mfcc', 'posteriors', 'bottleneck', 'string']
cluster_features = ['mfcc', 'posteriors', 'bottleneck']

if not arguments.src:
    if arguments.verbose:
        print("Current directory '{}' is chosen as source directory".format(os.getcwd))
    src = os.getcwd()
else:
    if arguments.verbose:
        print("Directory '{}' is chosen as source directory".format(arguments.src))
    src = arguments.src

if arguments.feature.lower() in features:
    if arguments.verbose:
        print("The {} features were chosen.".format(arguments.feature))
    feature = arguments.feature
else:
    print("MFCC features were chosen implicitly.")
    feature = 'mfcc'

if arguments.cluster_feature in features:
    if arguments.verbose:
        print("The {} cluster features were chosen.".format(arguments.cluster_feature))
    cluster_feature = arguments.cluster_feature
else:
    print("MFCC features for clusters were chosen implicitly. (ignore when not clustering)")
    cluster_feature = 'mfcc'


def FeatureToExtension(feature):
    """
    Switch to get file extension based on a given feature
    :param feature: feature in string
    :return: correct file extension
    """
    if feature == 'mfcc':
        return '*.wav'
    elif feature == 'posteriors':
        return '*/*.lin'
    elif feature == 'bottleneck':
        return '*/*.fea'
    elif feature == 'string':
        return '*/*.txt'


def GetFiles(file, feature):
    return glob.glob(file + FeatureToExtension(feature))


def LabelFiles(src, feature, exact_path=None, label=None, eval_only=False):
    """
    Labels target and non-target data
    :param src: source with train and eval subfolders with target and non/target files
    :param feature: which files to load
    :param exact_path: path to folder with goal, clear data
    :param label: list of wanted labels
    :param eval_only: unlabelled files
    :return: list of loaded and labelled files
    """
    if exact_path is None and label is None:
        test_target = np.array(GetFiles(src + 'eval/eval_goal/', feature))
        test_labels = np.array([1] * len(test_target))  # 1 obtain pre-recorded messsage
        test_clear = np.array(GetFiles(src + 'eval/eval_clear/', feature))
        test_labels = np.concatenate((test_labels, np.array([0] * len(test_clear))), axis=0)
        test_files = np.concatenate((test_target, test_clear), axis=0)

        train_target = np.array(GetFiles(src + 'train/train_goal/', feature))
        train_labels = np.array([1] * len(train_target))  # 1 obtain pre-recorded messsage
        train_clear = np.array(GetFiles(src + 'train/train_clear/', feature))
        train_labels = np.concatenate((train_labels, np.array([0] * len(train_clear))), axis=0)
        train_files = np.concatenate((train_target, train_clear), axis=0)

        return train_files, train_labels, test_files, test_labels
    elif exact_path is not None and label is not None:
        files0 = np.array(GetFiles(exact_path[0], feature))
        labels = np.array([label[0]] * len(files0))
        files1 = np.array(GetFiles(exact_path[1], feature))
        labels = np.concatenate((labels, np.array([0] * len(files1))), axis=0)
        files = np.concatenate((files0, files1), axis=0)
        return files, labels
    elif eval_only:
        return np.array(GetFiles(exact_path))
    else:
        raise Exception(
            "Type both exact_paths and labels (target first, second non-target) or leave it as None for LabelFiles funtion")


if __name__ == "__main__":
    """
    switch for all possible systems to evaluate
    """
    system = arguments.system.lower()
    exact_path = None
    label = None
    if not arguments.dev:
        exact_path = [src + 'eval_goal/', src + 'eval_clear/']
        label = [1, 0]
        test_files, test_labels = LabelFiles(src, feature, exact_path=exact_path, label=label)
        train_files, train_labels = None, None
    else:
        train_files, train_labels, test_files, test_labels = LabelFiles(src, feature)
    frame_reduction = arguments.frame_reduction if arguments.frame_reduction is not None else 1
    if len(test_files) != 0:
        par_from = 0
        par_to = len(test_files)
        if arguments.parallelize_from:
            par_from = arguments.parallelize_from
        if arguments.parallelize_to:
            par_to = arguments.parallelize_to
        if par_from > par_to:
            raise argparse.ArgumentTypeError("Parallelize from is bigger than parallel to")

    if system == 'basedtw':
        result_list = DTWsystem.BaseDtwUnknown([train_files, train_labels], [test_files, test_labels],
                                               feature=feature, reduce_dimension=True, parallelize=[par_from, par_to])
    if system == 'arenjansen' or system == 'rqa_unknown':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                       feature=feature,
                                       frame_reduction=frame_reduction, reduce_dimension=True)
    if system == 'rqacluster_sdtw_unknown' or system == '2pass_cluster_sdtw_unknown':
        result_list = RQAsystem.RqaDtw([train_files, train_labels], [test_files, test_labels],
                                       feature=feature,
                                       frame_reduction=frame_reduction, reduce_dimension=True,
                                       second_pass=True, sdtw=True, cluster=True, metric='cosine',
                                       parallelize=[par_from, par_to])
    if system == 'fuzzy_match_base':
        result_list = FuzzyStringSystem.FuzzyMatchSystem([train_files, train_labels], [test_files, test_labels],
                                                         system=system, parallelize=[par_from, par_to])
    if system == 'rqacluster_fuzzy_match_unknown':
        result_list = FuzzyStringSystem.FuzzyMatchSystem([train_files, train_labels], [test_files, test_labels],
                                                         system=system, clust_feature=cluster_feature,
                                                         parallelize=[par_from, par_to])
    if system == 'rqacluster_fuzzy_match_known':
        result_list = FuzzyStringSystem.FuzzyMatchSystem([train_files, train_labels], [test_files, test_labels],
                                                         system=system, clust_feature=cluster_feature,
                                                         parallelize=[par_from, par_to])
