"""
File : evaluate.py
Author : Dominik Boboš
Description : Evaluate performance of different systems (plot DET, EER, minDCF, the best threshold)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# import third_party_scripts.pytel.det as det
from third_party_scripts.pytel import det

for system in sys.argv[1:]:
    f = open(system, "r")
    system_name = system.split('/')[-1][4:-4]
    lines = f.readlines()
    non_target = []
    target = []
    scores = []  # for scikit_learn
    labels = []  # for scikit_learn
    overlap_score = []
    min_value = 0
    for line in lines:
        # [0] - filename
        # [1] - class
        # [2] - hard decision
        # [3] - ratio function for DTW system
        # [-3] - cluster index for cluster systems detecting exact frames of the pre-recorded message
        # [-2] - cluster index for cluster systems/ detected frames with the pre-recorded message for
        # cluster and exact detection systems
        # [-1] - soft score
        if line.split('\t')[1] == '0':
            non_target.append(line.split('\t')[-1][:-1])
            labels.append(0)
        else:
            target.append(line.split('\t')[-1][:-1])
            labels.append(1)
        scores.append(float(line.split('\t')[-1][:-1]))

    f.close()
    if system.find('evalRQA_SDTW_unknown_posteriors') == -1 and system.find('evalRQAclusteredFuzzyMatch') == -1 and (
            system.find('DTW') != -1 or system.find('RQA') != -1):
        target = np.array(target).astype(np.float64)
        min_value = np.min(target)
        print("Min value:", min_value, "Mean value:",
              np.mean(np.concatenate([target, np.array(non_target).astype(np.float64)])))
        target = min_value / target
        non_target = min_value / np.array(non_target).astype(np.float64)
    else:
        target = np.array(target).astype(np.float64)
        non_target = np.array(non_target).astype(np.float64)
        print("Max value:", np.max(target), "Mean value:", np.mean(np.concatenate([target, non_target])))
    fpr, tpr, thresholds = metrics.roc_curve(labels, np.min(scores) / scores if min_value != 0 else scores)
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)
    d = det.Det(tar=target, non=non_target)
    d.plot_det(label=system_name)
    d.plot_min_dcf(0.5)  # for sub Eval
    # d.plot_min_dcf(0.08)	# for Eval
    # d.plot_Pmiss_at_Pfa(0.5)
    print("+===============================================+")
    print("|SYSTEM: {}\t\t\t\t|".format(system_name))
    print("|	Best Threshold: {} with G-Mean: {}".format(min_value / thresholdOpt if min_value != 0 else thresholdOpt,
                                                           gmeanOpt))
    if overlap_score:
        print("|	Mean overlap score of the detection: ", np.mean(overlap_score), "	|")
    print("|	EER: ", d.eer(), "	|")
    print("|	Pmiss_at_Pfa0.1: ", d.Pmiss_at_Pfa(0.1), "	|")
    print("|	Pmiss_at_Pfa0.3: ", d.Pmiss_at_Pfa(0.3), "	|")
    print("|	Pmiss_at_Pfa0.5: ", d.Pmiss_at_Pfa(0.5), "	|")
    print("|	Pmiss_at_Pfa0.7: ", d.Pmiss_at_Pfa(0.7), "	|")
    print("|	Pmiss_at_Pfa0.9: ", d.Pmiss_at_Pfa(0.9), "	|")
    print("+===============================================+\n")
plt.legend()
plt.show()
