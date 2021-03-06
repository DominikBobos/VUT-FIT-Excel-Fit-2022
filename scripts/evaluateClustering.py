"""
File : evaluateClustering.py
Author : Dominik Boboš
Description : External evaluation of reference clusters to predicted clusters (computing purity, Rand Index, NMI)
"""
import operator

import numpy as np
from sklearn import metrics

import ArrayFromFeatures


def DivideClasses(cluster_list):
	"""
	find dominance of files ID in cluster and name the corresponding cluster class according to dominant ID
	:param cluster_list: clusters
	:return: labelled dictionary of clusters with labels
	"""
	message_ids = []
	total_ids_count = {}
	clusters_classes = {}
	for idx, cluster_item in enumerate(cluster_list):
		predominance_ids = {}
		for file in cluster_item:
			parsed_file = ArrayFromFeatures.Parse(file[0])
			if len(parsed_file) < 6:	# no message 
				ID = "X00"
			else:
				ID = parsed_file[3]		# message ID
			if ID in predominance_ids:
				predominance_ids[ID] += 1
			else:
				predominance_ids[ID] = 1	# new type
			if ID in total_ids_count:
				total_ids_count[ID] += 1
			else:
				total_ids_count[ID] = 1
		dominance_ID = max(predominance_ids.items(), key=operator.itemgetter(1))[0]
		clusters_classes[idx] = dominance_ID
	return clusters_classes, total_ids_count

def ConvertsIdsToInt(id_dict):
	"""
	convert the actual IDs to pure integers to make evaluation possible
	:param id_dict: dictionary of clusters with labels
	:return: mapping dictionary to real label ID to integer version
	"""
	id_labels = {}
	for idx in range(len(id_dict)):
		id_name = id_dict[idx]
		if id_name in id_labels:
			continue
		else:
			id_labels[id_name] = len(id_labels)
	return id_labels


def AddLabelsToFiles(cluster_list_true, cluster_list_pred, id_dict_true, id_dict_pred, id_labels):
	"""
	label files from clusters by integer IDs
	:param cluster_list_true: reference clusters
	:param cluster_list_pred: predicted clusters
	:param id_dict_true: dict of real ID classes of the reference cluster
	:param id_dict_pred: dict of predicted ID classes of the predicted cluster
	:param id_labels: mapping dictionary to real labels to their integer version
	:return: list of reference(true) labels and predicted labels
	"""
	true_labels = []
	pred_labels = []
	file_dict = {}
	for idx, cluster in enumerate(cluster_list_true):
		for cluster_item in cluster:
			integer_label = id_labels[id_dict_true[idx]]
			file_dict[cluster_item[0].split('/')[-1]] = [integer_label]
	for idx, cluster in enumerate(cluster_list_pred):
		for cluster_item in cluster:
			try:
				integer_label = id_labels[id_dict_pred[idx]]
				file_dict[cluster_item[0].split('/')[-1]].append(integer_label)
			except KeyError:
				pass # having something more
	for filename, label in file_dict.items():
		if len(label) == 2:
			true_labels.append(label[0])
			pred_labels.append(label[1])
	return true_labels, pred_labels


def ComputePurity(y_true, y_pred):
	"""
	compute purity
	:param y_true: reference cluster labels
	:param y_pred: predicted cluster labels
	:return: purity score
	"""
	# compute contingency matrix (also called confusion matrix)
	contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
	return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def GetLabels(pkl_clust_true, pkl_clust_pred):
	"""
	get labels for clusters to evaluation metrics
	:param pkl_clust_true: reference cluster labels
	:param pkl_clust_pred: predicted cluster labels
	:return: list of reference labels, list of predicted cluster labels
	"""
	cluster_list_true = ArrayFromFeatures.OpenPickle(pkl_clust_true)
	cluster_list_pred = ArrayFromFeatures.OpenPickle(pkl_clust_pred)
	id_dict_true, total_ids_count_true = DivideClasses(cluster_list_true)
	id_dict_pred, total_ids_count_pred = DivideClasses(cluster_list_pred)
	id_labels = ConvertsIdsToInt(id_dict_true)
	true_labels, pred_labels = AddLabelsToFiles(cluster_list_true, cluster_list_pred, id_dict_true, id_dict_pred, id_labels)
	return true_labels, pred_labels


def Evaluate(labels_true, labels_pred):
	"""
	compute Purity, Rand Index, NMI
	:param labels_true:
	:param labels_pred:
	:return:
	"""
	purity = ComputePurity(labels_true, labels_pred)
	rand_score = metrics.rand_score(labels[0], labels[1])
	n_mutual_information = metrics.normalized_mutual_info_score(labels_true, labels_pred) 
	return purity, rand_score, n_mutual_information
	


if __name__ == "__main__":
	import sys
	labels = GetLabels(sys.argv[1], sys.argv[2])
	purity, rand_score, n_mutual_information = Evaluate(labels[0], labels[1])
	print("+=======================================================================+")
	print("|Golden standard cluster:\t{}\t|".format(sys.argv[1].split('/')[-1]))
	print("|Predicted cluster:\t\t{}\t|".format(sys.argv[2].split('/')[-1]))
	print("|	Purity: ", purity, "\t\t\t\t\t|")
	print("|	Rand score: ", rand_score, "\t\t\t\t|")
	print("|	NMI: ", n_mutual_information, "\t\t\t\t\t|")
	print("+=======================================================================+\n")
	print("")
