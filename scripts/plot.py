import wave

import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import cm
from scipy import ndimage

import ArrayFromFeatures

phn_labels = ['a', 'a:', 'au', 'b', 'c', 'd', 'dZ', 'dz', 'e', 'e:',
              'eu', 'F', 'f', 'g', 'h_', 'i', 'i:', 'int', 'J',
              'J_', 'j', 'k', 'l', 'm', 'N', 'n', 'o', 'o:', 'ou',
              'P_', 'p', 'pau', 'r', 'S', 's', 'spk', 't', 'tS',
              'ts', 'u', 'u:', 'v', 'x', 'Z', 'z', 'oth']

diag_matrix = np.zeros((100, 100), np.int32)
# np.fill_diagonal(np.fliplr(a), 1)  # flip
np.fill_diagonal(diag_matrix, 1)


def Plot(feature1=None, feature2=None, dist=None, wp=None, sim_list=None, dtw_name="", info=[], gram_matrix=None,
         dtw_dist=None):
    feat_name = 'Features'
    filename1 = ''
    filename2 = ''
    if info != []:
        # tick1 = info[0][-1] / feature1.shape[0]
        # tick2 = info[1][-1] / feature2.shape[0]
        tick1 = 0.01
        tick2 = 0.01
        if info[0][-2] == "lin":
            feat_name = 'Phoneme posteriors'
        else:
            feat_name = 'MFCC'
        if len(info[0]) < 11:
            filename1 = info[0][0]
        else:
            filename1 = info[0][0] + ' ' + info[0][3]
        if len(info[0]) < 11:
            filename2 = info[1][0]
        else:
            filename2 = info[1][0] + ' ' + info[1][3]
    else:
        tick1 = 0.01
        tick2 = 0.01

    if feature1 is not None and feature2 is not None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('File 1: {}'.format(filename1))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(feat_name)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick1))
        ax.xaxis.set_major_formatter(ticks_x)
        ax.tick_params(axis='x', rotation=0)
        if feature1.shape[1] == 46:
            plt.yticks(np.arange(len(phn_labels)), phn_labels)
        features_mfcc1 = np.swapaxes(feature1, 0, 1)
        cax = ax.imshow(features_mfcc1, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')
        fig.colorbar(cax, ax=ax)

        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('File 2: {}'.format(filename2))
        ax.set_xlabel("Time [s]")
        # ax.set_ylabel(feat_name)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick2))
        ax.xaxis.set_major_formatter(ticks_x)
        ax.tick_params(axis='x', rotation=0)
        if (feature2.shape[1] == 46):
            plt.yticks(np.arange(len(phn_labels)), phn_labels)
        features_mfcc2 = np.swapaxes(feature2, 0, 1)
        cax = ax.imshow(features_mfcc2, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')
        fig.colorbar(cax, ax=ax, label="probability")

    if dist is not None and wp is not None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick2))
        ax.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * tick1))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.tick_params(axis='x', rotation=0)
        if gram_matrix is not None:
            cax = ax.imshow(gram_matrix, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='equal')
            fig.colorbar(cax, ax=ax, label="Gram matrix values")
        else:
            cax = ax.imshow(dist, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='equal')
            fig.colorbar(cax, ax=ax, label="distance cost")
        ax.set_xlabel("File 2 Time [s]")
        ax.set_ylabel("File 1 Time [s]")
        # fig.suptitle(dtw_name + ' DTW alignment path')
        ax.set_title("DTW distance: {0:.6f}".format(dtw_dist))

        ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='coral', linewidth=2.0)
        if sim_list is not None:
            color = iter(cm.rainbow(np.linspace(0, 1, len(sim_list))))
            for arr in sim_list:
                c = next(color)
                ax.plot(arr[:, 1], arr[:, 0], label='similarity', color=c, linewidth=4.0)  # , linestyle='dotted')
        ax.legend()
    plt.tight_layout()
    # plt.savefig('DTW.png')
    if (feature1 is not None and feature2 is not None) or (dist is not None and wp is not None):
        pass


def PlotPhnAudio(phn_posteriors=None, file=None, info=[]):
    feat_name = 'Features'
    filename = ''
    if info != []:
        # tick = info[0][-1] / phn_posteriors.shape[0]
        tick = 0.1
        if info[0][-2] == "lin":
            feat_name = 'Phoneme posteriors'
        else:
            feat_name = 'MFCC'
        if len(info[0]) < 11:
            filename = info[0][0]
        else:
            filename = info[0][0] + ' ' + info[0][3]
    else:
        tick = 0.01

    if phn_posteriors is not None:
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(2, 1, 1)
        ax.set_title('File 1: {}, repeat: {} times,'.format(filename, info[0][7]))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(feat_name)
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick))
        ax.xaxis.set_major_formatter(ticks_x)
        ax.tick_params(axis='x', rotation=20)
        if (phn_posteriors.shape[1] == 46):
            ax.yaxis.set_major_locator(plt.MaxNLocator(46))
            ax.yaxis.set_major_formatter(ticker.IndexFormatter(phn_labels))
        # labels = [item.get_text() for item in ax.get_yticklabels()]
        # labels[1] = 'Testing'
        # ax.set_yticklabels(labels)
        # plt.yticks(np.arange(len(phn_labels)), phn_labels)
        features_mfcc1 = np.swapaxes(phn_posteriors, 0, 1)
        cax = ax.imshow(features_mfcc1, interpolation='nearest', cmap=cm.nipy_spectral, origin='lower', aspect='auto')
        # ax2 = fig.add_subplot(2, 1, 2)
        spf = wave.open(file.replace("lin", "wav"), "r")
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, dtype='int16')
        fs = spf.getframerate()
        time = np.linspace(0, len(signal) / fs, num=len(signal))
        # ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * tick/100))
        # ax2.xaxis.set_major_formatter(ticks_x)
        # ax2.tick_params(axis='x', rotation=20)
        ax2.plot(time, signal)
        plt.axis(xmin=0, xmax=time[-1])


def PlotRQA(gram_matrix, score, path, reduction, original_shape, feature):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 5))
    # librosa.display.specshow(gram_matrix, x_axis='frames', y_axis='frames', ax=ax[0])
    ax[0].imshow(gram_matrix, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='equal')
    ax[0].set_title('Recurrence matrix')
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * reduction / 100))
    ax[0].xaxis.set_major_formatter(ticks_x)
    ax[0].yaxis.set_major_formatter(ticks_x)
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Time [s]")
    ax[0].tick_params(axis='both')
    ax[0].set(title='Recurrence matrix')
    # librosa.display.specshow(score, x_axis='frames', y_axis='frames', ax=ax[1])
    ax[1].imshow(score, interpolation='nearest', cmap=cm.gist_earth, origin='lower', aspect='equal')
    # ax[1].set(title='Alignment score matrix')
    ax[1].plot(path[:, 1], path[:, 0], label='Optimal path', color='c', linewidth=3.0)
    ax[1].legend()
    ax[1].set_title('RQA analysis')
    ax[1].label_outer()
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * reduction / 100))
    ax[1].xaxis.set_major_formatter(ticks_x)
    ax[1].yaxis.set_major_formatter(ticks_x)
    ax[1].tick_params(axis='both')
    ax[1].set_xlabel("Time [s]")
    fig.suptitle("Original frames count:{}\nReduced frames count:{}".format(original_shape, feature.shape[0]))


# print(len(path))
# print("SUM", np.sum(gram_matrix[path])/len(path))
# print("RQA SUM", np.sum(score[path])/len(path))


def gram_matrix(feature):
    matrix = feature.dot(
        feature.T)  # (feature.dot(feature.T)) / (np.linalg.norm(feature) * np.linalg.norm(feature.T))
    # np.fill_diagonal(np.flip(matrix), np.zeros((4, 4)))
    return 0.5 * (matrix + 1)


# return feature


def image_filter(matrix, threshold=0.7, percentile=70, variance=5):
    matrix[matrix < threshold] = 0.0
    matrix[matrix >= threshold] = 1.0
    matrix = ndimage.percentile_filter(matrix, percentile=percentile, footprint=diag_matrix, mode='constant', cval=0.0)
    matrix = ndimage.gaussian_filter(matrix, variance)
    # np.put(matrix, np.diag(np.flip(matrix)), 0.0)
    # np.fill_diagonal(np.flip(matrix), 0.0)
    # np.fill_diagonal(np.flip(matrix[:,1:]), 0.0)
    return matrix


if __name__ == "__main__":
    file1 = "path_to_file.lin/fea/wav"
    file2 = "path_to_file.lin/fea/wav"
    reduce_dimension = True
    feature_type = 'posteriors'  # 'mfcc', 'posteriors', 'bottleneck'
    feature1 = ArrayFromFeatures.GetArray(file1, feature_type, reduce_dimension)
    feature2 = ArrayFromFeatures.GetArray(file2, feature_type, reduce_dimension)
    parsed1 = ArrayFromFeatures.Parse(file1)
    parsed2 = ArrayFromFeatures.Parse(file2)

    # librosa DTW
    cost_matrix, wp = librosa.sequence.dtw(X=feature1.T, Y=feature2.T, metric='euclidean', subseq=False,
                                           global_constraints=False,
                                           weights_mul=np.array([np.sqrt([2]), 1, 1], dtype=np.float64))
    dtw_dist = cost_matrix[wp[-1, 0], wp[-1, 1]]
    print(dtw_dist)
    # Fast DTW
    '''
	cost_matrixFast, wp = fastdtw(feature1, feature2, dist=euclidean)
	dtw_dist = cost_matrixFast/(feature1.shape[0]+feature2.shape[0])
	wp = np.asarray(wp)
	sim_list = Similarity(wp, [0.9, 1.1])[0]
	print(sim_list)
	print(dtw_dist)
	'''

    # sdtw
    '''
	reduction = 5
	feature1 = ArrayFromFeatures.ReduceFrames(feature1, size=reduction)
	feature2 = ArrayFromFeatures.ReduceFrames(feature2, size=reduction)
	path = sdtw.segmental_dtw(feature1, feature2, R=4, L=200, dist='euclidean')
	dtw_dist = path[0]
	wp = np.asarray(path[1][3])*reduction
	print(dtw_dist)
	'''

    # own implementation of image filter
    '''
	gram_matrix = gram_matrix(feature1)
	gram_matrix = image_filter(gram_matrix)
	'''

    # RQA analysis
    '''
	reduction = 5
	original_shape = feature1.shape[0]
	# feature1_orig = feature1
	feature1 = ArrayFromFeatures.ReduceFrames(feature1, size=reduction)
	gram_matrix = librosa.segment.recurrence_matrix(feature1.T,width=40//reduction, k=feature1.shape[0]//reduction,
	                                        mode='affinity',	#['connectivity', 'distance', 'affinity']
	                                        metric='cosine')
	score, path = librosa.sequence.rqa(gram_matrix, np.inf, np.inf, knight_moves=True)
	'''
    sim_list = None
    gram_matrix = None
    # Plot(feature1, feature2, cost_matrix, wp, sim_list, dtw_name="", info=[parsed1, parsed2],
	# gram_matrix=gram_matrix, dtw_dist=dtw_dist)
    PlotPhnAudio(feature1, file=file1, info=[parsed1])
    # PlotRQA(gram_matrix, score, path, reduction, original_shape, feature1)
    plt.show()
