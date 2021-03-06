"""
File : split.py
Author : Dominik Boboš
Description : Splits the recording to smaller segments
"""

import argparse  # for parsing arguments
import glob  # for file path, finding files, etc
import math  # for rounding numbers
import os
import shutil  # for copying files
import sys
from collections import Counter  # for statistics

from pydub import AudioSegment  # for splitting audio files + ffmpeg is needed


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Arguments parsing
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
parser = argparse.ArgumentParser()
parser.add_argument("--sec",
                    help="duration in seconds of the wanted cut length")  # how long cuttings we want in seconds
parser.add_argument("--src", help="source directory path")  # directory to import
parser.add_argument("--dst", help="destination directory path")  # directory to export
parser.add_argument("--stats", action="store_true",
                    help="show statistics about the wav files in either source or destination directory")
parser.add_argument("--lt",
                    help="Will cut longer recordings than specified in seconds")  # shows/splits audio longer than
# specified in seconds
parser.add_argument("--st",
                    help="Will cut shorter recordings than specified in seconds")  # shows/splits audio shorter than
# specified in seconds
parser.add_argument("--rm", action="store_true", help="Will remove the original file after splitting it")
parser.add_argument("--copy", action="store_true", help="Copy files to the specified folder")
parser.add_argument("--move", action="store_true", help="Move files to the specified folder")
parser.add_argument("--count", help="How many files you want to move or copy")
arguments = parser.parse_args()

if not arguments.sec and not arguments.stats and not arguments.copy and not arguments.move:
    sys.stderr.write("Please specify the duration in seconds of the wanted cut length using --sec=NUMBER .\n")
    sys.exit(1)
if arguments.sec:
    splitSec = int(arguments.sec)
if not arguments.src:
    src = os.getcwd()
else:
    if os.path.isdir(arguments.src):
        src = arguments.src
    else:
        sys.stderr.write("Invalid source directory.\n")
        sys.exit(1)
if not arguments.dst:
    dst = os.getcwd()
else:
    if os.path.isdir(arguments.dst):
        dst = arguments.dst
    else:
        try:
            os.mkdir(arguments.dst)  # create destination directory
            dst = arguments.dst
        except:
            sys.stderr.write("Invalid destination directory.\n")
            sys.exit(1)


def get_duration(file):
    """
    Gets duration in seconds from the given audio file
    :param file:
    :return:
    """
    return AudioSegment.from_wav(file).duration_seconds


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Partial splitting of the given audio file
Basically it cuts the given file from time fromSec to 
toSec and export it to the destination directory dst 
with given filename splitFilename.
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""


def single_split(fromSec, toSec, splitFilename, file):
    """
    Partial splitting of the given audio file
    Basically it cuts the given file from time fromSec to
    toSec and export it to the destination directory dst
    with given filename splitFilename.
    :param fromSec: cut from fromSec [in seconds]
    :param toSec: cut to toSec [in seconds]
    :param splitFilename: naming by numbering splitted segments
    :param file: audio file to split
    :return:
    """
    t1 = fromSec * 1000
    t2 = toSec * 1000
    splitAudio = AudioSegment.from_wav(file + '.wav')[t1:t2]
    splitFilename = splitFilename.split("/")[-1]
    splitAudio.export(dst + splitFilename, format="wav")


def multiple_split(splitSec):
    """
    Splits all .wav files located in source directory src.
    Each segment is splitSec long - only the last segment is shorter
    than splitSec (leftover).
    :param splitSec: duration of each segment in seconds
    :return:
    """
    files = glob.glob(src + '*.wav')
    for file in files:
        order = 0
        # filter out unwanted files
        if arguments.lt and int(arguments.lt) > get_duration(file):
            continue
        if arguments.st and int(arguments.st) < get_duration(file):
            continue
        try:  # test if the current file was processed before to get the order
            file.split('_')
            order = int(file.split('_')[1])
        except:
            pass
        totalSec = get_duration(file)  # total duration
        try:
            file.split('.')[2]  # test if the current file was processed before to get the right filename
            file = file.split('.')[0] + '.' + file.split('.')[1]
        except:
            file = file.split('.')[0]
        # cut the file
        for i in range(0, math.ceil(totalSec), splitSec):
            if i >= math.ceil(totalSec) - splitSec:
                seconds = get_duration(file + '.wav')
                seconds = seconds - i
                splitFilename = file.split('_')[0] + '_' + str(order + i // splitSec) + '_' + "{:.2f}".format(
                    seconds) + ".wav"
            else:
                splitFilename = file.split('_')[0] + '_' + str(order + i // splitSec) + '_' + str(splitSec) + ".wav"
            single_split(i, i + splitSec, splitFilename, file)
            print(splitFilename.split('/')[-1], ' : ', str(i // splitSec) + ' Done')
            if i >= math.ceil(totalSec) - splitSec:
                print('Splited successfully\n')
        if arguments.rm and get_duration(file + '.wav') > splitSec:
            print('Removing file: ', file + ".wav")
            os.remove(file + '.wav')


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Counts the total count of all files
counts the total duration of all files
orders and counts the occurrences of files
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""


def count(files):
    """
    Counts the total count of all files
    counts the total duration of all files
    orders and counts the occurrences of files
    :param files: list of files
    :return: total duration, histogram, count of all files
    """
    statList = []
    total = 0
    count = 0
    for file in files:
        if arguments.lt and int(arguments.lt) > get_duration(file):
            continue
        elif arguments.st and int(arguments.st) < get_duration(file):
            continue
        else:
            current = get_duration(file)
            count += 1
            total += current
            statList.append(current)
    stat = Counter(statList)
    return total, stat, count


def stats():
    """
    Shows statistics about audio files in a given directory
    """
    srcFiles = glob.glob(src + '*.wav')
    dstFiles = glob.glob(dst + '*.wav')
    if len(srcFiles) != 0:
        srcTotal, srcStat, srcCount = count(srcFiles)
        print("____________________________________________________________________________________")
        print("\nTotal count of all source recordings:  \t", srcCount)
        print("Total length of source records: \t", "{:.2f}".format(srcTotal), "[s] |",
              "{:.2f}".format(srcTotal / 60), "[m] |", "{:.2f}".format(srcTotal / 3600), "[h]", "\n")
        for item in srcStat.most_common():
            print("Records lengths [sec]: \t", item[0], "  \t Count: ", item[1])
        print("____________________________________________________________________________________\n")
    if len(dstFiles) != 0:
        dstTotal, dstStat, dstCount = count(dstFiles)
        print("____________________________________________________________________________________")
        print("\nTotal count of destination recordings:  \t", dstCount)
        print("Total length of destination records: \t", "{:.2f}".format(dstTotal), "[sec] |",
              "{:.2f}".format(dstTotal / 60), "[min] |", "{:.2f}".format(dstTotal / 3600), "[h]", "\n")
        for item in dstStat.most_common():
            print("Records lengths [sec]: \t", item[0], "  \t Count: ", item[1])
        print("____________________________________________________________________________________\n")
    if len(srcFiles) == 0 and len(dstFiles) == 0:
        print("No statistics available.")


def move_copy():
    """
    Copies or moves files from src to dst
    """
    srcFiles = glob.glob(src + '*.wav')
    i = 0
    for file in srcFiles:
        # filter out unwanted files
        if arguments.count and int(arguments.count) == i:
            break
        if arguments.lt and int(arguments.lt) > get_duration(file):
            continue
        if arguments.st and int(arguments.st) < get_duration(file):
            continue
        if arguments.copy:
            shutil.copy2(file, dst)
            print('Copying file: ', file, "to: ", dst)
            if arguments.rm:
                print('Removing file: ', file)
                os.remove(file)
        if arguments.move:
            shutil.move(file, dst)
            print('Moving file: ', file, "to: ", dst)
        i += 1


if arguments.move or arguments.copy:
    move_copy()
elif not arguments.stats:
    multiple_split(splitSec)
elif arguments.stats:
    stats()
