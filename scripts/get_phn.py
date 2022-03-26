"""
File : get_phn.py
Author : Dominik Boboš
Description : This script extracts features from wav files by using third party utilities
"""
import argparse  # for parsing arguments
import glob  # for file path, finding files, etc
import os
import sys
from subprocess import call

"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Arguments parsing
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
parser = argparse.ArgumentParser()
parser.add_argument("--src", help="source directory path")  # directory to import
parser.add_argument("--dst", help="destination directory path")  # directory to export
parser.add_argument("--str", action="store_true", help="Export phonemes to txt")
parser.add_argument("--phn", action="store_true", help="Export phoneme posteriors")
parser.add_argument("--lat", action="store_true", help="Export lattices from phonemes posteriors")
parser.add_argument("--bnf", action="store_true", help="Export bottle-neck features")
parser.add_argument("--resample", action="store_true", help="Resample to sampling rate 8kHz while preserving speed")
parser.add_argument("--posttostrlat", action="store_true", help="Resample to sampling rate 8kHz while preserving speed")
arguments = parser.parse_args()

"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Script paths 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
phnrec = "set_phnrec" 
bnf = "third_party_scripts/BUT-Phonexia-BottleneckFeatureExtractor_20180301/audio2bottleneck.py"
lattice = "third_party_scripts/lattice_generation/lattice.sh"

if not arguments.src:
    src = os.getcwd()
else:
    if os.path.isdir(arguments.src):
        src = arguments.src
    else:
        sys.stderr.write("Invalid source directory.\n")
        sys.exit(1)
if not arguments.dst:
    dst = os.getcwd() + '/'
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


def get_posteriors():
    files = glob.glob(src + '*.wav')
    if not os.path.isfile("set_phnrec"):
        sys.stderr.write("set_phnrec script is not found")
        sys.exit(1)
    is_initialized = False
    for idx, file in enumerate(files):
        src_file = file
        # src_file = src_file.replace(" ", "\\ ") #same goes for '(', ')'
        print("{}/{} (file: {})".format(idx, len(files), file.split('/')[-1]))
        if arguments.str:
            dst_file = dst + file.split('/')[-1].replace('.wav', '.txt')
            if os.path.exists(dst_file):
                continue
            call(""". {}
phnrec -c $phnrecdir/PHN_CZ_SPDAT_LCRC_N1500/ -w lin16 -t {} -i "{}" -o "{}"
			""".format(phnrec, "str", src_file, dst_file), shell=True)
        elif arguments.phn:
            dst_file = dst + file.split('/')[-1].replace('.wav', '.lin')
            if os.path.exists(dst_file):
                print("File already exists")
                continue
            call(""". {}
phnrec -c $phnrecdir/PHN_CZ_SPDAT_LCRC_N1500/ -w lin16 -t {} -i "{}" -o "{}"
			""".format(phnrec, "post", src_file, dst_file), shell=True)
        elif arguments.resample:
            if os.path.exists(dst + file.split('/')[-1]):
                print("File already exists")
                continue
            fs = 8000
            speed = src_file.split('S(')[-1][:4]
            print(speed)
            call('ffmpeg -i "{}" -af "asetrate={},atempo={}" "{}"'.format(
                src_file, fs, speed, dst + file.split('/')[-1]), shell=True)
        elif arguments.bnf:
            dst_file = dst + file.split('/')[-1].replace('.wav', '.fea')
            nn_weights = "FisherMono"  # "FisherTri" "BabelMulti"
            if os.path.exists(dst_file):
                print("File already exists")
                continue
            if os.path.exists("/media/dominik/2TB_ADATAHD330/IBT/train/train_clear/train_clear_bnf/" +
                              file.split('/')[-1].replace('.wav', '.fea')):
                print("File already exists in train_clear_bnf")
                continue
            if os.path.exists(
                    "/media/dominik/2TB_ADATAHD330/IBT/eval/eval_clear/eval_clear_bnf/" + file.split('/')[-1].replace(
                        '.wav', '.fea')):
                print("File already exists in eval_clear_bnf")
                continue
            if os.path.exists(
                    "/media/dominik/2TB_ADATAHD330/IBT/cut_edited/cut_bnf/" + file.split('/')[-1].replace('.wav',
                                                                                                          '.fea')):
                print("File already exists in cut_bnf")
                continue
            call("python3 {} '{}' '{}' '{}'".format(bnf, nn_weights, src_file, dst_file), shell=True)
        elif arguments.lat:
            if os.path.exists(dst + '/lattice/' + file.split('/')[-1].replace('.wav', '.latt')):
                print("File already exists")
                continue
            if not is_initialized:
                call("""cd {}
./post_to_lattice_init.sh '{}'
			""".format(lattice.split('lattice.sh')[0], dst), shell=True)
                is_initialized = True
            # edit model config
            call("cp /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config.gmm /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config",
                 shell=True)
            call("""cd {}
./lattice.sh '{}' '{}' '{}'
			""".format(lattice.split('lattice.sh')[0], src_file, dst, file.split('/')[-1][:-4]), shell=True)
            call("cp /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config.backup /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config",
                 shell=True)
        else:
            print("Please select wanted operation. Use --help for details.")


if arguments.posttostrlat:
    files = glob.glob(src + '/' + '*.lin')
    is_initialized = False
    for idx, file in enumerate(files):
        src_file = file
        print("{}/{} (file: {})".format(idx, len(files), file.split('/')[-1]))
        if os.path.exists(dst + '/' + file.split('/')[-1].replace('.lin', '.latt')):
            print("File already exists")
            continue
        # edit model config
        call("cp /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config.gmm /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config", shell=True)

        if not is_initialized:
            call("""cd {}
./post_to_lattice_init.sh '{}'
		""".format(lattice.split('lattice.sh')[0], dst), shell=True)
            is_initialized = True

        call("""cd {}
./post_to_lattice.sh '{}' '{}' '{}'
		""".format(lattice.split('lattice.sh')[0], src_file[:-4], dst, file.split('/')[-1][:-4]), shell=True)
        call("cp /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config.backup /etc/PhnRec/PHN_CZ_SPDAT_LCRC_N1500/config",
             shell=True)

else:
    get_posteriors()
