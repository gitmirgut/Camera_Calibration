# Diese CameraCalibration funktioniert mittels einer moeglichkeit des
# downscalings der Bilder. Die Bilder werden Schrittweise verkleinert
# und dann jeweils nach dem Schachbrettmuster gesucht
# Anschließend werden daraus die nötigen Daten für die Calibrierung errechnet
import argparse
from tools import calibrationtools
import numpy as np
import os
import shelve
from tools import imgtools



def calibrate(args):
    calibrationtools.calibrate(args.s, args.path, args.width, args.height, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='BeesBook camera calibration processor.',
        description='Determines the camera parameters for image rectification'
    )
    subparsers = parser.add_subparsers(
        title='subcommands',
        description='The programm includes differnet functions for the '
        ' rectification of images.',
        help='The following functions are included:'
    )

    parser_calibrate = subparsers.add_parser(
        'calib',
        help='finds the camera parameters/calibrates the camera'
    )

    parser_calibrate.add_argument(
        '-s',
        help='automatic scale of images',
        action='store_true'
    )

    parser_calibrate.add_argument(
        'path',
        help='path of the images with the pattern',
        type=str
    )

    parser_calibrate.add_argument(
        'width',
        help='width of the chessboard pattern',
        type=int
    )

    parser_calibrate.add_argument(
        'height',
        help='height of the chessboard pattern',
        type=int
    )

    parser_calibrate.add_argument(
        'output',
        help='output path of the camera parameters',
        type=str
    )

    parser_calibrate.set_defaults(func=calibrate)

    args = parser.parse_args()
    args.func(args)
