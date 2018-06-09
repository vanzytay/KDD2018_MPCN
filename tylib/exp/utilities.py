from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from termcolor import colored

def printc(text, color):
    print(colored(text, color))

def print_args(args, path=None):
    ''' Print arguments to log file
    '''
    if path:
        output_file = open(path, 'w')
    args.command = ' '.join(sys.argv)
    items = vars(args)
    output_file.write('=============================================== \n')
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    output_file.write('=============================================== \n')
    if path:
        output_file.close()
    del args.command

def mkdir_p(path):
    ''' Makes path if path does not exist
    '''
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        pass
