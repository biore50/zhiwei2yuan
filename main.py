#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
# from torchlight import import_class
from processor.recognition import REC_Processor
# from processor.demo import Demo
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')


    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = REC_Processor()
    processors['demo'] = REC_Processor()
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

        # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors['recognition']
    p = REC_Processor()

    p.start()




