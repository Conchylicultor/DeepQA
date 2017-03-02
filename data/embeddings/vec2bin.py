#!/usr/bin/python

import sys
import getopt
import numpy as np

from tqdm import tqdm

input_path = 'wiki.fr.vec'
output_path = 'wifi.fr.bin'

def vec2bin(input_path, output_path):
    input_fd  = open(input_path, "rb")
    output_fd = open(output_path, "wb")

    header = input_fd.readline()
    output_fd.write(header)

    vocab_size, vector_size = map(int, header.split())

    for line in tqdm(range(vocab_size)):
        word = []
        while True:
            ch = input_fd.read(1)
            output_fd.write(ch)
            if ch == b' ':
                word = b''.join(word).decode('utf-8')
                break
            if ch != b'\n':
                word.append(ch)
        vector = np.fromstring(input_fd.readline(), sep=' ', dtype='float32')
        output_fd.write(vector.tostring())

    input_fd.close()
    output_fd.close()


def main(argv):
   inputfile = False
   outputfile = False
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('vec2bin.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg

   if not inputfile or not outputfile:
       print('vec2bin.py -i <inputfile> -o <outputfile>')
       sys.exit(2)

   print('Converting %s to binary file format' % inputfile)
   vec2bin(inputfile, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])
