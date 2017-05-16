from PIL import Image
from StringIO import StringIO
import requests
import argparse
import cPickle as pickle
import json
import numpy as np
import time
import sys

# Takes in a URL to an image
def URLtoPixels(url, width, height):
	start = time.clock()
	try:
		resp = requests.get(url, timeout=0.5)
	except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:  # Response timeout
		return None
	if resp.status_code != 200:  # HTTP response error
		return None
	try:
		img = Image.open(StringIO(resp.content))
		img = img.resize((width, height), resample=Image.LANCZOS)
		pixels = list(img.getdata())
	except IOError as e:  # Invalid/corrupted image data
		return None
	try:
		if len(pixels[0]) != 3:  # Not RGB image format
			return None
	except TypeError as e:  # If pixels is not a valid array
		return None
	# Taken from http://stackoverflow.com/questions/1109422/getting-list-of-pixel-values-from-pil
	pixels = np.array([pixels[i * width:(i + 1) * width] for i in xrange(height)])
	return pixels

def saveData(outputBasePath, pixelData, labels, subredditToIndex):
	with open(outputBasePath + "_subredditIndex", 'wb') as outfile:
		pickle.dump(subredditToIndex, outfile, protocol=2)
	with open(outputBasePath + "_labels", 'wb') as outfile:
		pickle.dump(labels, outfile, protocol=2)
	np.save(outputBasePath + "_data", pixelData)

# Converts URL's from data to pixel values
# inputFile is the name of a data file that consists of N rows (where N is the number of samples), 
# where each row is of the form [subreddit, image URL, number of upvotes]
# outputBasePath is the base path of the output files, which will also consist of N rows, where each row
# is of the form [subreddit, array of pixels, of shape width x height x 3]. Output files will be pixel data,
# labels, and subreddit to index mapping, each in their own file
def pixelfyData(inputFile, outputBasePath, size, startLine=0, verbose=False, cacheSubToIndFile=None):
	pixelData = []
	labels = []
	subredditToIndex = {}
	numErrors = 0
	lineNum = 0

	if cacheSubToIndFile:
		infile = open(cacheSubToIndFile, 'r')
		subredditToIndex = pickle.load(infile)

	start = time.clock()
	with open(inputFile, 'r') as infile:
		for line in infile:
			lineNum += 1
			if lineNum <= startLine:
				continue
			rowdata = json.loads(line)
			subreddit = rowdata[0]
			url = rowdata[1]

			if subreddit not in subredditToIndex:
				subredditToIndex[subreddit] = len(subredditToIndex)
			pixels = URLtoPixels(url, size, size)
			if pixels is None:
				numErrors += 1
			else:
				pixelData.append(pixels)
				labels.append(subredditToIndex[subreddit])

			if verbose and lineNum % 100 == 0:
				print "Downloaded {} images...".format(lineNum)
				print "Time elapsed: {}s".format(time.clock() - start)
				print "Number of failed images so far: {}".format(numErrors)
			if lineNum % 1000 == 0:
				saveData(outputBasePath, np.array(pixelData), labels, subredditToIndex)

	saveData(outputBasePath, np.array(pixelData), labels, subredditToIndex)
	print "Number of total failed image downloads: {}".format(numErrors)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Converts URL data to pixel data')
	parser.add_argument('input', help="input filepath")
	parser.add_argument('output', help="output filepath")
	parser.add_argument('-s', '--size', type=int, default=128, help="image size")
	parser.add_argument('-v', '--verbose', action='store_true', default=False, help="show progress")
	parser.add_argument('-sl', '--startline', type=int, default=0, help="line number to start at")
	parser.add_argument('-c', '--cachefile', help="subredditToIndex cache filepath")

	args = parser.parse_args()

	pixelfyData(args.input, args.output, args.size, args.startline, args.verbose, args.cachefile)