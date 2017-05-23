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
# Returns a tuple of pixels and error string (if applicable)
def URLtoPixels(url, width, height):
	start = time.clock()
	try:
		resp = requests.get(url, timeout=0.5)
	except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:  # Response timeout
		return (None, "Response timeout")
	if resp.status_code != 200:  # HTTP response error
		return (None, "HTTP response error")
	try:
		img = Image.open(StringIO(resp.content))
		img = img.resize((width, height), resample=Image.LANCZOS)
		pixels = list(img.getdata())
	except IOError as e:  # Invalid/corrupted image data
		return (None, "Invalid/unreadable image data")
	try:
		if len(pixels[0]) != 3:  # Not RGB image format
			return (None, "Pixels not RGB format")
	except TypeError as e:  # Not RGB image format
		return (None, "Pixels not RGB format")
	# Taken from http://stackoverflow.com/questions/1109422/getting-list-of-pixel-values-from-pil
	pixels = np.array([pixels[i * width:(i + 1) * width] for i in xrange(height)])
	return (pixels, None)

def saveData(outputBasePath, pixelData, subreddit_labels, nsfw_labels, subredditToIndex):
	with open(outputBasePath + "_subredditIndex", 'wb') as outfile:
		pickle.dump(subredditToIndex, outfile, protocol=2)
	with open(outputBasePath + "_subredditlabels", 'wb') as outfile:
		pickle.dump(subreddit_labels, outfile, protocol=2)
	with open(outputBasePath + "_nsfwlabels", 'wb') as outfile:
		pickle.dump(nsfw_labels, outfile, protocol=2)
	np.save(outputBasePath + "_data", pixelData)

def printSummary(num_lines, start_time, num_errors, errors):
	print "Processed {} posts...".format(num_lines)
	print "Time elapsed: {}s".format(time.clock() - start_time)
	print "Number of failed images so far: {}".format(num_errors)
	for error, count in errors.iteritems():
		print "# of {}: {}".format(error, count)
	print "-------------------------------------------------------"

# Converts URL's from data to pixel values
# inputFile is the name of a data file that consists of N rows (where N is the number of samples), 
# where each row is of the form [subreddit, image URL, number of upvotes]
# outputBasePath is the base path of the output files, which will also consist of N rows, where each row
# is of the form [subreddit, array of pixels, of shape width x height x 3]. Output files will be pixel data,
# labels, and subreddit to index mapping, each in their own file
def pixelfyData(inputFile, outputBasePath, size, startLine=0, verbose=False, cacheSubToIndFile=None):
	pixelData = []
	subreddit_labels = []
	nsfw_labels = []
	subredditToIndex = {}
	num_errors = 0
	errors = {}  # error frequency count, key is error string, value is count
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
			nsfw = rowdata[2]

			if subreddit not in subredditToIndex:
				subredditToIndex[subreddit] = len(subredditToIndex)
			pixels, error_msg = URLtoPixels(url, size, size)
			if pixels is None:
				num_errors += 1
				errors[error_msg] = errors.get(error_msg, 0) + 1
			else:
				pixelData.append(pixels)
				subreddit_labels.append(subredditToIndex[subreddit])
				if nsfw:
					nsfw_labels.append(1)
				else:
					nsfw_labels.append(0)

			if verbose and lineNum % 100 == 0:
				printSummary(lineNum, start, num_errors, errors)
			if lineNum % 1000 == 0:
				saveData(outputBasePath, np.array(pixelData), subreddit_labels, nsfw_labels, subredditToIndex)

	saveData(outputBasePath, np.array(pixelData), subreddit_labels, nsfw_labels, subredditToIndex)
	printSummary(lineNum, start, num_errors, errors)

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