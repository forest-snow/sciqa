import argparse
import json
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='options')
	parser.add_argument('--input_file', type=str, required=True,
		help="Location of the input file (.json)")
	parser.add_argument('--output_name', type=str, required=True,
		help="Name of the output file(s)")

	args = parser.parse_args()

	data = json.load(open(args.input_file, 'r'))

	with open(args.output_name + '.txt', 'w') as f:
		for line in data:
			text = line['text'].replace('\n', '')
			for sentence in sent_tokenize(text):
				f.write(sentence + '\n')
