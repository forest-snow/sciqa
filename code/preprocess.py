import argparse
import json

def process_question(text, answer_pool):
	for answer in answer_pool:
		sub = answer.replace('_', ' ')
		text = text.replace(sub, answer)
	return text

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='options')
	parser.add_argument('--text_file', type=str, required=True,
		help="Location of the text file (.json)")
	parser.add_argument('--answer_set', type=str, required=True,
		help="Location of the answer set (.json)")
	parser.add_argument('--outfile', type=str, required=True,
		help="Location of the output file (.json)")

	args = parser.parse_args()
	data = json.load(open(args.text_file, 'r'))
	answer_pool = json.load(open(args.answer_set, 'r'))

	# filter out unigrams
	answer_set = set()
	for answer in answer_pool:
		if '_' in answer:
			answer_set.add(answer)

	for line in data:
		line['text'] = process_question(line['text'], answer_set)

	with open(args.outfile, 'w') as outfile:
		json.dump(line, outfile)
