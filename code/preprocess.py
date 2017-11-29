import argparse
import json

from csv import DictReader

def process_question(text, answer_pool):
	for answer in answer_pool:
		sub = answer.replace('_', ' ').strip()
		text = text.replace(sub, answer)
	return text

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='options')
	parser.add_argument('--text_file', type=str, required=True,
		help="Location of the text file (.json)")
	parser.add_argument('--wiki_data', type=str, required=True,
		help="Location of the wiki data (.json)")
	parser.add_argument('--outfile', type=str, required=True,
		help="Location of the output file (.json)")

	args = parser.parse_args()
	data = json.load(open(args.text_file, 'r'))
	wiki_data = list(DictReader(open(args.wiki_data, 'r')))

	# filter out unigrams
	entities = set()
	for wiki in wiki_data:
		link_list = wiki['links'][2:-2].split('\', \'')
		if '_' in wiki['title']:
			entities.add(wiki['title'])
		for link in link_list:
			if '_' in link and '(' not in link:
				entities.add(link)

	print(len(entities))
	for line in data:
		line['text'] = process_question(line['text'], entities)

	with open(args.outfile, 'w') as outfile:
		json.dump(data, outfile)
