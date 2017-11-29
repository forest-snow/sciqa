# build wiki summaries for all answers
# see README in data folder

import json
import mwclient
from csv import DictReader, DictWriter


if __name__ == "__main__":

    # Cast to list to keep it all in memory
    answer_pool = json.load(open('../data/quizbowl/answer_set.json', 'r'))
    site = mwclient.Site('en.wikipedia.org')

    # wiki summaries of all answers
    wiki_data = []
    for index, title in enumerate(answer_pool):
        page = {}
        doc = site.pages[title]
        page['id'] = index
        page['title'] = title
        # get categories
        categories = []
        for category in doc.categories(generator=False):
            category = category.split(':')[1]
            if not category.startswith('Wikipedia'):
                categories.append('_'.join(category.split()))
        page['categories'] = categories
        # get links
        links = []
        for link in doc.links(generator=False):
            if ':' not in link:
                links.append('_'.join(link.split()))
        page['links'] = links

        wiki_data.append(page)

    keys = wiki_data[0].keys()
    with open('wiki_data.csv', 'w', encoding='utf-8') as output_file:
        dict_writer = DictWriter(output_file, keys, extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(wiki_data)
