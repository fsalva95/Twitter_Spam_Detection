from nltk.stem.porter import *

plurals = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 'traditional', 'reference', 'colonizer','plotted']

stemmer = PorterStemmer()

singles = [stemmer.stem(plural) for plural in plurals]

print(' '.join(singles))
