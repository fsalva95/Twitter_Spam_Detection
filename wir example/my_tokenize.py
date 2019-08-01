from nltk.tokenize import word_tokenize, sent_tokenize
 
s = "Good muffins cost $3.88\nin New York.  Please buy me two of them.\n\nThanks."

print(word_tokenize(s))

print(sent_tokenize(s))
