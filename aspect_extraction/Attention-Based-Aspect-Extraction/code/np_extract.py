from textblob import TextBlob, Word
from preprocess import parseTextBlob, load_sentiment_terms


text = 'We arrived late afternoon on Friday, unfortunately we missed the game drive, but were welcomed firstly by the beautiful Nyala buck that live in the trees surrounding the lodge and then by the staff with friendly smiles and warm welcomes,  drinks were offered and a spectacular view of the reserve and a tour of the main lodge area.'
wiki = TextBlob(text)
print(wiki.tags)
print(wiki.noun_phrases)

terms = ['smiles', 'view', 'trees', 'spectacular', 'drive', 'drinks', 'area', 'welcomes', 'staff', 'Friday', 'tour', 'friendly']
tags = wiki.tags

words = wiki.words
print(words)
print(Word("smiles"))

noun_tags = ['NN', 'NNP']
b = [x.correct() for x,y in wiki.tags if y not in noun_tags]
print(b)

b = wiki.correct()
print('corrected:', b)

lemas = ' '.join(Word(w).lemmatize() for w in b.words)
print(lemas)


sent = load_sentiment_terms('sentiments.txt')
print('========================================')
b = parseTextBlob(text, sent)
print(b)

print('='*20)
text1 = "The patriach was not interested in our vehicles being any where near his herd and so he was heading straight for us, ears waving, but what a beautiful sight to behold, truly God's Country!"
print(parseTextBlob(text1, sent))

