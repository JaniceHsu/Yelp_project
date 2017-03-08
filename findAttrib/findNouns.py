import nltk
#from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize
from nltk.tokenize import StanfordTokenizer
from nltk import ne_chunk
import nltk.data
import os


#required java installation
os.environ['JAVAHOME'] = "C:/Program Files/Java/jdk1.8.0_31/bin"
#sentence tokenizer
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
#stanford tokenizer to comply with stanford taggers
tokenizer = StanfordTokenizer('../stanford-postagger/stanford-postagger.jar')
#st = StanfordNERTagger('../stanford_ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
#					   '../stanford_ner/stanford-ner.jar',
#					   encoding='utf-8')


sp = StanfordPOSTagger('../stanford-postagger/models/english-bidirectional-distsim.tagger',
					   '../stanford-postagger/stanford-postagger.jar',
					   encoding='utf-8')

Ball_text = """Kobe Bean Bryant (born August 23, 1978) is an American retired professional basketball player. He played his entire 20-year career with the Los Angeles Lakers of the National Basketball Association (NBA). He entered the NBA directly from high school and won five NBA championships with the Lakers. Bryant is an 18-time All-Star, 15-time member of the All-NBA Team, and 12-time member of the All-Defensive team."""

sentToken = sent_detector.tokenize(Ball_text) #separate into sentences

tokenized_text = tokenizer.tokenize(Ball_text)
POS_text = sp.tag(tokenized_text)


print("END")