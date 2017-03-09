import nltk
import string
from nltk.corpus import stopwords
from nltk.collocations import *
from bs4 import BeautifulSoup
from collections import defaultdict

bigramObject = nltk.collocations.BigramAssocMeasures()
stopWords= stopwords.words("english")


def main(): 
  # All program output will be placed in output.txt
  outputfile = open("output.txt","w")

  # Created an array for the different doc numbers to make looping easer
  for x in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]:
    processDoc("reut2-0" + x + ".sgm", outputfile)

  outputfile.close()

# Remove all punctuation
def removePunctuation(s):
  puncArray = list(string.punctuation)
  for punc in puncArray:
    s  = s.replace(punc, ' ')

  return s

# Remove all words of length 2 or less
def removeShort(s):
  words = s.split()
  newWords = ""
  for word in words:
    if len(word) > 2:
      newWords += word + " "

  return newWords

# Remove all stopwords - this will be used for one of the feature vectors
def removeStopWords(s):
  
  for stop in stopWords:
    #add spaces before and after to indicate independent word to replace
    #otherwise it will just replace all insances of 'i', 'a', etc.
    s  = s.replace(" " + stop + " ", ' ')
    #print "\n" + stop + "\n"

  return s

# Remove any "word" that is wholly a number
def removeNumbers(s):
  words = s.split()
  newWords = ""
  for word in words:
    if not word.isdigit():
      newWords += word + " "

  return newWords

# Function for formatting output to look like "vectors"
def listPrint(list):
  return "<" + ",".join(list) + ">"

# Function that will do the majority of processing for each individual document
def processDoc(filename, outputfile):
  document = BeautifulSoup(open(filename), 'html.parser')
  all_articles = document.find_all('reuters')

  # Iterate through all articles in single document.  Article is defined as anything between a set of <REUTER></REUTER> tags
  for article in all_articles:
    topics = article.topics
    places = article.places

    # topicItems is a tree containing topic data children
    # lamda d represents each child in topics.find_all('d')

    # topicItems now contains a list of all items
    topicItems = map(lambda d: d.contents[0], topics.find_all('d'))

    # placeItems now contains a list of all items
    placeItems = map (lambda p: p.contents[0], places.find_all('d'))

    

    #contains feature vector with all stop words removed
    stopWordsFV = defaultdict(lambda: 0, {})

    if article.body:
      # Remove punctuation & convert text to lowercase
      articleText = removePunctuation(article.body.contents[0].lower())
      # Remove short words
      articleText = removeShort(articleText)
      # Remove reuters word plastered at the end of each article
      articleText = articleText.replace("reuter", " ")
      # Remove #s
      articleText = removeNumbers(articleText)
     
      # Contains feature vector of bigrams w/o stop words removed
      # Most common sets of two words that occur near each other
      bigramFinder = BigramCollocationFinder.from_words(articleText.split(" "))
      bigramFV = map(lambda x: " ".join(x), bigramFinder.nbest(bigramObject.pmi, 10))

      # Remove stop words for next feature vector
      articleText = removeStopWords(articleText)

      for word in articleText.split():
        stopWordsFV[word] += 1

      # Output FV and topics n stuff to file the way we want it to look
      outputfile.write(article["newid"])    
      outputfile.write(" ")
      outputfile.write(listPrint(topicItems))
      outputfile.write(" ")
      outputfile.write(listPrint(placeItems))
      outputfile.write(" ")
      outputfile.write(listPrint(bigramFV))
      outputfile.write(" ")
      outputfile.write(listPrint(map(lambda word: "(" + word + "," + str(stopWordsFV[word]) + ")", stopWordsFV)))
      outputfile.write("\n")
      outputfile.write("\n")



main()