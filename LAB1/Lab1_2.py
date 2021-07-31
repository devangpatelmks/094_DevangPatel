# CELL 0
# Import neccessary library functions
import nltk # Python library for NLP
import matplotlib.pyplot as plt # Library for visualization
import random, os, json
from pathlib import Path

# CELL 1
# Download sample movie reviews dataset
nltk.download('movie_reviews')

# Assign directory of files
pos_dir = '/root/nltk_data/corpora/movie_reviews/pos/'
neg_dir = '/root/nltk_data/corpora/movie_reviews/neg/'

# Make a list of reviews
pr_list = []
nr_list = []

# CELL 2
# Path for txt files.
pos_files = Path(pos_dir).glob('cv*.txt')
neg_files = Path(neg_dir).glob('cv*.txt')

# Get filenames and sort it.
pfiles = []
nfiles = []
for i in pos_files:
  pfiles.append(os.path.basename(i))
for j in neg_files:
  nfiles.append(os.path.basename(j))
pfiles.sort()
nfiles.sort()

# Append to list.
for i in pfiles:
  fpath = pos_dir + i
  f = open(fpath, 'r')
  p_kv = {'filename': i, 'review': f.read()}
  pr_list.append(p_kv)
  f.close
    
for j in nfiles:
  fpath = neg_dir + j
  f = open(fpath, 'r')
  n_kv = {'filename': j, 'review': f.read()}
  nr_list.append(n_kv)
  f.close

# CELL 3
print('Number of positive reviews: ', len(pr_list))
print('Number of negative reviews: ', len(nr_list))
print('\nThe type of positive reviews is: ', type(pr_list))
print('The type of a review entry is: ', type(nr_list[0]))

# CELL 4
# Declare a figure with a custom size
fig = plt.figure(figsize=(5, 5))
# Labels for the two classes
labels = 'Positives', 'Negative'
# Sizes for each slide
sizes = [len(pr_list), len(nr_list)]
# Declare pie chart, where the slices will be ordered and plotted
# counter-clockwise:
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')
# Display the chart
plt.show()

# CELL 5
# Print positive in green
rand_pos_dict = pr_list[random.randint(0,1000)]
print('\033[92m' + rand_pos_dict['review'])

# Print negative in red
rand_neg_dict = nr_list[random.randint(0,1000)]
print('\033[91m' + rand_pos_dict['review'])

# CELL 6
# Download the stopwords from NLTK
nltk.download('stopwords')

# CELL 7
import re # Library for regular expression operations
import string # For string operations
from nltk.corpus import stopwords # Module for stop words that come with NLTK
from nltk.stem import PorterStemmer # Module for stemming
from nltk.tokenize import TweetTokenizer #Module for tokenizing strings

# CELL 8
sample_neg_dict = nr_list[532]
sample_neg_review = sample_neg_dict['review']
print(sample_neg_review)

# CELL 9
# Remove hyperlinks
sample_neg_review = re.sub(r'https?:\/\/.*[\r\n]*', '', sample_neg_review)
# Remove hashtags; only removing the hash # sign from the word
sample_neg_review = re.sub(r'#', '', sample_neg_review)

# CELL 10
# Instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False)
# Tokenize reviews
review_tokens = tokenizer.tokenize(sample_neg_review)

print(sample_neg_review)
print()
print('Tokenized review:')
print(review_tokens)

# CELL 11
# Import the english stop words list from NLTK
stopwords_english = stopwords.words('english')
print('Stop words:')
print(stopwords_english)

print('\nPunctuation:')
print(string.punctuation)

# CELL 12
print(review_tokens)
review_clean = []
for word in review_tokens: # Go through every word in your tokens list
  if (word not in stopwords_english and # Remove stopwords
  word not in string.punctuation): # Remove punctuation
    review_clean.append(word)

# CELL 13
# Instantiate stemming class
stemmer = PorterStemmer()
# Create an empty list to store the stems
review_stem = []
for word in review_clean:
  stem_word = stemmer.stem(word) # Stemming word
  review_stem.append(stem_word) # Append to the list

print(review_clean)
print()
print('Stemmed words:')
print(review_stem)