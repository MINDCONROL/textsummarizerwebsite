# %%

import streamlit as st
import pandas as pd
import torch

# %%
text ="""
Urbanization, a transformative global phenomenon, represents the increasing concentration of human populations in densely populated areas known as cities. This intricate process, driven by a complex interplay of socio-economic, political, and environmental forces, has profoundly reshaped human societies and the landscapes they inhabit. Understanding the nuances of urbanization is crucial for navigating the challenges and harnessing the opportunities it presents in our rapidly evolving world.

The engines of urbanization are multifaceted. Economic opportunities stand as a primary draw, with cities often serving as hubs for industry, commerce, and innovation, offering a wider array of jobs and higher earning potential compared to rural areas. This economic magnetism pulls individuals and families in search of improved livelihoods. Social factors also play a significant role. Cities frequently offer greater access to education, healthcare, cultural amenities, and diverse social networks, enhancing the quality of life for many. The allure of a vibrant social scene and the promise of upward mobility further fuel urban migration."""

# %%
#count the characters in the text
text_char = len(text)

# %%
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
#from collections import Counter


# %%
#load the spacy model
nlp = spacy.load("en_core_web_sm")

# %%
#process the text with spacy
doc = nlp(text)

# %%
# Create a list of tokens by iterating through each token in the 'doc' object.
# For each token, convert it to lowercase using token.text.lower().
# Apply a series of filters to include only relevant tokens:
    # Exclude punctuation using 'not token.is_punct'.
    # Exclude whitespace characters using 'not token.is_space'.
    # Exclude stop words (common words like 'the', 'a', 'is') using 'not token.is_stop'.
    # Explicitly exclude newline characters ('\n').
    # Explicitly exclude tab characters ('\t').
    # Explicitly exclude carriage return characters ('\r').
# The result is a list named 'tokens' containing the cleaned and lowercased textual content of the 'doc' object.
tokens = [token.text.lower() for token in doc
          if not token.is_punct  # Remove punctuation
          and not token.is_space  # Remove whitespace
          and not token.is_stop   # Remove stop words
          and token.text not in ["\n", "\t", "\r"]] # Remove newline/tab/carriage return

# %%
tokens

# %%
#alternative way to create tokens lists
tokens1 = []  # Initialize an empty list to store the processed tokens.
stopwords = list(STOP_WORDS)  # Get the default stop words from spaCy and convert them to a list for efficient 'in' checking.
allowed_pos = ['ADJ', 'PROPN', 'NOUN', 'VERB', 'ADV']  # Define a list of allowed Part-of-Speech (POS) tags that we want to keep (Adjective, Proper Noun, Noun, Verb, Adverb).

for token in doc:  # Iterate through each individual token that has been processed by the spaCy language model.
    if token.text in stopwords or token.text in punctuation:
        continue  # If the current token's text is found within our list of stop words OR if it's a punctuation mark, we skip this token and move to the next one.
    if token.pos_ in allowed_pos:
        tokens1.append(token.text.lower())  # If the current token's Part-of-Speech tag is present in our 'allowed_pos' list, we convert the token's text to lowercase and append it to our 'tokens1' list.



# %%
tokens1

# %%
from collections import Counter

# %%
word_freq = Counter(tokens1) # Create a Counter object to count the frequency of each token in the 'tokens1' list.

# %%
word_freq

# %%
max_freq = max(word_freq.values()) # Find the maximum frequency value from the 'word_freq' Counter object.
max_freq

# %%
for word in word_freq.keys(): # Iterate through each unique word in the 'word_freq' Counter object.
    word_freq[word] = (word_freq[word] / max_freq) # Normalize the frequency of each word by dividing its count by the maximum frequency value.  

# %%
word_freq

# %%
sent_token = [sent.text for sent in doc.sents] # Create a list of sentences by iterating through each sentence in the processed document.
#sent_token
sent_scores = {} # Initialize an empty dictionary to store the scores of each sentence.
#for sent in doc.sents: # Iterate through each sentence in the processed document.
    #for word in sent: # For each word in the current sentence.
        #if word.text.lower() in word_freq.keys(): # Check if the lowercase version of the word is present in our 'word_freq' dictionary.
            #if sent in sent_scores.keys(): # If the current sentence is already present in our 'sent_scores' dictionary.
                #sent_scores[sent] += word_freq[word.text.lower()] # Increment the score of the current sentence by the frequency of the word.
           # else:
                #sent_scores[sent] = word_freq[word.text.lower()] # If the sentence is not already present, initialize its score with the frequency of the word.

for sent in sent_token: # Iterate through each sentence in the processed document.
  for word in sent.split(): # For each word in the current sentence.
    if word.lower() in word_freq.keys():
      if sent not in sent_scores.keys(): # If the current sentence is not already present in our 'sent_scores' dictionary.
        sent_scores[sent] = word_freq[word]# Initialize the score of the current sentence with the frequency of the word.
      else:
      
          sent_scores[sent] += word_freq[word] # Increment the score of the current sentence by the frequency of the word.

  print(word) # Print the word being processed.
  print(sent) # Print the sentence being processed. 


# %%
sent_scores # Display the final scores of each sentence.

# %%
import pandas as pd
# Create a DataFrame from the 'sent_scores' dictionary, where the index is the sentence and the values are the scores.
#df = pd.DataFrame.from_dict(sent_scores, orient='index', columns=['Score']) # Create a DataFrame from the 'sent_scores' dictionary, where the index is the sentence and the values are the scores.
#df = df.sort_values(by='Score', ascending=False) # Sort the DataFrame by the 'Score' column in descending order.
pd.DataFrame(list(sent_scores.items()), columns=['Sentence', 'Score']).sort_values(by='Score', ascending=False) # Create a DataFrame from the 'sent_scores' dictionary, where the index is the sentence and the values are the scores.

# %%
from heapq import nlargest # Import the 'nlargest' function from the 'heapq' module to find the largest elements in an iterable.
num_sentences = 3 # Define the number of sentences to extract for the summary.

n = nlargest(num_sentences, sent_scores, key=sent_scores.get) # Use the 'nlargest' function to find the top 'number of sentences' sentences based on their scores.
" ".join(n) # Join the selected sentences into a single string, separating them with a newline character.
# Display the final summary. 


