The data represents a number of tweets, recently downloaded from Twitter, and our goal is to identify which emoji was used in the tweet, by employing Machine Learning methods.

The list of possible emojis ("classes") is as follows:
Clap: 👏 (U+1F44F)                1         
Cry: 😭 (U+1F62D)                 0.3
Disappoint: 😞 (U+1F61E)          0.1
Explode: 🤯 (U+1F92F)             0.2
FacePalm: 🤦 (U+1F926)		  0.4
Hands: 🙌 (U+1F64C)		  0.9
Neutral: 😐 (U+1F610)		  0.6
Shrug: 🤷 (U+1F937)               0.5
Think: 🤔 (U+1F914) 		  0.7
Upside: 🙃 (U+1F643)		  0.8	

A full description of the 18 files in this archive follows:

RAW FILES
- train_raw.txt: This contains the raw text of the 37K training tweets, with emojis (and some other characters) removed. This file is tab-delimited, and has three fields: ID, Class, Tweet-text.
  The ID simply corresponds to the line number, prefixed by the character "1": line 1 is ID "11"; line 37413 is ID "137413", and so on.
  The Class corresponds to the emojis above. All tweets have only one of the ten emojis types (but they also may have contained other emojis, before we removed them).

- dev_raw.txt: This contains the raw text of the 12K development tweets. This file has the same format as train_raw.txt, except:
  The ID corresponds to the line number, prefixed by the character "2": line 1 is ID "21", and so on.

- test_raw.txt: This contains the raw text of the 12K test tweets. This file has the same format as train_raw.txt, except:
  The ID corresponds to the line number, prefixed by the character "3": line 1 is ID "31", and so on.
  The Class has been replaced with the character "?", for all test tweets

TOP 10 FILES
- top10.txt: This is a list of the tokens from the tweets, whose presence was automatically determined to be predictive of one of more emoji classes, using two statistical methods (Mutual Information and Chi-Square). If you are interested in how these were determined, you can post about it to the LMS Discussion Forum.

- train_top10.csv: This is a comma-separated-value (CSV) file, where each line corresponds to one of the 37K training tweets, where we have recorded the frequencies of the tokens in top10.txt. This file has the following format: ID,List-of-token-frequencies,Class

- train_top10.arff: This is the same as train_top10.csv, but with an ARFF header, to make it suitable for use with Weka.

- dev_top10.csv: This is a CSV file, where each line corresponds to one of the 12K development tweets. It has the same format as train_top10.csv

- dev_top10.arff: This is the same as dev_top10.csv, but with an ARFF header, to make it suitable for use with Weka.

- test_top10.csv: This is a CSV file, where each line corresponds to one of the 12K test tweets. It has the same format as train_top10.csv, and the Class has been replaced with the character "?"

- test_top10.arff: This is the same as test_top10.csv, but with an ARFF header, to make it suitable for use with Weka.

MOST 100 FILES
- most100.txt: This is a list of the 100 tokens which appear with the greatest frequency in the training collection.

- train_most100.csv: This is a comma-separated-value (CSV) file, where each line corresponds to one of the 37K training tweets, where we have recorded the frequencies of the tokens in most100.txt. This file has the following format: ID,List-of-token-frequencies,Class

- train_most100.arff: This is the same as train_most100.csv, but with an ARFF header, to make it suitable for use with Weka.

- dev_most100.csv: This is a CSV file, where each line corresponds to one of the 12K development tweets. It has the same format as train_most100.csv

- dev_most100.arff: This is the same as dev_most100.csv, but with an ARFF header, to make it suitable for use with Weka.

- test_most100.csv: This is a CSV file, where each line corresponds to one of the 12K test tweets. It has the same format as train_most100.csv, and the Class has been replaced with the character "?"

- test_most100.arff: This is the same as test_most100.csv, but with an ARFF header, to make it suitable for use with Weka.

README
- README.txt: This is the file you are currently reading.
