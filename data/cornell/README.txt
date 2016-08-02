Cornell Movie-Dialogs Corpus

Distributed together with:

"Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs"
Cristian Danescu-Niculescu-Mizil and Lillian Lee
Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, ACL 2011.

(this paper is included in this zip file)

NOTE: If you have results to report on these corpora, please send email to cristian@cs.cornell.edu or llee@cs.cornell.edu so we can add you to our list of people using this data.  Thanks!


Contents of this README:

	A) Brief description
	B) Files description
	C) Details on the collection procedure
	D) Contact


A) Brief description:

This corpus contains a metadata-rich collection of fictional conversations extracted from raw movie scripts:

- 220,579 conversational exchanges between 10,292 pairs of movie characters
- involves 9,035 characters from 617 movies
- in total 304,713 utterances
- movie metadata included:
	- genres
	- release year
	- IMDB rating
	- number of IMDB votes
	- IMDB rating
- character metadata included:
	- gender (for 3,774 characters)
	- position on movie credits (3,321 characters)


B) Files description:

In all files the field separator is " +++$+++ "

- movie_titles_metadata.txt
	- contains information about each movie title
	- fields: 
		- movieID, 
		- movie title,
		- movie year, 
	   	- IMDB rating,
		- no. IMDB votes,
 		- genres in the format ['genre1','genre2',É,'genreN']

- movie_characters_metadata.txt
	- contains information about each movie character
	- fields:
		- characterID
		- character name
		- movieID
		- movie title
		- gender ("?" for unlabeled cases)
		- position in credits ("?" for unlabeled cases) 

- movie_lines.txt
	- contains the actual text of each utterance
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance

- movie_conversations.txt
	- the structure of the conversations
	- fields
		- characterID of the first character involved in the conversation
		- characterID of the second character involved in the conversation
		- movieID of the movie in which the conversation occurred
		- list of the utterances that make the conversation, in chronological 
			order: ['lineID1','lineID2',É,'lineIDN']
			has to be matched with movie_lines.txt to reconstruct the actual content

- raw_script_urls.txt
	- the urls from which the raw sources were retrieved

C) Details on the collection procedure:

We started from raw publicly available movie scripts (sources acknowledged in 
raw_script_urls.txt).  In order to collect the metadata necessary for this study 
and to distinguish between two script versions of the same movie, we automatically
 matched each script with an entry in movie database provided by IMDB (The Internet
 Movie Database; data interfaces available at http://www.imdb.com/interfaces). Some
 amount of manual correction was also involved. When  more than one movie with the same
 title was found in IMBD, the match was made with the most popular title 
(the one that received most IMDB votes)  

After discarding all movies that could not be matched or that had less than 5 IMDB 
votes, we were left with 617 unique titles with metadata including genre, release 
year, IMDB rating and no. of IMDB votes and cast distribution.  We then identified 
the pairs of characters that interact and separated their conversations automatically 
using simple data processing heuristics. After discarding all pairs that exchanged 
less than 5 conversational exchanges there were 10,292 left, exchanging 220,579 
conversational exchanges (304,713 utterances).  After automatically matching the names 
of the 9,035 involved characters to the list of cast distribution, we used the 
gender of each interpreting actor to infer the fictional gender of a subset of 
3,321 movie characters (we raised the number of gendered 3,774 characters through
 manual annotation). Similarly, we collected the end credit position of a subset 
of 3,321 characters as a proxy for their status.


D) Contact:

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)