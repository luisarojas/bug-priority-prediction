# $$Bug\space Priority\space Prediction$$

### $$ Luisa\space Rojas\space Garcia $$ $$ Alexandar\space Mihaylov $$ $$ CSCI\space 6100G $$

## Dependencies

* `python 3`
* `scikit-learn`
* `imblearn`:
	
	```
	conda install -c glemaitre imbalanced-learn
	```
	<font color=red>Note: Also state other ways to install or provde the link for more information.</font>


## Work or Limits:

The Sentiment Analysis may not take into account technical terms, giving them scores that are not accurate.

## Imbalanced Data
	
* The model seems to learn to always guess the priority category with the highest percentage - usually "Major". This causes the accuracy to always be as good as the percentage of this category in the data (labels).

	* This was observed when the model was ran for all of the data combined, as well as for each specific project being worked with.

	* In order to deal with the imbalanced data, we used the following library: http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html. There are three different approaches that were tested:

		[] Naive random over-sampling
		
		[] Synthetic Minority Oversampling Technique (SMOTE)
		
		[] Adaptive Synthetic (ADASYN)
		
		* TEST RUN RESULTS
		
			* Naive: accuracy = 20.7% (*ran once*)
			* Smote: accuracy = 25.28205% (*ran once*)
			* Adasyn: 20.56663% (*ran once*)

## Features for best results

`type`, `reporter`, `summary`, `description `, `description_words`

(using 0.01 learning rate, 10,000 epochs and 10 perceptrons (1 layer))

### Removing 1 feature at a time

* `reporter`, `summary`, `description`, `description_words`: 20.30769%
* `type`, `summary`, `description`, `description_words`: 23.33333%
* `type`, `reporter`, `description`, `description_words`: 20.10256%
* `type`, `reporter`, `summary`, `description_words`: 21.46154%
* `type`, `reporter`, `summary`, `description`: 20.00000%

### Removing 2 features at a time

* 01: `summary`, `description`, `description_words`: 20.41026%
* 02: `reporter`, `description`, `description_words`: 20.00000%
* 03: `reporter`, `summary`, `description_words`: 20.00000%
* 04: `reporter`, `summary`, `description`: 20.00000%

* 12: `type`, `description`, `description_words`: 20.00000%
* 13: `type`,`summary`, `description_words`: 20.00000%
* 14: `type`, `summary`, `description`: 18.71795%

* 23: `type`, `reporter`, `description_words`: 19.74359%
* 24: `type`, `reporter`, `description `: 22.48718%

* 34: `type`, `reporter`, `summary`: 20.07692%

### Removing 3 features at a time

* 012: `description`, `description_words`: 20.00000%
* 013: `summary`, `description_words`: 21.12820%
* 014: `summary`, `description`: 21.61538%
* 023: `reporter`, `description_words`: 20.82051%
* 024: `reporter`, `description`: 20.02564%
* 034: `reporter`, `summary`: 20.00000%

* 123: `type`, `description_words`: 20.33333%
* 124: `type`, `description`: 32.64103%
* 134: `type`, `summary`: 27.84615%

* 234: `type`, `reporter`: 20.33333%

### Removing 4 features at a time

* 1234: `type`: 28.61539%
* 0234: `reporter`: 20.10256%
* 0134: `summary`: 20.84615%
* 0124: `description`: 20.41026%
* 0123: `description_words`: 20.00000%
