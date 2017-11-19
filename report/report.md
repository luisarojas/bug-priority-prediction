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