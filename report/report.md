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
		
		Using 0.01 learning rate, 10,000 epochs, 10 perceptrons (1 layer):
		
		* **Naive**: accuracy = 20.7% (*ran once*)
		* **Smote**: accuracy = 25.28205% (*ran once*)
		* **Adasyn**: 20.56663% (*ran once*)

## Features for best results

### Identifying the best features

Original features: `type`, `reporter`, `summary`, `description `, `description_words`

Using 0.01 learning rate, 10,000 epochs, 10 perceptrons (1 layer), and NAIVE for data balancing:

#### Using 4 Features

| Features Used | Accuracy |
|---------------|----------|
| `reporter`, `summary`, `description`, `description_words` | 20.30769% |
| `type`, `summary`, `description`, `description_words` | 23.33333% |
| `type`, `reporter`, `description`, `description_words` | 20.10256% |
| `type`, `reporter`, `summary`, `description_words` | 21.46154% |
| `type`, `reporter`, `summary`, `description` | 20.00000% |

#### Using 3 Features

| Features Used | Accuracy |
|---------------|----------|
| `summary`, `description`, `description_words` | 20.41026% |
| `reporter`, `description`, `description_words` |20.00000% |
| `reporter`, `summary`, `description_words` | 20.00000% |
| `reporter`, `summary`, `description` | 20.00000% |
| `type`, `description`, `description_words` | 20.00000% |
| `type`,`summary`, `description_words` | 20.00000% |
| `type`, `summary`, `description` | 32.69231% |
| `type`, `reporter`, `description_words` | 19.74359% |
| `type`, `reporter`, `description ` | 22.48718% |
| `type`, `reporter`, `summary` | 20.07692% |

#### Using 2 Features

| Features Used | Accuracy |
|---------------|----------|
| `description`, `description_words` | 20.00000% |
| `summary`, `description_words` | 21.12820% |
| `summary`, `description` | 21.61538% |
| `reporter`, `description_words` | 20.82051% |
| `reporter`, `description` | 20.02564% |
| `reporter`, `summary` | 20.00000% |
| `type`, `description_words` | 20.33333% |
| `type`, `description` | 32.64103% |
| `type`, `summary` | 27.84615% |
| `type`, `reporter` | 20.33333% |

#### Using 1 Feature

| Features Used | Accuracy |
|---------------|----------|
| `type` | 28.61539% |
| `reporter` | 20.10256% |
| `summary` | 20.84615% |
| `description` | 20.41026% |
| `description_words` | 20.00000% |

According to the data collected above, the feature combinations that achieved the highest accuracy score are shown below. These were selected to also be ran using three different balancing data appraoches:

| Ranking | Features Used | Accuracy (NAIVE) | Accuracy (SMOTE) | Accuracy (ADASYN) | Average Accuracy |
|:-------:|---------------|:--------:|:--------:|:--------:|:-------:|
| 1 | `type`, `summary`, `description` | 32.69231% | 31.84615% | 24.32361% | 29.62069% |
| 2 | `type`, `description` | 32.64103% | 31.84615% | 22.20551% | 28.897563%
| 3 | `type` | 28.61539% | 28.38461% | 24.61696% | 27.205653% |
| 4 | `type`, `summary` | 27.84615% | 32.33333% | 26.01171% | 28.730397% |
| 5 | `type`, `summary`, `description`, `description_words` | 23.33333% | 20.20513% | 20.60197% | 21.380143% |

After running those with the three balancing data approaches and calculating their corresponding accuracies, the new ranking is the following:

| Ranking | Features Used | Accuracy (NAIVE) | Accuracy (SMOTE) | Accuracy (ADASYN) | Average Accuracy |
|:-------:|---------------|:--------:|:--------:|:--------:|:-------:|
| 1 | `type`, `summary`, `description` | 32.69231% | 31.84615% | 24.32361% | 29.62069% |
| 2 | `type`, `description` | 32.64103% | 31.84615% | 22.20551% | 28.897563%
| &uarr; 3 | `type`, `summary` | 27.84615% | 32.33333% | 26.01171% | 28.730397% |
| &darr; 4 | `type` | 28.61539% | 28.38461% | 24.61696% | 27.205653% |
| 5 | `type`, `summary`, `description`, `description_words` | 23.33333% | 20.20513% | 20.60197% | 21.380143% |

Even though both the Naive and SMOTE approaches show excellent results, SMOTE will be used for the next calculations.
This is because previous to feature manipulation, SMOTE achieved the highest accuracy score (25.28205%) - using all the features in the dataset.

## Automatically identifying the most influential features

## Recall, Precision, and F1