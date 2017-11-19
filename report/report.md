# $$Bug\space Priority\space Prediction$$


### $$ Luisa\space Rojas\space Garcia $$ $$ Alexandar\space Mihaylov $$ $$ CSCI\space 6100G $$


## `1 Layer`,`learning_rate = 0.001`, `num_perceptrons = 100`
```
 Using  100  perceptrons in the hidden layer ...
  EPOCH: 1000 Cost = 1.539145350456238
  EPOCH: 2000 Cost = 1.511128783226013
  EPOCH: 3000 Cost = 1.484516143798828
  EPOCH: 4000 Cost = 1.459460973739624
  EPOCH: 5000 Cost = 1.436717510223389
  EPOCH: 6000 Cost = 1.416223645210266
  EPOCH: 7000 Cost = 1.398078799247742
  EPOCH: 8000 Cost = 1.382284283638000
  EPOCH: 9000 Cost = 1.368311166763306
  EPOCH: 10000 Cost = 1.356206417083740
  > Accuracy = 61.86007% vs. Random = 19.70990%
```

* Mention as Future Work or Limits:

	The Sentiment Analysis may not take into account technical terms, giving them scores that are not accurate.
	
* The model seems to learn to always guess the priority category with the highest percentage - usually "Major". This causes the accuracy to always be as good as the percentage of this category in the data (labels).

* In order to deal with the imbalanced data, we used the following library: http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html