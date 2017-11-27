<link rel="shortcut icon" type="image/x-icon" href="/favicon.ico?">

# Project Description

## Problem Statement

Given any high impact bug, identify its priority.

Methods in research of Software Engineering focus on predicting, localizing, and triaging bugs, but do not consider their impact or weight on the users and on the developers.

For this reason, we want to distinguish different kinds of bugs by placing in them in different priority categories: **Critical**, **Blocker**, **Major**, **Minor**, **Trivial**.

## Motivation

Bug priority categorization allows for improved delegation of bug resolution amongst developers of different domain expertise.

## Method

We will implement a Deep Neural Network to classify the different types of bugs. Different aspects will be taken into account:

- Feature Selection: Reduce the dimensionality of the data.

- Sentiment Analysis: Textual attributes will be quantified by applying sentiment analysis models and, thus, generating new features.

- Hidden Layers: Determine the optimal number of hidden layers, as well as their perceptron quantity and corresponding activation functions.

- Analysis: The model will be applied to all projects; the results will be compared.

### About The Data

We will work with a large dataset of high impact bugs, which was created by manually reviewing four thousand issue reports in four open source projects (Ambari, Camel, Derby and Wicket). The projects were extracted from JIRA, a platform for managing reported issues.

There are 1000 examples per project; there will be 4000 examples to work with in total.

These projects were selected because they met the following criteria for project selection:

- Target projects have a large number of (at least several thousand) reported issues , which enables the use for prediction model building and/or machine learning.

- Target projects use JIRA as an issue tracking system.

- Target projects are different from each other in application domains.

## Related Work

*Include other work to predict bug priority using this same dataset, then compare with our approach.*

*To do this, look up what research that cites the dataset that predict defect priority with other methods.*

| Paper | Method |
|-------|--------|
| [Automated Identification of High Impact Bug Reports Leveraging Imbalanced Learning Strategies](http://ieeexplore.ieee.org.uproxy.library.dc-uoit.ca/stamp/stamp.jsp?arnumber=7552013&tag=1 "Paper") |  Naive Bayes Multinominal |

## References

[1] M. Ohira et al., “A dataset of high impact bugs: Manually-classified issue reports,” IEEE Int. Work. Conf. Min. Softw. Repos., vol. 2015–August, pp. 518–521, 2015.

[2] Open Source Software Engineering Lab, “High Impact Bug Dataset”. 2015. [Online]. Available: http://oss.sys.wakayama-u.ac.jp/?p=1009 [Accessed: 08- Nov- 2017].