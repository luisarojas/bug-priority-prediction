# CSCI 6100G: Advanced Topics in Software Design

### Project Proposal 

**Alexandar Mihaylov (100536396)**<br>
**Luisa Rojas-Garcia (100518772)**

---

## Problem Statement

Given any high impact bug, identify its priority.

Methods in research of Software Engineering focus on predicting, localizing, and triaging bugs, but do not consider their impact or weight on the users and on the developers.

For this reason, we want to distinguish different kinds of bugs by placing in them in different priority categories:

- Critical
- Blocker
- Major
- Minor
- Trivial

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

<font color=lightgrey>
### Definitions of High Impact Bugs 

#### Process

*"A bug can impact on a bug management process in a project. When an unexpected bug is found in an unexpected component, developers in the projects would need to reschedule task assignments in order to give first priority to fix the newlyfound bug."*

- **Surprise bugs**:

	- *"It can disturb the workflow and/or task scheduling of developers, since it appears in unexpected timing (e.g., bugs detected in post-release) and locations (e.g., bugs found in files that are rarely changed in pre-release)."*
	
	- *"... the co-changed files and the amount of time between the latest pre-release date for changes and the release date can be good indicators of predicting surprise bugs"*

- **Dormant bugs**: 
	
	- *"A bug that was introduced in one version (e.g., Version 1.1) of a system, yet it is Not reported until AFTER the next immediate version (i.e., a bug is reported against Version 1.2 or later)."*
	
	- *"... were fixed faster than non-dormant bugs."*
	
- **Blocking bugs**: 

 	- *"... blocks other bugs from being fixed".*
	
 	- *"... a fixer needs more time to fix a blocking bug and other developers need to wait for being fixed to fix the dependent bugs."*

#### Products

They directly affect user experience and satisfaction with software products.

- **Security bugs**:
	
	- *"... can raise a serious problem which often impacts on uses of software products directly."*
	
	- *"In general, security bugs are supposed to be fixed as soon as possible."*
	
- **Performance bugs**: 

	- *"... programming errors that cause significant performance degradation."*

	- *"The “performance degradation” contains poor user experience, lazy application responsiveness, lower system throughput, and needles waste of computational resources."*

	- *"... a performance bug needs more time to be fixed than a non-performance bug."*

- **Breakage bugs**:

	- *"A functional bug which is introduced into a product because the source code is modified to add new features or to fix existing bugs."*

	- *"A breakage bug causes a problem which makes usable functions in one version unusable after releasing newer versions."*

### Manual Classification

Surprise and Dormant bugs were automatically labeled using a script since they can be easily detected by definition. As for the rest:

- For a project, a graduate student reviewed a thousand issue reports and labeled one or more bug types on each issue (i.e., multiple labelling is allowed). Four graduate students participated in this session.

- *"Four faculty members of the authors independently did the same thing as the students."*

- *"A student and faculty member who reviewed the same issue reports discussed differences of labeling between them until reaching a common understanding and labeling the same types on a single issue."*

</font>

## Related Work

*Include other work to predict bug priority using this same dataset, then compare with our approach.*

*To do this, look up what research that cites the dataset that predict defect priority with other methods.*

| Paper | Method |
|-------|--------|
| [Automated Identification of High Impact Bug Reports Leveraging Imbalanced Learning Strategies](http://ieeexplore.ieee.org.uproxy.library.dc-uoit.ca/stamp/stamp.jsp?arnumber=7552013&tag=1 "Paper") |  Naive Bayes Multinominal

## References

[1] M. Ohira et al., “A dataset of high impact bugs: Manually-classified issue reports,” IEEE Int. Work. Conf. Min. Softw. Repos., vol. 2015–August, pp. 518–521, 2015.

[2] Open Source Software Engineering Lab, “High Impact Bug Dataset”. 2015. [Online]. Available: http://oss.sys.wakayama-u.ac.jp/?p=1009 [Accessed: 08- Nov- 2017].