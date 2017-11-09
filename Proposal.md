# CSCI 6100G: Advanced Topics in Software Design

### Project Proposal 

**Alexandar Mihaylov ()**<br>
**Luisa Rojas-Garcia (100518772)**

---

## Problem Statement

## Motivation

*"Although in the field of Mining Software Repositories (MSR) there are many promising approaches to predicting, localizing, and triaging bugs, most of them do not consider impacts of each bug on users and developers but rather treat all bugs with equal weighting, excepting a few studies on high impact bugs including security, performance, blocking, and so forth."*

## Method

### About The Data

*"... dataset of high impact bugs which was created by manually reviewing four thousand issue reports in four open source projects (Ambari, Camel, Derby and Wicket)."* *"... where JIRA5 is used for managing reported
issues"*

*"These projects were selected because they met the following criteria for our project selection."*

- *"Target projects have a large number of (at least several thousand) reported issues."*, which enables the use for prediction model building and/or machine learning.

### Bug Classification

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

## References

[1] M. Ohira et al., “A dataset of high impact bugs: Manually-classified issue reports,” IEEE Int. Work. Conf. Min. Softw. Repos., vol. 2015–August, pp. 518–521, 2015.

[2] Website: http://oss.sys.wakayama-u.ac.jp/?p=1009