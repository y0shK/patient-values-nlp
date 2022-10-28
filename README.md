# What Do Patients Value? Using Natural Language Processing to Define a Good Doctor's Visit

## Description
In this project, we wanted to find out what patients value in their doctors, especially with regards to specific qualities or attitudes. To achieve this goal, we define patient-valued qualities as nouns or adjectives that patients use to describe highly-rated doctors, and we conside in what context these words are used. More specifically, logistic regression is used to identify word presence or absence in reviews and highly frequent or salient words are considered more by the model. Dependency parsing is then used to contextualize these nouns and adjectives by tying them back to specific phrases in which they are used (e.g., "a caring doctor").

## Future Work
Future work in this project aims to assess gender bias in the review corpus and continue to validate our findings. Current action items for this project include the following:

* running subsets of the dataset for providers of different gender to see if trends of gender bias cause discrepancy in ratings for similar traits
* finding baseline models to run the Yelp dataset for validation
* finding benchmark datasets to further validate the performance of our model

## Dataset
The Yelp dataset (here hyperlinked but not directly attached as a file) is used for this project (allowed for academic, non-commercial research).

## Publications
This code is used for publications in the following venues:
* Consortium for Computing Sciences in Colleges
* Southern California Conference for Undergraduate Researchers

## How to Run
1. Download [Yelp dataset](https://www.yelp.com/dataset)
2. Put business and review academic dataset json files in the same directory as json preprocessing Python script
3. Run json preprocessing script to get json used as input for patient values Python script
4. Run patient values Python script
