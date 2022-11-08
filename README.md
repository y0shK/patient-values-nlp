# What Do Patients Value? Using Natural Language Processing to Define a Good Doctor's Visit

## Description
In this project, we wanted to find out what patients value in their doctors, regarding specific qualities or attitudes. To achieve this goal, we identify the keywords that patients use to describe 
highly-rated doctors, and we consider in what context these words are used. More specifically, we employed logistic regression to identify the most frequent and salient words of both positive and 
negative sentiment. Then using dependency parsing, we contextualized these key adjectives back to the nouns they modified to observe the specific phrases of importance (e.g., "a 
caring doctor").

## Future Work
Future work in this project aims to assess gender bias in the review corpus and continue to validate our findings. Current action items for this project include the following:

* running subsets of the dataset for providers of different gender to see if trends of gender bias cause discrepancy in ratings for similar traits
* finding baseline models to run the Yelp dataset for validation
* finding benchmark datasets to further validate the performance of our model

## Dataset
The Yelp dataset (hyperlinked below) is used for this project (allowed for academic, non-commercial research).

## Publications
This code is used for publications in the following venues:
* Consortium for Computing Sciences in Colleges
* Southern California Conference for Undergraduate Researchers

## How to Run
1. Download [Yelp dataset](https://www.yelp.com/dataset)
2. Put business and review academic dataset JSON files in the same directory as JSON preprocessing Python script
3. Run JSON preprocessing script to get JSON used as input for patient values Python script
4. Run patient values Python script
