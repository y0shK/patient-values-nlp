"""
Preprocessing script that generates a json file to be used as input for patient_values.py
Author: Yash Karandikar
"""

# extract reviews from Yelp json to be used as input for patient_values.py

import os
import json
import gender_guesser.detector as gender

d = gender.Detector()

yelp_reviews = []

# https://stackoverflow.com/questions/12280143/how-to-move-to-one-folder-back-in-python/12280199
# https://stackoverflow.com/questions/12451431/loading-and-parsing-a-json-file-with-multiple-json-objects
# https://stackoverflow.com/questions/49640513/unicodedecodeerror-charmap-codec-cant-decode-byte-0x9d-in-position-x-charac

"""
1. obtain reviews associated with healthcare
"""

# obtain only those reviews that are related to healthcare, specifically relating to string "doctor" or "MD"

with open('yelp_academic_dataset_business.json', encoding="utf8") as f:
    for line in f:
        if "doctors" in line or "Doctors" in line\
                or ", MD" in line or ", M.D." in line:
            yelp_reviews.append(json.loads(line))

review_dict = {}

for review in yelp_reviews:
    review_dict[review["business_id"]] = review["name"]

yelp_stars = []

# break after some amount of reviews if shorter runtime desired
count = 0 # change this number if running for evaluation_set.json to get different set of reviews

with open('yelp_academic_dataset_review.json', encoding="utf8") as f:
    for line in f:
        yelp_stars.append(json.loads(line))
        count += 1
        if count > 10000000: # e.g., 1,000,000 but this number can change depending on if initial count changes
            break


# check if the business ID is the same
# check the key of the hash table against the business ID from yelp_stars
# create a new dictionary to join business IDs between business.json and reviews.json
joined_review_dict = {}

"""
2. use Python library to guess gender of provider for future work of gender analysis.
   Future work aims to ascertain to what extent gender bias exists in the reviews (e.g., female providers ranked lower than male providers for same traits?)
"""

# count total rows with "MD" or "MD"
# count individual statistics
    # male
    # female
    # mostly_male
    # mostly_female
    # unknown

gender_stats = {'male': 0,
                'female': 0,
                'mostly_male': 0,
                'mostly_female': 0,
                'unknown': 0}

reviews_count = 0
total_size = 0

for review in yelp_stars:
    if review["business_id"] in review_dict.keys():

        for k, v in review_dict.items():
            if k == review["business_id"]:
                # v == name

                gender_md = ""

                if "MD" in v or "M.D." in v:
                    first_name = v.split()[0]
                    gender_md = d.get_gender(first_name)

                    if str(gender_md) == "male":
                        gender_stats["male"] += 1
                    elif str(gender_md) == "female":
                        gender_stats["female"] += 1
                    elif str(gender_md) == "mostly_male":
                        gender_stats["mostly_male"] += 1
                    elif str(gender_md) == "mostly_female":
                        gender_stats["mostly_female"] += 1
                    elif str(gender_md) == "unknown":
                        gender_stats["unknown"] += 1

                    reviews_count += 1

                    joined_review_dict[k] = (v, review["text"], review["stars"], gender_md)

                total_size += 1


# write the reviews to a json file
# https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file

with open('new_train_set_1billion.json', 'w', encoding='utf8') as f:
    json.dump(joined_review_dict, f, indent=4)

