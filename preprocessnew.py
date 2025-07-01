import pandas as pd
import numpy as np
import json
import os
import re
import emoji
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from nltk.corpus import words
import wordninja

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def make_json(train=True, path="dataset"):
    word="train"
    if train:
        word="train"
        path=os.path.join(path,"training")

    else:
        word="test"
        path=os.path.join(path,"testing")
        
    
    
    all_data = {}  # Dictionary to store sentences and their labels
    for lang in ["en","hi","ta"]:
        for ind in range(3):
            filepath = os.path.join(path, f"{word}_{lang}_l{ind+1}.csv")
            file_key = os.path.splitext(filepath)[0][-2:]  # Extract "l1", "l2", "l3"
            df = pd.read_csv(filepath,engine="python")  # Load CSV file
            maxrange=8
            if lang=="hi":
                maxrange=7
            print(len(df))

            for i in tqdm(range(len(df)), desc=f"Processing {file_key}"):
                text = df.iloc[i, 0]  # Extract text column
                text = text.replace("<handle replaced>", " ").replace("&amp;", "and").strip()

                labels = []
                total = 0
                count = 0

                for value in range(2, maxrange):  # Extract labels
                    v = df.iloc[i, value]

                    if v != "NL":
                        try:
                            v = float(v)
                            if not np.isnan(v):
                                labels.append(v)
                                total += v
                                count += 1
                        except ValueError:
                            pass  # Skip non-numeric values safely

                final_label = round((total / count) + 0.001) if count > 0 else None  # Avoid division by zero

                # Store data
                if text not in all_data:
                    all_data[text] = {"text": text,"tokens":tokenize(text),"line":i+1}

                if "labels" not in all_data[text]:
                    all_data[text]["labels"] = {file_key: final_label}
                else:
                    all_data[text]["labels"][file_key] = final_label

    # Convert dictionary to list format
    processed_data = list(all_data.values())

    # Save as JSON files
    output_filepath = os.path.join(path, f"final.json")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_filepath}")

def split_hashtag_smart(hashtag):
    word_list = set(words.words())
    hashtag = hashtag.lstrip('#') # Remove '#' and lowercase
    result = wordninja.split(hashtag)
    return result

def split_on_capitals(text):
    if len(text)==0:
        return []
    inds = []
    prev = text[0]

    for i in range(1, len(text)):
        if (prev.islower() and text[i].isupper()):  # Detect transition from uppercase to lowercase
            inds.append(i)
        prev = text[i]

    # Use the indices to split the text
    parts = []
    start = 0
    for idx in inds:
        parts.append(text[start:idx])
        start = idx
    parts.append(text[start:])  # Add the last part

    return parts

def tokenize(text):
    rem = {',', '.', '(', ')', '!', '?', ':', ';', '”', '“', '"',"*","^","‘","’","'"}
    splits = {',', ' ', '-','.','/'}

    # Convert text to lowercase
    #text = text.lower()

    # Separate emojis from text
    text = ''.join(f' {char} ' if emoji.is_emoji(char) else char for char in text)

    # Remove punctuation from both ends of words (except `'s`)
    words = text.split()
    cleaned_words = []

    for word in words:

        word = word.strip("".join(rem))  # Remove punctuation

        temp_word = ""
        for char in word:
            if char in splits:
                if temp_word:
                    cleaned_words.append(temp_word)  # Store collected word
                    temp_word = ""  # Reset buffer
            else:
                temp_word += char  # Build word character by character

        if temp_word:
            cleaned_words.append(temp_word)  # Store last collected word

    final_words=[word.lower() for word in cleaned_words]

    #print (final_words)

    return final_words


# Example usage:
text = "Hello 😊! How's it going? I'm good 😃. Let's go to the park-it's fun! You can't be serious. f*ck this"
print(tokenize(text))


def read_json(filename="final_tokenised_en.json", output_file="testing/output_en.txt"):
    with open(filename, encoding="utf-8") as file:
        content = json.load(file)

    with open(output_file, "w", encoding="utf-8") as out_file:
        for i in tqdm(content, desc="Tokenizing Sentences"):  # Tracking progress here
            out_file.write(str(tokenize(i["text"])) + "\n")


# Uncomment to generate JSON from CSV
# make_json()

# Process and save tokenized output
#read_json()
if __name__ =="__main__":
    make_json(False)
    make_json(True)
