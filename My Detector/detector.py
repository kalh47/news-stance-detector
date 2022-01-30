import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
import string
from nltk.stem.porter import PorterStemmer
import heapq
# gather other project resources
from webscraper import scrape
import ml as ML
# rounding
import math
# optimising
import time
# pp only english characters
from re import match
import unicodedata
# saving object (trained ML)
import pickle
import random


# Load training/testing? datasets
csv_bodies = pd.read_csv('./all_bodies.csv', header=0)
# csv_headlines_stances = pd.read_csv('./all_stances.csv', header=0)  #redundant
csv_headlines = pd.read_csv('./new_all_headlines.csv', header=0)
csv_stances = pd.read_csv('./new_all_stances.csv', header=0)
stances = ['agree', 'discuss', 'unrelated', 'disagree']
print(csv_bodies, "\n", csv_headlines, "\n", csv_stances)

# # old code for reorganising the corpus
# # replace stances with numbers matching their index in the list of stances (for LR)
# csv_headlines_stances['Stance'] = csv_headlines_stances['Stance'].replace([stances[0]], '0')
# csv_headlines_stances['Stance'] = csv_headlines_stances['Stance'].replace([stances[1]], '1')
# csv_headlines_stances['Stance'] = csv_headlines_stances['Stance'].replace([stances[2]], '2')
# csv_headlines_stances['Stance'] = csv_headlines_stances['Stance'].replace([stances[3]], '3')
# csv_headlines_stances['Stance'] = csv_headlines_stances['Stance'].astype(int)
#
# # create new table for unique headlines and their headline IDs,
# # then replace original headlines column with headline IDs
# unique_headlines = []
# headline_ID = 0
# csv_new_headline_stances = pd.DataFrame(columns=['Headline ID', 'Body ID', 'Stance'])
# # iterate through each headline-stance
# for index, row in csv_headlines_stances.iterrows():
#     h = row['Headline']
#     if h not in unique_headlines:
#         headline_ID = len(unique_headlines)
#         unique_headlines.append(h)
#     else:
#         headline_ID = unique_headlines.index(h)
#     csv_new_headline_stances.loc[index] = [headline_ID, row['Body ID'], row['Stance']]
#     if index % 1000 == 0:
#         print(index)
#
# print(csv_new_headline_stances)
# csv_headlines = pd.DataFrame({'Headline': unique_headlines, 'Headline ID': list(range(0, len(unique_headlines)))})
# print(csv_headlines)
# csv_new_headline_stances.to_csv('new_all_stances.csv', index=False)
# csv_headlines.to_csv('new_all_headlines.csv', index=False)
# print(csv_bodies)
# print(csv_headlines_stances)

train_data_h = csv_headlines.copy()
train_data_s = csv_stances.copy()
train_data_b = csv_bodies.copy()
train_X_h = train_data_h[["Headline ID", "Headline"]]
train_X_s = train_data_s[["Body ID", "Headline ID"]]
train_X_b = train_data_b[["Body ID", "articleBody"]]
train_y = train_data_s["Stance"]

use_train_X = [train_X_h, train_X_s, train_X_b]
use_train_y = train_y
print("-Training Data Loaded-")

# number of most frequent words to use for tf idf and bow (only first 1000 used in both functions for timing reasons)
n_most_freq = 5000

test_headline, test_body = scrape("https://www.bbc.co.uk/news/uk-england-london-55730459")
# print(test_headline, test_body)
# print("-Target data loaded-")


# global to optimise function calls
porter = PorterStemmer()
remove_punc = str.maketrans('', '', string.punctuation)

# Preprocessing
def pp(s):
    toktok = ToktokTokenizer()
    # tokenize + lower case + punctuation removal (must do first)
    tokens = toktok.tokenize(s.lower().translate(remove_punc))
    # stop word removal + remove empty word (optional numbers too) + stemming + remove unicode (unencodable chars)
    tokens = [porter.stem(str(unicodedata.normalize('NFD', t).encode('ascii', 'ignore'))[2:-1]) for t in tokens
              if t not in stopwords.words() and t.isalpha() and match('[a-z]', t)]
    return tokens


# NLP - NLP is done separately on bodies and articles and then the vectors corresponding to each body-article pair are
# concatenated to finalise the ML input

# tokenization of a corpus of texts so that a list of tokens (preprocessed representation of words) for each text as
# well as word frequencies across the whole corpus can be returned
def tokenize_text(texts):
    tokenized_texts = []
    word_freq = {}
    for text in texts:
        # pre process text
        tokens = pp(text)
        # for each token in text
        for token in tokens:
            # tally the frequency of word token in the text
            if token not in word_freq.keys():
                word_freq[token] = 1
            else:
                word_freq[token] += 1
        # store processed text
        tokenized_texts.append(tokens)
    return tokenized_texts, word_freq


# TF-IDF
def tfidf(corpus, target, filename):
    # tokenize everything
    tokenized_corpus, word_freq = tokenize_text(corpus)
    # temporary (when loading preprocessed data)
    # (tokenized_corpus, most_freq, idfs) = load_dict(filename)
    # tokenized_corpus = np.asarray(tokenized_corpus, dtype=object)
    if target:
        # tokenized_corpus, word_freq = tokenize_text(corpus)  # temporary
        # if preparing target data, load most frequent tokens from training
        (throw, most_freq, idfs) = load_dict(filename)
    else:
        # keep the n most frequent word tokens across the corpus (and order by descending frequency?)
        most_freq = heapq.nlargest(n_most_freq, word_freq, key=word_freq.get)

        # generate IDFs for each token in most_freq
        idfs = {}
        for token in most_freq:
            texts_with_words = 0
            # count frequency of token in each text
            for text in tokenized_corpus:
                if token in text:
                    texts_with_words += 1
            # calculate IDF
            idfs[token] = np.log(len(tokenized_corpus)/texts_with_words)
        # save for future
        save_dict((tokenized_corpus, most_freq, idfs), filename)
    most_freq = most_freq[:1000]

    # generate TF dictionary - each term (token) has a vector of tfs, each entry is for its tf in each text, so this
    # vector is what will become the column and is done this way to be able to apply idfs to each token easily
    tfs = {}
    for token in most_freq:
        text_tf_vector = []
        for text in tokenized_corpus:
            word_freq = 0
            # tally number of occurrences of most_freq token in text
            for word in text:
                if token == word:
                    word_freq += 1
            # calculate and store term frequency for term in text
            word_tf = word_freq/len(text)
            text_tf_vector.append(word_tf)
        tfs[token] = text_tf_vector

    # create TF-IDF model from results
    tfidfs = []
    for token in tfs.keys():
        tfidf_corpus = []
        for tf_text in tfs[token]:
            tfidf_corpus.append(tf_text * idfs[token])  # append tfidf
        tfidfs.append(tfidf_corpus)
    # convert to numpy arrays for speed optimisation
    tf_idf_model = np.asarray(tfidfs, dtype=object)
    tf_idf_model = np.transpose(tf_idf_model)
    return tf_idf_model


def quickSol(data):
    counts = []
    headlines = data[0]#['Headline'].tolist()
    numOfData = len(headlines)
    # print(headlines)
    bodies = data[1]
    # print(bodies)
    bodyID = -1
    (tokenized_headlines, most_freq, idfs) = load_dict('big_h.txt')
    tokenized_headlines = np.asarray(tokenized_headlines, dtype=object)
    (tokenized_bodies, most_freq, idfs) = load_dict('big_b.txt')
    tokenized_bodies = np.asarray(tokenized_bodies, dtype=object)
    for i in range(0, numOfData):
        count = 0
        h_tokens = tokenized_headlines[i]  # pp(headlines.iloc[i]['Headline'])
        currentBodyID = int(headlines.iloc[i]['Body ID'])
        if currentBodyID != bodyID:
            b_tokens = tokenized_bodies[currentBodyID]  # pp(
            # bodies.loc[bodies['Body ID'] == currentBodyID]['articleBody'].tolist()[0])
            bodyID = currentBodyID
        for token in b_tokens:
            if token in h_tokens:
                count += 1
        # create vector of counts
        counts.append([count])
        if i % math.ceil(numOfData/10) == 0:
            print(str(i)+"/"+str(numOfData))
    print("Finished")
    return np.asarray(counts, dtype=object)


def quickSoltarget(data):
    h_tokens = pp(data[0])
    b_tokens = pp(data[1])
    count = 0
    for token in b_tokens:
        if token in h_tokens:
            count += 1
    # create vector of counts
    return np.asarray([[count]], dtype=object)


# For BoW and TF-IDF, function will take corpus and calculate where each vector is a body or headline (the functions are
# called twice once for all the headlines and once for all the necessary bodies then outside the function this
# is calculated together for ml_input)
def bow(corpus, target, savefilename):
    # corpus of texts prepared before function called
    tokenized_corpus, word_freq = tokenize_text(corpus)
    # alternate (for when loading preprocessed data)
    # (tokenized_corpus, most_freq, throw) = load_dict(savefilename)

    # list of most frequent token is important as it acts as the vocabulary across the corpus
    if target:
        # if preparing target data, load most frequent tokens from training
        (throw, most_freq, throw) = load_dict(savefilename)
    else:
        # keep top n frequent tokens across the corpus
        most_freq = heapq.nlargest(n_most_freq, word_freq, key=word_freq.get)
        # save this data for target NLP and for reducing function call time
        save_dict(most_freq, savefilename)
    most_freq = most_freq[:1000]

    text_vectors = []
    for text_tokens in tokenized_corpus:
        text_vector = []
        # for each n most frequent token
        for token in most_freq:
            # tally
            if token in text_tokens:
                text_vector.append(1)
            else:
                text_vector.append(0)
        text_vectors.append(text_vector)
    text_vectors = np.asarray(text_vectors, dtype=object)
    return text_vectors

# despite being implemented to save dictionaries outside of the running program, these functions can be used to save
# other data structure too
def save_dict(dic, filename):
    f = open(filename, 'w')
    f.write(str(dic))
    f.close()


def load_dict(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return eval(data)


def combine_training_data(h_vectors, b_vectors):
    for index, row in use_train_X[0].iterrows():
        # extra precaution
        body_index = use_train_X[1][use_train_X[1]['Body ID'] == row['Body ID']].index.item()
        temp_h_vector = h_vectors[index]
        temp_b_vector = b_vectors[body_index]
        new_vector = np.concatenate((temp_h_vector, temp_b_vector))
        # # add cosine similarity - can only do if vectors same length, need large data and limiting most_freq
        # cosine_sim = np.dot(temp_h_vector,temp_b_vector)/(np.linalg.norm(temp_h_vector)*np.linalg.norm(temp_b_vector))
        # new_vector = np.concatenate((new_vector, [cosine_sim]))
        if index == 0:
            joint_vectors = np.array([new_vector])
        else:
            joint_vectors = np.append(joint_vectors, [new_vector], axis=0)
        if index % 500 == 0:
            if index % 5000 == 0:
                print("save made ")
                np.savetxt("inter_input.gz", joint_vectors, delimiter=",", fmt="%f")
            print(index)

    #     print(new_vector)
    #     print()
    # print(joint_vectors)
    # print()
    return joint_vectors


# new_combine_training_data(h_vecs, b_vecs, "inter_ml_input_bow.gz")
def new_combine_training_data(h_vectors, b_vectors, filename):
    # vector positions correspond to headline and body IDs
    for index, row in use_train_X[1].iterrows():  # both IDs
        temp_h_vector = h_vectors[row['Headline ID']]
        temp_b_vector = b_vectors[row['Body ID']]
        new_vector = np.concatenate((temp_h_vector, temp_b_vector))
        # add cosine similarity - (can only do if vectors same length, need large data and limiting most_freq)
        cosine_sim = np.dot(temp_h_vector, temp_b_vector)/(np.linalg.norm(temp_h_vector)*np.linalg.norm(temp_b_vector))
        new_vector = np.concatenate((new_vector, [cosine_sim]))
        if index == 0:
            joint_vectors = np.array([new_vector])
        else:
            joint_vectors = np.append(joint_vectors, [new_vector], axis=0)
        if index % 500 == 0:
            if index % 5000 == 0:
                print("save made ")
                np.savetxt(filename, joint_vectors, delimiter=",")
            print(index)
    np.savetxt(filename, joint_vectors, delimiter=",")
    print("completed and saved! ")
    return joint_vectors


def get_stance_from_prediction(p):
    # split possible values into 4 equally sized segments
    if p < 0.75:
        prediction_stance = 0
    elif p < 1.5:
        prediction_stance = 1
    elif p < 2.25:
        prediction_stance = 2
    else:
        prediction_stance = 3
    return prediction_stance


def final_predict_stance(url):
    target_headline, target_body = scrape(url)
    # NLP on target data - TF-IDF with 1000 feature sized bags (2000 total)
    headline_vector = tfidf([target_headline], True, "big_h.txt")
    body_vector = tfidf([target_body], True, "big_b.txt")
    headline_vector = headline_vector.tolist()
    body_vector = body_vector.tolist()
    headline_vector[0].extend(body_vector[0])  # concat to make headline_vector into complete input
    # load ML object (trained using 75380 samples)
    with open('final_ml_algo', 'rb') as config_dictionary_file:
        final_ml_algo = pickle.load(config_dictionary_file)
    final_predictions = final_ml_algo.predict(headline_vector)
    final_prediction = get_stance_from_prediction(final_predictions[0])
    return stances[int(final_prediction)]


# simple menu for testing
while True:
    print()
    print("--== Stance Detector ==--")
    print("30. To use the final stance detection tool enter option 30.")
    print()
    print("--= Debugging Main Menu =--")
    print("Note: Not all options work due to changes in file/variable names and/or contents throughout development")
    print("1. Use Sample Training Data")
    print("2. Use All Training Data")
    print("3. Simple NLP with ML result")
    print("4. Run NLP (Create ML input)")
    print("5. Train and Test ML")
    print("6. load/save trained ml")
    print("7. Load alternate test data")
    print("8. Load ML input")
    print("9. Save ML input")
    print("0. Exit program")
    print()
    print("-Extras-")
    print("10. Num of unique headlines")
    print("11. Custom PP")
    print("12. Gen smaller ml inputs (250,500,750)")
    print("13. ML with custom sample size")
    print("14. ML with custom feature size and custom sample size")
    print("15. Predict first row from training data on above ML")
    print("16. Testing run")
    print("17. Stance spread")
    print("18. Custom train and test")
    print("19. Predict stance on URL (TF-IDF 1000)")
    print("20. k-fold testing with DT, LR, DNN")
    option = input("Enter Option: ")
    print()
    start = time.time()

    if option == "1":
        use_train_X = [train_X_h.iloc[0:10], train_X_s.iloc[0:10], train_X_b.iloc[0:10]]
        use_train_y = train_y.iloc[0:10]
        print("Sample data loaded")
    elif option == "2":
        use_train_X = [train_X_h, train_X_s, train_X_b]
        use_train_y = train_y
        print("All data loaded")
    elif option == "3":
        # call function to skip complex NLP and count number of words in title that appear in text
        ml_input = quickSol(use_train_X)
        # train with correct stances
        svm = ML.SVM(ml_input, use_train_y)
        lr = ML.LR(ml_input, use_train_y)
        dt = ML.DT(ml_input, use_train_y)
        dnn = ML.DT(ml_input, use_train_y)
        # format as dataframe to reuse function
        # # old
        # tempDF1 = pd.DataFrame([[0, test_headline]], columns=['Body ID', 'Headline'])
        # tempDF2 = pd.DataFrame([[0, test_body]], columns=['Body ID', 'articleBody'])
        # test_input = quickSol([tempDF1, tempDF2])
        # new
        test_input = quickSoltarget([test_headline, test_body])
        # print(use_train_y)
        print()
        print("SVM result:", stances[int(svm.predict(test_input)[0])])
        print("LR result:", stances[int(get_stance_from_prediction(lr.predict(test_input)[0]))])
        print("DT result:", stances[int(dt.predict(test_input)[0])])
        print("DNN result:", stances[int(dnn.predict(test_input)[0])])
        print()

    elif option == "4":
        print("--= NLP Menu =--")
        print("1. TF-IDF")
        print("2. BoW")
        print("x 3. Simple")
        option = input("Enter Option: ")
        print()
        start = time.time()
        if option == "1":

            # # generations of seperate nlp products
            h_filename = 'big_h.txt'
            b_filename = 'big_b.txt'
            # # # temp for chunky processing
            # h_vecs = tfidf(use_train_X[0]['Headline'].tolist(), False, h_filename)
            # # np.savetxt("big_h.csv", h_vecs, delimiter=",") # file too large, don't save, nlp is quick without pp
            # b_vecs = tfidf(use_train_X[2]['articleBody'].tolist(), False, b_filename)
            # # np.savetxt("big_b.csv", b_vecs, delimiter=",")
            # np.savetxt("new_h_TFIDF_1000.gz", h_vecs, delimiter=",")
            # np.savetxt("new_b_TFIDF_1000.gz", b_vecs, delimiter=",")

            # generating final ml input - concatenating to create final training input
            # h_vecs = np.loadtxt("new_h_TFIDF_1000.gz", delimiter=",")
            # b_vecs = np.loadtxt("new_b_TFIDF_1000.gz", delimiter=",")
            # print("loaded")
            # print()
            # ml_input = new_combine_training_data(h_vecs, b_vecs, "train_input_TFIDF.gz")
            # print("training input done", ml_input)
            # print()

            # Load prepared ml_input
            # ml_input = np.loadtxt("train_input_TFIDF.gz", delimiter=",")
            # ml_input = ml_input.astype(np.float) #temp
            # np.savetxt("TFIDF_input_float.gz", ml_input, delimiter=",") #temp

            ml_input = np.loadtxt("TFIDF_input_float.gz", delimiter=",")
            ml_input = np.asarray(ml_input, dtype=np.float32)
            print("ML training input loaded")
            print(ml_input)
            if np.isnan(ml_input.any()):
                print("has NaN or inf")
                print("coords", np.where(np.isnan(ml_input)))
                np.nan_to_num(ml_input)
                print("fixed")
            else:
                print("no NaN or inf")
            print()

            # NLP on target data
            h_vec = tfidf([test_headline], True, h_filename)
            b_vec = tfidf([test_body], True, b_filename)
            target_input = np.concatenate((h_vec[0], b_vec[0]))
            # add cosine similarity
            cosine_sim = np.dot(h_vec[0], b_vec[0])/(np.linalg.norm(h_vec[0])*np.linalg.norm(b_vec[0]))
            target_input = np.asarray([np.concatenate((target_input, [cosine_sim]))], dtype=np.float32)
            # target_input = target_input.astype(np.float)
            print("target input done", target_input)
            print()

        elif option == "2":
            # NLP template
            # done -produce vectors for all examples (all article bodies is assuming all the write ones were given for
            # the headlines given) as two separate bags,
            # done? -within the functions, structures must be stored so that the exact same process is used on the
            # target data as the training data
            # storing dictionaries as .txt files

            # # generations of seperate nlp products
            h_filename = 'big_h.txt'  # .npy, .csv
            b_filename = 'big_b.txt'
            # h_vecs = bow(use_train_X[0]['Headline'].tolist(), False, h_filename)
            # b_vecs = bow(use_train_X[2]['articleBody'].tolist(), False, b_filename)
            # np.savetxt("new_h_BoW_1000.gz", h_vecs, delimiter=",")
            # np.savetxt("new_b_Bow_1000.gz", b_vecs, delimiter=",")

            # # concatenating to create final training input
            # h_vecs = np.loadtxt("new_h_BoW_1000.gz", delimiter=",")
            # b_vecs = np.loadtxt("new_b_Bow_1000.gz", delimiter=",")
            # # -piece together the results so that all the headline vectors are concatenated with their bodies vectors
            # # and kept in order for fit stances later
            # print("loaded")
            # print()
            # ml_input = new_combine_training_data(h_vecs, b_vecs, "train_input_BoW.gz")
            # print("training input done", ml_input)
            # print()

            # Load prepared ml_input
            ml_input = np.loadtxt("train_input_BoW.gz", delimiter=",")
            ml_input = np.asarray(ml_input, dtype=np.float32)
            # tf.reshape(ml_input, [len(ml_input), len(ml_input[0])])
            # ml_input = ml_input[:1000] # temp debugging, didnt help
            # use_train_y = use_train_y[:1000]
            print("ML training input loaded")
            print()
            print("ml shape:")
            print(len(ml_input), len(ml_input[0]))
            print()
            print("ml answers shape:")
            print(len(use_train_y))
            print()

            #  -(use this result to train, store this result and the other structures so all can be used to quickly
            # do predictions with any ml)
            #  -target prep here
            h_vec = bow([test_headline], True, h_filename)
            b_vec = bow([test_body], True, b_filename)
            target_input = np.concatenate((h_vec[0], b_vec[0]))
            # add cosine similarity
            cosine_sim = np.dot(h_vec[0], b_vec[0])/(np.linalg.norm(h_vec[0])*np.linalg.norm(b_vec[0]))
            target_input = np.asarray([np.concatenate((target_input, [cosine_sim]))], dtype=np.float32)
            print("target input done", target_input)
            print()

            print("target shape:")
            print(len(target_input[0]))
            print()

        elif option == "3":
            ml_input = quickSol(use_train_X)
            target_input = quickSoltarget([test_headline, test_body])

    elif option == "5":
        print("--= ML Menu =--")
        print("1. SVM")
        print("2. DT")
        print("3. LR")
        print("4. DNN")
        print("5. Skip Training")
        option = input("Enter Option: ")
        print()
        start = time.time()
        if option == "1":
            ml_algo = ML.SVM(ml_input, use_train_y)
        elif option == "2":
            ml_algo = ML.DT(ml_input, use_train_y)
        elif option == "3":
            ml_algo = ML.LR(ml_input, use_train_y)
        elif option == "4":
            ml_algo = ML.DNN(ml_input, use_train_y)
        prediction = ml_algo.predict(target_input)
        prediction = get_stance_from_prediction(prediction)
        print(stances[int(prediction)])
    elif option == "6":
        print("1 = save, 2 = load")
        option = input("Enter Option: ")
        print()
        start = time.time()
        if option == "1":
            # SAVE
            with open('final_ml_algo', 'wb') as config_dictionary_file:
                pickle.dump(temp_ml_algo, config_dictionary_file)
        elif option == "2":
            # LOAD
            with open('final_ml_algo', 'rb') as config_dictionary_file:
                temp_ml_algo = pickle.load(config_dictionary_file)
    elif option == "7":
        input_url = input("Enter URL: ")
        print()
        start = time.time()
        test_headline, test_body = scrape(input_url)
        print("loaded.")

    elif option == "8":
        file = input("Enter file name: ")
        print()
        start = time.time()
        # Retrieve training arrays
        ml_input = np.loadtxt(file, delimiter=",")
        print(file, " loaded.")
        print("samples:", len(ml_input), "features per sample:", len(ml_input[0]))

    elif option == "9":
        file = input("Enter file name: ")
        print()
        start = time.time()
        # Store training arrays
        np.savetxt(file, ml_input, delimiter=",", fmt="%f")
        # save_dict(ml_input, file)
    elif option == "10":
        h_unique = []
        for h_raw in use_train_X[0]["Headline"]:
            # print(h_raw)
            if h_raw not in h_unique:
                h_unique.append(h_raw)
        print("total unique headlines = ", len(h_unique), "/", len(use_train_X[0]))

    elif option == "11":
        custom_input = input("Enter text: ")
        print()
        start = time.time()
        print(pp(custom_input))
    elif option == "12":
        # ml_input = np.loadtxt("train_input_BoW.gz", delimiter=",")
        ml_input = np.loadtxt("train_input_TFIDF.gz", delimiter=",")
        # convert to list for faster joining operations
        ml_input = ml_input.tolist()
        print("full matrix loaded")
        print("dimensions", len(ml_input), len(ml_input[0]))
        index_counter = 0
        ml_input250_array = []
        ml_input500_array = []
        ml_input750_array = []
        ml_input1000_array = []

        for row in ml_input:
            # using known indexing to make each row (num of features in sample) smaller
            # currently 1000 features per bag and 1 (broken) cosine similarity feature (2001 per row)
            # 250 features per bag
            new_row = row[0:250]
            new_row.extend(row[1000:1250])
            ml_input250_array.append(new_row)
            # 500 features per bag
            new_row = row[0:500]
            new_row.extend(row[1000:1500])
            ml_input500_array.append(new_row)
            # 750 features per bag
            new_row = row[0:750]
            new_row.extend(row[1000:1750])
            ml_input750_array.append(new_row)
            # 1000 features per bag
            new_row = row[0:2000]
            ml_input1000_array.append(new_row)

            if index_counter % 5000 == 0:
                print("index passed:", index_counter)
            index_counter += 1
        # np.savetxt("train_input_BoW_250.gz", ml_input250_array, delimiter=",", fmt="%f")
        # np.savetxt("train_input_BoW_500.gz", ml_input500_array, delimiter=",", fmt="%f")
        # np.savetxt("train_input_BoW_750.gz", ml_input750_array, delimiter=",", fmt="%f")
        # np.savetxt("train_input_BoW_1000.gz", ml_input1000_array, delimiter=",", fmt="%f")
        np.savetxt("train_input_TFIDF_250.gz", ml_input250_array, delimiter=",", fmt="%f")
        np.savetxt("train_input_TFIDF_500.gz", ml_input500_array, delimiter=",", fmt="%f")
        np.savetxt("train_input_TFIDF_750.gz", ml_input750_array, delimiter=",", fmt="%f")
        np.savetxt("train_input_TFIDF_1000.gz", ml_input1000_array, delimiter=",", fmt="%f")
        print("4 files saved!")

    elif option == "13":
        # only take first n rows
        nsamples = int(input("num of samples to test: "))
        start = time.time()
        temp_ml_input = ml_input[:nsamples]
        temp_use_train_y = use_train_y[:nsamples]

        # temp_ml_algo = ML.SVM(temp_ml_input, temp_use_train_y)
        # print("SVM trained")
        # temp test each ml works
        temp_ml_algo = ML.DT(temp_ml_input, temp_use_train_y)
        print("DT trained")
        # temp_ml_algo = ML.LR(temp_ml_input, temp_use_train_y)
        # print("LR trained")
        # temp_ml_algo = ML.DNN(np.asarray(temp_ml_input), np.asarray([[y] for y in temp_use_train_y]))
        # print("DNN trained")

    elif option == "14":
        # only take first n rows
        nsamples = int(input("num of samples to test: "))
        # keep first feature of each sample for testing
        nfeatures = int(input("num of features to test: "))
        start = time.time()
        temp_ml_input = [sample[:nfeatures] for sample in ml_input[:nsamples]]
        temp_use_train_y = use_train_y[:nsamples]

        temp_ml_algo = ML.SVM(temp_ml_input, temp_use_train_y)
        print("SVM trained")
        # temp test each ml works
        temp_ml_algo = ML.DT(temp_ml_input, temp_use_train_y)
        print("DT trained")
        temp_ml_algo = ML.LR(temp_ml_input, temp_use_train_y)
        print("LR trained")
        # dnn input must be numpy arrays
        temp_ml_algo = ML.DNN(np.asarray(temp_ml_input), np.asarray([[y] for y in temp_use_train_y]))
        print("DNN trained")
        # single feature sample size can go above 403 to at least 50000 (9.5 seconds to train)
        # max 75380 works and took 27.5 seconds to train

        # 200 features and 1000 samples works
        # 251 and 50k works. and 403 features?
        # but not 501
        # its the cosine similarity????????

    elif option == "15":

        temp_test_input = [ml_input[0]]
        print(stances[int(get_stance_from_prediction(temp_ml_algo.predict(temp_test_input)))])

    elif option == "16":
        # testing
        f = open("testresults.txt", "w")
        start = time.time()
        storedinputs = ["train_input_BoW_250.gz", "train_input_BoW_500.gz", "train_input_BoW_750.gz",
                        "train_input_BoW_1000.gz", "train_input_TFIDF_250.gz", "train_input_TFIDF_500.gz",
                        "train_input_TFIDF_750.gz", "train_input_TFIDF_1000.gz"]
        ntests = 2000
        test_index = 68000
        test_actual = use_train_y[test_index:test_index+ntests].to_list()
        for filename in storedinputs:
            f.write(filename+"\n")
            ml_input = np.loadtxt(filename, delimiter=",")
            test_input = ml_input[test_index:test_index+ntests]
            print(filename, "loaded")
            for i in range(4):
                # temp_ml_input = ml_input[i*18845:(i+1)*18845]
                # temp_use_train_y = use_train_y[i*18845:(i+1)*18845]
                temp_ml_input = ml_input[(i+1)*10000:(i+2)*10000]
                temp_use_train_y = use_train_y[(i+1)*10000:(i+2)*10000]

                # random testing
                # ntests = 2000
                # while True:
                #     test_index = random.randint(0, 75380-ntests)
                #     if test_index not in range(i*18845, (i+1)*18845) and (test_index + 2000) not in \
                #             range(i*18845, (i+1)*18845):
                #         break
                # test_input = ml_input[test_index:test_index+ntests]
                # test_actual = use_train_y[test_index:test_index+ntests].to_list()
                for m in range(4):
                    if m == 0:
                        temp_ml_algo = ML.SVM(temp_ml_input, temp_use_train_y)
                        print(i, "SVM trained")
                        f.write("SVM: ")
                    elif m == 1:
                        temp_ml_algo = ML.DT(temp_ml_input, temp_use_train_y)
                        print(i, "DT trained")
                        f.write("DT: ")
                    elif m == 2:
                        temp_ml_algo = ML.LR(temp_ml_input, temp_use_train_y)
                        print(i, "LR trained")
                        f.write("LR: ")
                    elif m == 3:
                        temp_ml_algo = ML.DNN(np.asarray(temp_ml_input), np.asarray([[y] for y in temp_use_train_y]))
                        print(i, "DNN trained")
                        f.write("DNN: ")

                    predictions = temp_ml_algo.predict(test_input)
                    correct = 0
                    # print(test_index)#, predictions, test_actual)
                    for t_i in range(0, ntests):
                        if get_stance_from_prediction(predictions[t_i]) == test_actual[t_i]:
                            correct += 1
                    print("accuracy=", correct, "/2000")
                    f.write(str(correct)+"/2000\n")

                current = time.time()
                print("Elapsed time:", current-start)
        f.close()

    elif option == "17":
        # finding a diverse enough section of the data results for testing.
        for findex in range(0, 73380, 2000):
            lindex = findex+2000
            stances = use_train_y[findex:lindex].to_list()
            spread = [0, 0, 0, 0]
            for n in stances:
                spread[n-1] = spread[n-1]+1
            print(findex, "to", lindex, "has spread: ", spread)

    elif option == "18":
        # only take first n rows past index 10000
        nsamples = int(input("num of samples to train with: "))
        start = time.time()
        temp_ml_input = ml_input[10000:nsamples]
        temp_use_train_y = use_train_y[10000:nsamples]
        ntests = 6000
        test_index = 68000
        test_input = ml_input[test_index:test_index+ntests]
        test_actual = use_train_y[test_index:test_index+ntests].to_list()
        for m in range(1,4):
            if m == 0:
                temp_ml_algo = ML.SVM(temp_ml_input, temp_use_train_y)
                print("SVM trained")
            elif m == 1:
                temp_ml_algo = ML.DT(temp_ml_input, temp_use_train_y)
                print("DT trained")
            elif m == 2:
                temp_ml_algo = ML.LR(temp_ml_input, temp_use_train_y)
                print("LR trained")
            elif m == 3:
                temp_ml_algo = ML.DNN(np.asarray(temp_ml_input), np.asarray([[y] for y in temp_use_train_y]))
                print("DNN trained")
            predictions = temp_ml_algo.predict(test_input)
            correct = 0
            # print(test_index)#, predictions, test_actual)
            for t_i in range(0, ntests):
                if get_stance_from_prediction(predictions[t_i]) == test_actual[t_i]:
                    correct += 1
            print("accuracy=", correct, "/2000")

    elif option == "19":
        input_url = input("Enter URL: ")
        start = time.time()
        test_headline, test_body = scrape(input_url)
        # NLP on target data
        h_vec = tfidf([test_headline], True, "big_h.txt")
        b_vec = tfidf([test_body], True, "big_b.txt")
        print(h_vec, b_vec)
        h_vec = h_vec.tolist()
        b_vec = b_vec.tolist()
        h_vec[0].extend(b_vec[0])  # concat to make complete input
        predictions = temp_ml_algo.predict(h_vec)
        prediction = get_stance_from_prediction(predictions[0])
        print(stances[int(prediction)])

    elif option == "20":
        # testing
        # f = open("kFoldTesting.txt", "w") # rewrites
        f = open("kFoldTesting.txt", "a")
        start = time.time()
        #do 3 at a time
        # storedinputs = ["train_input_BoW_250.gz", "train_input_BoW_500.gz", "train_input_BoW_750.gz",
        # storedinputs = ["train_input_BoW_1000.gz", "train_input_TFIDF_250.gz", "train_input_TFIDF_500.gz",
        #                 "train_input_TFIDF_750.gz",
        storedinputs = ["train_input_TFIDF_1000.gz"]
        for filename in storedinputs:
            # save after each file
            f.close()
            f = open("kFoldTesting.txt", "a")
            f.write(filename+"\n")
            ml_input = np.loadtxt(filename, delimiter=",")
            ml_input = ml_input.tolist()
            print(filename, "loaded")
            samplesize = len(ml_input)
            ntests = int(samplesize/4)
            for m in range(1, 4):
                accuracies = []
                for i in range(4):
                    # copy training data without test data
                    print("training ranges:")
                    if i == 0:
                        temp_ml_input = ml_input[(i+1)*ntests:samplesize]
                        temp_use_train_y = use_train_y[(i+1)*ntests:samplesize].to_list()
                        print((i+1)*ntests, samplesize)
                    else:
                        temp_ml_input = ml_input[0:i*ntests]
                        temp_use_train_y = use_train_y[0:i*ntests].to_list()
                        print(0, i*ntests)
                        if (i+1)*ntests < samplesize:
                            temp_ml_input.extend(ml_input[(i+1)*ntests:samplesize])
                            temp_use_train_y.extend(use_train_y[(i+1)*ntests:samplesize].to_list())
                            print((i+1)*ntests, samplesize)
                    # copy test data
                    test_input = ml_input[i*ntests:(i+1)*ntests]
                    test_actual = use_train_y[i*ntests:(i+1)*ntests].to_list()
                    print("testing ranges:", i*ntests, (i+1)*ntests)
                    if m == 1:
                        temp_ml_algo = ML.DT(temp_ml_input, temp_use_train_y)
                        print(i, "DT trained")
                        f.write("DT: ")
                    elif m == 2:
                        temp_ml_algo = ML.LR(temp_ml_input, temp_use_train_y)
                        print(i, "LR trained")
                        f.write("LR: ")
                    elif m == 3:
                        temp_ml_algo = ML.DNN(np.asarray(temp_ml_input), np.asarray([[y] for y in temp_use_train_y]))
                        print(i, "DNN trained")
                        f.write("DNN: ")
                    predictions = temp_ml_algo.predict(test_input)
                    correct = 0
                    for t_i in range(0, ntests):
                        if get_stance_from_prediction(predictions[t_i]) == test_actual[t_i]:
                            correct += 1
                    print("accuracy=", correct, "/", ntests)
                    currentaccuracy = correct*100/ntests
                    accuracies.append(currentaccuracy)
                    f.write(str(correct)+"/"+str(ntests)+" | accuracy:"+str(currentaccuracy)+"% \n")
                averageaccuracy = sum(accuracies)/len(accuracies)
                f.write("average accuracy: "+str(averageaccuracy)+"% \n")
                current = time.time()
                print("Elapsed time:", current-start)
        f.close()

    elif option == "30":
        input_url = input("Enter URL: ")
        start = time.time()
        print(final_predict_stance(input_url))

    elif option == "0":
        print("exiting...")
        print()
        break
    end = time.time()
    print("Elapsed time:", end-start)


