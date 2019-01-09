#!/usr/bin/env python3
# orient.py 
#
"""
Team members: Yuhan Zeng, Hai Hu, Piyush Vyas
"""

import sys
import math
import numpy as np
import heapq as hp
import random
import pickle
import shutil #Used for copying train-data.txt to nearest-model.txt


# Process input file data. The resulted data is a numpy array of [photo_id, photo_orient, array(pixels)]
def processInput(fname):
    result = []
    with open(fname, 'r') as f:
        for line in f:
            entry = line.split(" ")
            photo_id = entry[0]
            photo_orient = int(entry[1])
            photo_pixels = np.array(entry[2:], dtype=int)
            result.append([photo_id, photo_orient, photo_pixels])
    return result

'''
---------------------------------------------------
1. K nearest neighbors 
Author: Yuhan Zeng
Highest Accuracy on test-data.txt: 72.90%
---------------------------------------------------
'''
# Calculate Euclidean distance between two vectors
def Euclidean_dist(vetor1, vector2):
    return math.sqrt(np.sum(np.subtract(vetor1, vector2) ** 2))


# The training of knn is copying train-data into model_file
def knn_train(train_file):
   #os.system('cp %s %s' %(train_file, model_file))
   shutil.copyfile(train_file, model_file)
    
# Calculate the distance of test date to each train data, push them onto a heapq (priority queue),
# then find the k nearest neighbors to vote.
def knn_test(test_data, train_data):
    labels = []
    for data in test_data:
        label = knn_classify(data, train_data)
        labels.append([data[0], data[1], label])
    # write the test data with estimated labels into knn-output.txt
    with open('output_nearest.txt', 'w') as filehandle:
        for listitem in labels:
            filehandle.write('%s %s\n' % (listitem[0], listitem[2]))
    # return accuracy 
    return accuracy(labels) 

def accuracy(labels):
    # labels [photo_id, original_label, predicted_label]
    correct = 0
    for row in labels:
        if row[2] == row[1]:
            correct += 1
    return float(correct) / len(labels)

'''
Calculate the distance of test entry to all train data, push them onto a heapq (priority queue),
then find the k nearest neighbors to vote for the orientation of this test entry.
'''
def knn_classify(test_entry, train_data):
    distances = []
    #print(train_data)
    for data in train_data:
        dist = Euclidean_dist(test_entry[2], data[2])
        # distances is an array of the train data, [[distance, photo_id, photo_orient]], and distance will be used as a key for min-heap sort
        distances.append([dist, data[0], data[1]])
    # Get the k nearest neighbors from distances array
    #print(distances)
    neighbors = hp.nsmallest(k, distances)
    #print(neighbors)
    labels = [0, 0, 0, 0] # Used to store the number of votes for labels (0, 90, 180, 270)
    for neighbor in neighbors:
        i = int(neighbor[2] / 90)
        labels[i] += 1
    j = 0
    for i in range(0, 4):
        if (labels[i] > labels[j]):
            j = i
    return j * 90 # return the mostly voted orientation of this test data entry


'''
------------------------------------------------------------------------------------------
3. Random Forest. Highest Accuracy on test-data.txt: 74.56%
Author: Yuhan Zeng
Code reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/
------------------------------------------------------------------------------------------
'''

# Calculate Gini for a given node
def gini(children, classes):
    # number of records in the children groups
    num_records = 0
    for group in children:
        num_records += len(group)
    # Count the fraction of records p for each class
    gini = 0.0
    for group in children:
        size = len(group)
        if (size == 0):
            continue
        group_orients = [entry[1] for entry in group]
        p_sum = 0.0
        for x in classes:
            p = float(group_orients.count(x)) / size
            p_sum += p**2
        # calculate total weighted gini of each child group
        gini += (float(size) / num_records) * (1.0 - p_sum)
    return gini

# Calculate Entropy for a given node
def entropy(children, classes):
    # number of records in the children groups
    num_records = 0
    for group in children:
        num_records += len(group)
    # Count the fraction of records p for each class
    entropy = 0.0
    for group in children:
        size = len(group)
        if (size == 0):
            continue
        group_orients = [entry[1] for entry in group]
        p_sum = 0.0
        for x in classes:
            p = float(group_orients.count(x)) / size
            p_sum += p * math.log10(p)
        # calculate total weighted entropy of each child group
        entropy += (float(size) / num_records) * (-1) * p_sum
    return entropy


# Split a set of data records based on a given feature (pixel) and a threshold, return the children
# Code reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/
# Return type: a tuple of lists, each list is a list of data points
def split_trial(records, pixel, threshold):
    left = list()
    right = list()
    for entry in records:
        if entry[2][pixel] < threshold:
            left.append(entry)
        else:
            right.append(entry)
    children = (left, right)
    return children

# Get the best split of a set of records, and return this node
# Code reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/
# Return type: dict
def best_split(records, num_features):
    # Get a list of all the unique orients in this dataset (set() removes duplicates)
    orients = list(set(entry[1] for entry in records))
    best_pixel, best_threshold, best_impurity, best_children = 0, 0, 9999, None
    features = []
    # Randomly choose a certain number of pixels as features for split
    i = 0
    while i <= num_features:
        # Generate a random pixel index to use as the candidate feature
        pixel = random.randrange(len(records[0][2])) 
        if pixel not in features:
            features.append(pixel)
            i += 1
    for pixel in features:
        for entry in records:
            # 'pixel' is the index of pixel used as feature, and entry[2][pixel] is the value used as threshold
            children = split_trial(records, pixel, entry[2][pixel])
            impurity = gini(children, orients)
            #impurity = entropy(children, orients)
            if impurity < best_impurity:
                best_pixel, best_threshold, best_impurity, best_children = \
                pixel, entry[2][pixel], impurity, children
    node = {'isLeaf': False, 'pixel':best_pixel, 'threshold':best_threshold, 'children':best_children}
    return node

# Make a leaf node, return the orient label of this group of records
def make_leaf(records):
    # Get a list of all the classes in this group
    orients = list(entry[1] for entry in records)
    # Find the major class of this group as its orient label
    max_count = 0
    for x in set(orients):
        count = orients.count(x)
        if count > max_count:
            max_count = count
            label = x
    return {'isLeaf': True, 'label': label}
    
# Recursively split a node or make the node a leaf
# Code reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/
def recur_split(node, max_depth, min_size, num_features, depth):
    (left_child, right_child) = node['children']
    del(node['children']) # Delete value 'children' from node because it's no longer needed
    # If either left or right is empty, make a leaf
    if len(left_child) == 0 or len(right_child) == 0:
        node['left'] = node['right'] = make_leaf(left_child + right_child)
        return
    # If max_depth has been reached, make a leaf
    if depth >= max_depth:
        node['left'] = make_leaf(left_child)
        node['right'] = make_leaf(right_child)
        return
    # Split left child or make leaf
    if len(left_child) < min_size:
        node['left'] = make_leaf(left_child)
    else:
        node['left'] = best_split(left_child, num_features)
        recur_split(node['left'], max_depth, min_size, num_features, depth+1)
     # Split right child or make leaf
    if len(right_child) < min_size:
        node['right'] = make_leaf(left_child)
    else:
        node['right'] = best_split(right_child, num_features)
        recur_split(node['right'], max_depth, min_size, num_features, depth+1)
    return
    
# Classify a datapoint with a decision tree starting at node as root
def tree_classify(data, node):
    pixel = node['pixel']
    threshold = node['threshold']
    if data[2][pixel] < threshold:
        if node['left']['isLeaf'] == False:
            return tree_classify(data, node['left'])
        else:
            return node['left']['label']
    else:
        if node['right']['isLeaf'] == False:
            return tree_classify(data, node['right'])
        else:
            return node['right']['label']


# Grow a decision tree and return its root
def grow_tree(records, max_depth, min_size, num_features):
    root = best_split(records, num_features)
    recur_split(root, max_depth, min_size, num_features, 1)
    return root

# Randomly sample a portion of data from train data to train each decision tree
def sample_data(dataset, portion):
    sample = []
    added = []
    sample_size = math.floor(len(dataset) * portion)
    i = 0
    while i <= sample_size:
        row = random.randrange(len(dataset))
        if row not in added:  
            sample.append(dataset[row])
            added.append(row)
            i += 1
    return sample

'''
# Grow a random forest with a given number of decision trees (nums_trees) and given portion of dataset as the subsample size
def grow_forest(dataset, portion, num_trees, max_depth, min_size, num_features):
    forest = []
    i = 0
    while i < num_trees:
        sample = sample_data(dataset, portion)
        tree = grow_tree(sample, max_depth, min_size, num_features)
        forest.append(tree)
        i += 1
    return forest
'''

# Classify a datapoint with a given forest
def forest_classify(data, forest):
    labels = []
    for tree in forest:
        label = tree_classify(data, tree)
        labels.append(label)
    # Find the major class of labels list as its orient label
    max_count = 0
    for x in set(labels):
        count = labels.count(x)
        if count > max_count:
            max_count = count
            label = x
    return label

# Train a random forest with given parameters and output the forest
def train_forest(train, portion, num_trees, max_depth, min_size, num_features):
    forest = []
    i = 0
    while i < num_trees:
        sample = sample_data(train, portion)
        tree = grow_tree(sample, max_depth, min_size, num_features)
        forest.append(tree)
        i += 1
    # write the forest into forest-model.npy
    np.save(model_file, forest)

# random forest algorithm given train and test data, and all variable parameters
def test_forest(test):
    #forest = grow_forest(train, portion, num_trees, max_depth, min_size, num_features)
    forest = np.load(model_file)
    labels = []
    fout = open('output_forest.txt', 'w')
    for data in test:
        label = forest_classify(data, forest)
        labels.append([data[0], data[1], label]) # [photo_id, correct_label, predicted_label]
        fout.write("{} {}\n".format(data[0], label))
    fout.close()
    return accuracy(labels)


'''
------------------------------------------------------------------------------------------
3. Adaboost. Highest Accuracy on test-data.txt: 70.91%
Author: Hai Hu

idea:
build 4 adaboost classifiers, CL{1-4}, to do ``one vs. all'' classification.
e.g. for one data point x,
CL1(x)=-0.4, CL2(x)=-0.2, CL3(x)=0.4, CL4(x)=0.1, then we predict it is class3.

------------------------------------------------------------------------------------------
'''

def adaboost_train(input_file, model_file):
    print('training using:', input_file)
    error_thresh = 0.49  ## TODO hyperparameter
    print('error_thresh:', error_thresh)
    hypos_all = {}
    data = np.loadtxt(input_file, dtype=str, delimiter=' ')
    for myclass in [0, 90, 180, 270]:
        hypo_oneclass = adaboost_train_oneclase(data, myclass, error_thresh)
        hypos_all[myclass] = hypo_oneclass

    # save hypos
    model_file = "thresh_"+str(error_thresh)+"_"+model_file+".pkl"
    pickle.dump(hypos_all, open(model_file, "wb"), protocol=4)
    print("model saved to {}".format(model_file))

def adaboost_train_oneclase(data, myclass, error_thresh):
    print('training for class:', myclass)
    gold = data[:,1].astype(int)
    feat_mat = data[:,2:].astype(int)
    numTrain = gold.shape[0]  # number of training examples
    numFeats = feat_mat.shape[1]

    # if gold == [0,90,180,270], then gold = 1; else gold = -1
    gold = np.where(gold == myclass, 1, -1)
    # gold = [1,1,-1,-1,1,-1,...]

    # initialize hypo
    hypos = {}  # {(i,j): weight, ... }
    # initialize weights for X
    w = np.zeros((numTrain,))
    w.fill(1/numTrain)

    counter = 0

    # adaboost loop
    for i in range( numFeats ):
        for j in range( (i+1), numFeats ):
            # 1: learn a hypo #

            # 2: compute error

            # main idea: we need to select hypothesis with error < thresh
            # compute error:
            # h1: if col_5 >= col_10, 1; else -1, h1 = [1,1,1,1,-1,-1,...]
            # then np.multiply(gold, h1) computes correctly/wronly classified

            # first find out wrongly classified cases
            tmp = np.subtract(feat_mat[:, i], feat_mat[:, j])
            h = np.where(tmp > 0, 1, -1)  # feat_mat[:, i] > feat_mat[:, j]
            res = np.multiply(gold, h)    # if 1, correct; if -1, wrong
            wrong_idx = np.argwhere(res < 0)  # idx of wrongly classified
            error = w[wrong_idx].sum()

            i2j = True  # flag of whether i>j or i<j

            if error < 0.5:
                hypo = (i, j)
            else:
                hypo = (j, i)
                # recompute h, res, error
                h = np.where(tmp > 0, -1, 1)  # feat_mat[:, i] < feat_mat[:, j]
                res = np.multiply(gold, h)  # if 1, correct; if -1, wrong
                error = 1-error
                i2j = False  # flip flag

            if error > error_thresh: continue

            counter += 1
            term_update = error / (1-error)

            # 3: update weights
            corr_idx = np.argwhere(res > 0)  # idx of correctly classified
            new = w[corr_idx] * term_update
            np.put(w, corr_idx, new)  # update w

            # 4: normalize
            s = w.sum()
            w = w / s

            # 5: update weight for k in alpha
            if i2j: hypos[hypo] = math.log(1/term_update)
            else: hypos[hypo] = math.log(1/term_update)

    print('num of hypos: ', counter)
    return hypos

def adaboost_classify(v, hypos):
    """ classify one instance. return a float (-1,1)
    v: feat vector for the instance
    hypos: a dictionary { (i,j):weight ... }, which means
           v[i] > v[j] => 1 * weight; v[i] < v[j] => -1 * weight;
    """
    ans = 0
    for hypo in hypos:
        i, j = hypo[0], hypo[1]
        if v[i] - v[j] > 0: ans += hypos[hypo]
        else: ans -= hypos[hypo]
    return ans

def adaboost_test(input_file, model_file):
    """ for each test data point, we want four numbers:
    sum(model_1_hypo * model_1_weights)
    sum(model_2_hypo * model_2_weights)
    sum(model_3_hypo * model_3_weights)
    sum(model_4_hypo * model_4_weights)
    where each number presents the ``likelihood'' of the data point
    belonging to that class: the number is in (-1,1), the greater the number
    the more likely
    """
    print('testing with', input_file)
    data = np.loadtxt(input_file, dtype=str, delimiter=' ')
    photo_id = data[:,0]
    gold = data[:,1].astype(int)
    feat_mat = data[:,2:].astype(int)
    numTest = gold.shape[0]  # number of test examples

    # read in model
    hypos_all = pickle.load(open(model_file, "rb"))

    # predictions
    preds = { i : {0:0, 90:0, 180:0, 270:0,
                   'max_prob':-2, 'max_class':None} for i in range(numTest) }
    for myclass in [0, 90, 180, 270]:
        hypos = hypos_all[myclass]
        adaboost_test_oneclass(feat_mat, hypos, preds, myclass)

    # compute accuracy, output
    f_out = open('output_adaboost.txt', 'w')
    numCorr = 0
    for i in range(len(preds)):
        for myclass in [0, 90, 180, 270]:
            if preds[i][myclass] > preds[i]['max_prob']:
                preds[i]['max_prob'] = preds[i][myclass]
                preds[i]['max_class'] = myclass
        pred = preds[i]['max_class']
        f_out.write("{} {}\n".format(photo_id[i], pred))
        if pred == gold[i]: numCorr += 1

    f_out.close()

    print('acc:', numCorr / numTest)
    return numCorr / numTest

def adaboost_test_oneclass(feat_mat, hypos, preds, myclass):
    """ do prediction for one class
    and fill the number for that class """
    print('testing class', myclass)
    for row_idx in range(feat_mat.shape[0]):
        v = feat_mat[row_idx]
        preds[row_idx][myclass] = adaboost_classify(v, hypos)

'''
----------------------------
            Main
----------------------------
'''
# Process command line
(action, input_file, model_file, model) = sys.argv[1:] 
'''
--------------------- KNN Main ----------------------
Command for training:
    train train-data.txt nearest_model.txt nearest
Command for testing:
    test test-data.txt nearest_model.txt nearest
----------------------------------------------------
'''
if model == 'nearest':
    if action == 'train':
        knn_train(input_file)
    else:
        k = 48 # Number of nearest neighbors
        test_data = processInput(input_file)
        train_data = processInput(model_file)
        knn_accuracy = knn_test(test_data, train_data)
        print('k=', k, ', KNN accuracy is:', knn_accuracy * 100, '%')        
        
'''
-------------- Random Forest Main ------------------
Command for training:
    train train-data.txt forest_model.txt forest
Command for testing:
    test test-data.txt forest_model.txt forest
----------------------------------------------------
'''
if model == 'forest':
    portion = 0.05 # the percentage of train data subsampled to train one decision tree.
    num_trees = 400 # number of decision trees in the forest.
    max_depth = 48 # the maximum depth of one decision tree.
    min_size = 1 # the minimum size of one decision tree.
    num_features = 24 # the number of pixels subsampled as the split criteria.
    if action == 'train':
        train_data = processInput(input_file)
        train_forest(train_data, portion, num_trees, max_depth, min_size, num_features)
    else:
        test_data = processInput(input_file)
        forest_accuracy = test_forest(test_data)
        # print('Data portion:', portion, 'number of trees:', num_trees, 'max depth:', max_depth, 'number of features:', num_features)
        print('Model:', model_file)
        print('Forest accuracy:', forest_accuracy * 100, '%')

'''
--------------------- Adaboost Main --------------------
Command for training:
    train train-data.txt adaboost_model.txt adaboost
Command for testing:
    test test-data.txt adaboost_model.txt adaboost
----------------------------------------------------
'''
if model == 'adaboost':
    if action == 'train':
        adaboost_train(input_file, model_file)
    else:
        adaboost_accuracy = adaboost_test(input_file, model_file)
        print('Model:', model_file)
        print('Adaboost accuracy is:', adaboost_accuracy * 100, '%')
        
        
'''
-------------- Best Main (Random Forest) -----------------
Command for training:
    train train-data.txt best_model.txt best
Command for testing:
    test test-data.txt best_model.txt best
----------------------------------------------------
'''
if model == 'best':
    portion = 0.05 # the percentage of train data subsampled to train one decision tree.
    num_trees = 400 # number of decision trees in the forest.
    max_depth = 48 # the maximum depth of one decision tree.
    min_size = 1 # the minimum size of one decision tree.
    num_features = 24 # the number of pixels subsampled as the split criteria.
    if action == 'train':
        train_data = processInput(input_file)
        train_forest(train_data, portion, num_trees, max_depth, min_size, num_features)
    else:
        test_data = processInput(input_file)
        forest_accuracy = test_forest(test_data)
        # print('Data portion:', portion, 'number of trees:', num_trees, 'max depth:', max_depth, 'number of features:', num_features)
        print('model:', model_file)
        print('Accuracy:', forest_accuracy * 100, '%')
