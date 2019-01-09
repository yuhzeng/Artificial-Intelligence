### Our best algorithm: forest

## 1. KNN. Highest Accuracy: 72.90%
#### Command lines for KNN
#### Train:
`train train-data.txt nearest_model.txt nearest`
#### Test:
`test test-data.txt nearest_model.txt nearest`

In KNN model, the accuracy is dependent on the number of K, which is the number of nearest neighbors that are used to vote for the estimated label of a test image.
KNN is a lazy classifier thus no training time is needed. In the training process, the train-data.txt was simply copied into nearest_model.txt.

![knn-report](/images/knn-report.png)

| K    | Accuracy (%)
|----- | ------ |
| 5	   | 69.91  |
| 15   | 71.46  |
| 20	 | 71.57  |
| 30	 | 71.13  |
| 47	 | 72.34  |
| **48** | **72.90**|
| 49	 | 72.23  |
| 50	 | 72.23  |
| 100  | 70.91  |

As shown in the above figure and table, when K = 48, KNN model has the highest accuracy of 72.90% on test-data.txt.

Below shows a few sample images that were classified correctly and incorrectly, respectively.

#### Correctly classified:

![image1](/images/10008707066.jpg)
test/10008707066.jpg 0   

![image1](/images/102461489.jpg)
test/102461489.jpg 0

![image1](/images/1074374765.jpg)
test/1074374765.jpg 0

![image1](/images/10931472764.jpg)
test/10931472764.jpg 0

![image1](/images/10313218445.jpg)
test/10313218445.jpg 90

![image1](/images/11490994083.jpg)
test/11490994083.jpg 90

![image1](/images/13439211345.jpg)
test/13439211345.jpg 90

![image1](/images/13416660454.jpg)
test/13416660454.jpg 180

![image1](/images/13962200152.jpg)
test/13962200152.jpg 180

![image1](/images/10161556064.jpg)
test/10161556064.jpg 270

![image1](/images/10353444674.jpg)
test/10353444674.jpg 270

#### Incorrectly classified:

![image1](/images/10484444553.jpg)
test/10484444553.jpg 180 | Estimated label: 0

![image1](/images/11057679623.jpg)
test/11057679623.jpg 0 | Estimated label: 180

![image1](/images/1160014608.jpg)
test/1160014608.jpg 180 | Estimated label: 90

The pattern of the correctly classified images seems to be having a relatively clear boundary around the center of the image. For example, in the third example (test/1074374765.jpg 0), a straight boundary in the center divides the image into two parts with different contrast, and the KNN classifier can classify such images well. However, in the few sample images that were wrongly classified, there seemed to be no obvious boundaries, or there is a boundary, but one of the two divided parts occupy a very small portion in the image.


## 2. Random forest. Highest accuracy: 74.56%
#### Command lines for forest:
#### Train:
`train train-data.txt forest_model.npy forest`
#### Test:
`test test-data.txt forest_model.npy forest`

----------------
#### Description of code:
To train a random decision tree forest, a certain fraction of train data is randomly sampled each time and used to train a decision tree. The decision trees form a forest, and the forest is used to classify test data. In my forest algorithm, train data were randomly subsampled with replacement for each tree. Then, for each tree, since there are 8 * 8 * 3 = 192 pixels per image, the pixels were randomly subsampled without replacement and used as the split criteria. Gini index is calculated for each trial split and the one with minimum gini value will be returned as the best split for that parent node. More description can be found in the comments of each function in the code.

##### Definition of variables

`portion`: the percentage of train data subsampled to train one decision tree.

`num_trees`: number of decision trees in the forest.

`max_depth`: the maximum depth of one decision tree.

`min_size`: the minimum size of one decision tree.

`num_features`: the number of pixels subsampled as the split criteria.

##### Experimental results

We run experiments with respect to 4 of the above parameters:
`portion=train%`, `num_trees`, `max_depth`, `num_features`.

We tune them to see the relation between the parameters, accuracy and training
time.

In order to investigate the effects of each hyperparameter separately,
we only change one parameter in a setting, while keeping all the others
constant.

+ `portion = train%`

| train% | n_trees | max_depth | n_feats | train time | acc on test |
|--------|---------|-----------|---------|------------|-------------|
| 0.02   | 100     | 6         | 6       | 6 min      | 70.575      |
| 0.05   | 100     | 6         | 6       | 39 min     | 70.79646018 |
| 0.1    | 100     | 6         | 6       | 189 min    | 68.80530973 |

The accuracy does not seem to be consistant; training time seems to grow
at O(n^2).

+ `num_trees`

| train% | n_trees | max_depth | n_feats | train time | acc on test |
|--------|---------|-----------|---------|------------|-------------|
| 0.05   | 100     | 6         | 6       | 39 min     | 70.79646018 |
| 0.05   | 200     | 6         | 6       | 66 min     | 70.68584071 |
| 0.05   | 400     | 6         | 6       | 131 min    | 71.01769912 |

Accuracy seems to be going up with more trees, with training time
growing linearly.

+ `max_depth`

| train% | n_trees | max_depth | n_feats | train time | acc on test |
|--------|---------|-----------|---------|------------|-------------|
| 0.05   | 100     | 3         | 6       | 26 min     | 65.48672566 |
| 0.05   | 100     | 6         | 6       | 39 min     | 70.79646018 |
| 0.05   | 100     | 12        | 6       | 37 min     | 71.68141593 |
| 0.05   | 100     | 24        | 6       | 37 min     | 71.90265487 |
| 0.05   | 100     | 48        | 6       | 38 min     | 72.56637168 |

Accuracy seems to be going up consistantly, with constant training time!

+ `num_features`

| train% | n_trees | max_depth | n_feats | train time | acc on test |
|--------|---------|-----------|---------|------------|-------------|
| 0.05   | 100     | 6         | 6       | 39 min     | 70.79646018 |
| 0.05   | 100     | 6         | 12      | 73 min     | 70.46460177 |
| 0.05   | 100     | 6         | 24      | 115 min    | 71.90265487 |

Accuracy is going up, with training time growing linearly.

Note that since random forest is 'random', there is some randomness in
the results. If classifiers with the same setting were run again, we might
see slightly different results, but the general trend should still hold.

Based on the results, we decided to train a model with high `max_depth`,
`n_trees` and `n_feats`, but only a small amount of training data.

| train% | n_trees | max_depth | n_feats | train time | acc on test |
|--------|---------|-----------|---------|------------|-------------|
|**0.05**| **400** | **48**    | **24**  | **550 min**|**74.55752212**|
| 0.05   | 400     | 96         | 24      | 512 min    | 74.11504425 |

The above training was done on an Ubuntu server with `i7-3770@3.40GHz`
cpus.

#### Correctly classified:

![image1](/images/10099910984.jpg)
test/10099910984.jpg 0

![image1](/images/13878624363.jpg)
test/13878624363.jpg 0

![image1](/images/13888636937.jpg)
test/13888636937.jpg 90

![image1](/images/139395528.jpg)
test/139395528.jpg 180

![image1](/images/14138245050.jpg)
test/14138245050.jpg 270

#### Incorrectly classified:

![image1](/images/14357653808.jpg)
test/14357653808.jpg 270 | Estimated label: 90

![image1](/images/14511457203.jpg)
test/14511457203.jpg 90 | Estimated label: 270

![image1](/images/14645715459.jpg)
test/14645715459.jpg 90 | Estimated label: 270

![image1](/images/15016498.jpg)
test/15016498.jpg 0 | Estimated label: 90

From the above sample images shown, we can see that our forest model works well to classify the images with a dividing line around the center. However, in the incorrectly classified sample, our model tend to be confused on the cases of 90 or 270 degrees. The left and right part of the image tend to be misclassified.

## 3. Adaboost. Highest Accuracy: 70.91%
#### Command lines for KNN
#### Train:
`train train-data.txt adaboost_model.txt adaboost`
#### Test:
`test test-data.txt nearest_model.txt adaboost`

### Hypothesis learning

The first step for adaboost is to learn the hypothesis. We use the
hypothesis descibed in the assignment. That is, a decision stump between
the pixel at two positions _i_ and _j_ (here we are ignoring if it's R or G or B).

We loop over all pairs of (i,j) and first set the decision stump to be:

```python
for all training examples x:
    if pixel[i] > pixel[j]:
        x is class_1
    else:
        x is class_2
```

Now we compute the error based on this stump. To compute the error, we first need to find
out which examples are wrongly classified.
When doing so, we should
use matrix operations rather than loops, which will make the program more efficient.
To be more specific, we first compute a 1d array indicating whether `pixel[i] > pixel[j]`
for each training example, resulting in e.g. {1, -1, 1, 1, ...}. Next we build another
1d array indicating the class, where `class_1` gets 1, `class_2` gets -1. Then we just
do element-wise multiplication of the two arrays. In the resulting 1d array,
1 indicates correcly classified example, where -1 indicates incorrectly classifed
example, since 1 \times 1 = (-1) \times (-1) = 1.

Once we obtain the error for this stump, or hypothesis, if the error > 0.5,
then we flip the stump to `pixel[i] < pixel[j]`. Here we are not using all
possible hypotheses; we set an `error_thresh` to make sure that the
 hypothesis is discriminating enough to be included. This `error_thresh`
 is a hyperparamter that needs to be tuned (see next section).

After this we follow the adaboost algorithm to update the weights for the
examples and the weight for the hypothesis.

### One vs. all

Note that since we have 4 classes in this dataset (0, 90, 180, 270),
we need to use "one vs. all" method and build
4 classifiers. Thus in the above code, `class_1` might be `180` degrees, where
`class_2` will be all other 3 classes.

In the end, each classifier gives a score for one instance to be in
that class. The ultimate predication will be the class with the highest
score.

### Hyperparameter `error_thresh`

The table below shows the `error_thresh` parameter and the number of
hypotheses generated for each of the 4 classifiers. We can see that if the threshold
is set very high, then we are including a lot of hypotheses that may not work
very well eventually. If the threshold is too low, we are then eliminating
many useful hypotheses. The best `error_thresh` on this dataset is
somewhere around 0.49, with 70.9% accuracy on the test data.

| `error_thresh` | n_hypo_1 | n_hypo_2 | n_hypo_3 | n_hypo_4 | n_hypo_sum | acc. on test       |
|--------------|----------|----------|----------|----------|------------|-------------|
| 0.498        | 8550     | 8131     | 8664     | 8162     | 33507      | 69.69026549 |
| 0.495        | 2467     | 2438     | 2607     | 2344     | 9856       | 70.13274336 |
| 0.49         | 640      | 669      | 690      | 653      | 2652       | **70.90707965** |
| 0.48         | 177      | 179      | 178      | 177      | 711        | 70.68584071 |
| 0.47         | 81       | 79       | 79       | 80       | 319        | 70.57522124 |
| 0.46         | 42       | 45       | 43       | 47       | 177        | 69.80088496 |
| 0.45         | 31       | 29       | 29       | 31       | 120        | 68.5840708  |
| 0.4          | 7        | 5        | 5        | 7        | 24         | 66.48230088 |

#### Correctly classified:

![image1](/images/1074374765.jpg)
test/1074374765.jpg 0

![image1](/images/11057679623.jpg)
test/11057679623.jpg 0

![image1](/images/12063320613.jpg)
test/12063320613.jpg 90

![image1](/images/12735208893.jpg)
test/12735208893.jpg 180

![image1](/images/12605538693.jpg)
test/12605538693.jpg 270

#### Incorrectly classified:

![image1](/images/12799838924.jpg)
test/12799838924.jpg 0 | Estimated label: 270

![image1](/images/133910954.jpg)
test/133910954.jpg 0 | Estimated label: 180

![image1](/images/13395784165.jpg)
test/13395784165.jpg 90 | Estimated label: 180

![image1](/images/139395528.jpg)
test/139395528.jpg 180 | Estimated label: 270

![image1](/images/14016488749.jpg)
test/14016488749.jpg 90 | Estimated label: 0

The misclassified images tend to have no clear dividing line in the image, and the contrast between different objects are not very strong.


## 4. Best (forest). Highest Accuracy: 74.56%
#### Command lines for best
#### Train:
`train train-data.txt best_model.txt best`
#### Test:
`test test-data.txt best_model.txt best`

We used forest algorithm as our best. Below are the parameters used for training the best forest model.

| train% | n_trees | max_depth | n_feats | train time | acc on test |
|--------|---------|-----------|---------|------------|-------------|
|  0.05  |   400   |   48      |  24     |   550 min  |  74.55752212|
