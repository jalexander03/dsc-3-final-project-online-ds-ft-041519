
# **Module 3 Project - Mushroom Classification**

* Student name: **Alex Husted**
* Student pace: **Online Full Time - April Cohort**
* Scheduled project review date/time: **Friday July 26, 2019**
* Instructor name: **Rafael Carrasco**
* Blog post URL: **Work in Progress**

# **Project Outline**

This notebook represents the final project in Module 3 of Flatiron's Data Science Bootcamp. The Module began by introducing time series visualizations and trend identification. Then a deeper understanding of K-Nearest-Neighbors was provided along with model evaluation metrics. Following up, Module 3 initiated the disciplines of graph theory, clustering networks, and recommendation systems. Around the midpoint of the module, the curriculum dives into logistic regression, ROC curves, and class imbalance problems. Decision trees, random forests, and ensemble methods then gave a wholesome look into the powers of machine learning. Finally the lessons of SVM, clustering, and dimensionality reduction allowed for a satisfying conclusion to the module. 

In this project, I will be working with Mushroom Classification. This dataset includes descriptions of samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. The data has been drawn from *The Audubon Society Field Guide to North American Mushrooms*. Each species can be classified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. Although this dataset was originally contributed to the UCI Machine Learning repository nearly 30 years ago, mushroom hunting (otherwise known as "shrooming") is enjoying new peaks in popularity. Learning which features could spell certain death and which are most palatable is the goal for this dataset of mushroom characteristics.

## Data Information

The data set contains 8124 rows and the following 23 features:

`class`: edible(e) or poisonous(p)

`cap-shape`: bell(b), conical(c), convex(x), flat(f), knobbed(k), sunken(s)

`cap-surface`: fibrous(f), grooves(g), scaly(y), smooth(s)

`cap-color`: brown(n), buff(b), cinnamon(c), gray(g), green(r), pink(p), purple(u), red(e), white(w), yellow(y)

`bruises`: bruises(t), no bruises(f)

`odor`: almond(a), anise(l), creosote(c), fishy(y), foul(f), musty(m), none(n), pungent(p), spicy(s)

`gill-attachment`: attached(a), descending(d), free(f), notched(n)

`gill-spacing`: close(c), crowded(w), distant(d)

`gill-size`: broad(b), narrow(n)

`gill-color`: black(k), brown(n), buff(b), chocolate(h), gray(g), green(r), orange(o), pink(p), purple(u), red(e), white(w), yellow(y)

`stalk-shape`: enlarging(e), tapering(t)

`stalk-root`: bulbous(b), club(c), cup(u), equal(e), rhizomorphs(z), rooted(r), missing(?)

`stalk-surface-above-ring`: fibrous(f), scaly(y), silky(k), smooth(s)

`stalk-surface-below-ring`: fibrous(f), scaly(y), silky(k), smooth(s)

`stalk-color-above-ring`: brown(n), buff(b), cinnamon(c), gray(g), orange(o), pink(p), red(e), white(w), yellow(y)

`stalk-color-below-ring`: brown(n), buff(b), cinnamon(c), gray(g), orange(o), pink(p), red(e), white(w), yellow(y)

`veil-type`: partial(p), universal(u)

`veil-color`: brown(n), orange(o), white(w), yellow(y)

`ring-number`: none(n), one(o), two(t)

`ring-type`: cobwebby(c), evanescent(e), flaring(f), large(l), none(n), pendant(p), sheathing(s), zone(z)

`spore-print-color`: black(k), brown(n), buff(b), tan(h), green(r), orange(o), purple(u), white(w), yellow(y)

`population`: abundant(a), clustered(c), numerous(n), scattered(s), several(v), solitary(y)

`habitat`: grasses(g), leaves(l), meadows(m), paths(p), urban(u), waste(w), woods(d)

# **Project PreWork** 

Before moving forward with classification, there are necessary steps to become familiar with the Mushroom Dataset. First, importing libraries needed to complete an exploratory analysis of the data would be helpful. Then it's important to examine the features within the dataset. What are the qualities of each species? What characteristics make them similar or different? What characteristics are most important for classification? Questions like these will help develop a better understanding of the dataset and will eventually guide effective classification.


# **Exploring Mushrooms**

Exploratory Data Analysis, or EDA, is an integral part of understanding the Mushroom dataset. Before moving towards classification, it's vital to become familiar with different realtionships within the data. Analyzing these relationships will provide intuition about how to interpret the results of the proceeding models. Asking questions about these relationships beforehand might also supply additional knowledge about relationships that we might have not known existed. This section will further investigate the distribution of data and ask specific questions about the information lying inside the data set.

**Cap Shape**: There are certainly not very many characteristics of cap shape that could definitively decide if the mushroom is poisonous or edible. If the mushroom is knobbed, more often than not it could be poisonous. A bell shaped cap is more likely to be edible. 

**Cap Surface**: If the mushroom has a more fibrous surface, it’s more likely to be edible. Smooth and scaly mushrooms are slightly more likely to be poisonous. 

**Cap Color**: If the mushroom has a more white or gray surface, it’s likely to be edible.  Red or yellow cap colors tend to be more poisonous. 

**Bruises**: If the mushroom has bruises, it’s likely to be edible. If the mushroom does not have bruises, it’s more likely to be poisonous. 

**Odor**: If the mushroom has no odor, it’s extremely likely to be edible. Any odor that is foul or pungent is highly likely to be poisonous. 

**Gill Attachments**: There are no characteristics that could definitively classify the mushroom as edible or poisonous. 

**Gill Spacing**: If the gills are closely spaced, it’s like to be poisonous.  If they are crowded, then it’s more likely to be edible. 

**Gill Size**: If the gills are narrow, it’s like to be poisonous.  If they are broad, then it’s more likely to be edible. 

**Gill Color**: Colors of buff are almost certain to be poisonous. Gray and chocolate colors are also likely to be poisonous. White, purple, and brown colors are likely to be edible. 

**Stalk Shape**: Stalks are difficult to analyze for edibility. 

**Stalk Root**: Missing roots are very likely to be poisonous. Club and equal roots are likely to be edible. 

**Stalk Surface Above Ring**: Smooth stalks above the ring will likely be edible. Silky stalks are likely to be poisonous. 

**Stalk Surface Below Ring**: Smooth stalks above the ring will likely be edible. Silky stalks are likely to be poisonous. 

**Stalk Color Above Ring**: White and gray colors are likely to be edible.  Buff, brown, and purple are likely to be poisonous. 

**Stalk Color Below Ring**: White and gray colors are likely to be edible.  Buff, brown, and purple are likely to be poisonous. 

**Veil Color**: Veils are difficult to analyze for edibility. 

**Ring Number**: Ring Numbers are difficult to analyze for edibility. 

**Ring Type**: Pendant ring types are very likely to be edible. Evanescent and large ring types are likely to be poisonous. 

**Spore Print Color**: Black and brown are highly likely to be edible. Tan and white colors are highly likely to be poisonous. 

**Population**: Several mushrooms found in the population are likely to be poisonous. Numerous and abundant are likely to be edible. 

**Habitat**: Mushrooms found in grasses and woody areas are likely to be edible. Mushrooms found on paths and leaves are likely to be poisonous. 

# **Model Preparation**

The mushroom dataset is primarily categorical data. These data types are not ideal for model building, therefore, the data must be converted into numerical data types. This can be done using Label and One-Hot encoding. 

## Label Encoding

Most Machine Learning algorithms require numerical features. However, the dataset is composed of categorical features. We now must proceed to convert these to numerical data types. 

A typical approach is to perform _Label Encoding_. This is nothing more than just assigning a number to each category, that is:

(cat_a, cat_b, cat_c, etc.) → (0, 1, 2, etc.)

This technique works:

* When the features are binary (only have 2 unique values).
* When the features are _ordinal categorical_ (that is, when the categories can be ranked). A good example would be a feature called _t-shirt size_ with 3 unique values _small_, _medium_ or _large_, which have an intrinsic order.

**However**, in this case, only some of our features have 2 unique values (most of them have more), and none of them are _ordinal categorical_ (in fact they they are _nominal categorical_, which means they have no intrinsic order).

Therefore, we will only apply Label Encoding to those features with a binary set of values:

Label encoding has converted some of the features to values of 0 or 1. More importantly, our labels (the class column) are now 0=e, and 1=p. Other features are encoded into a numerical value based upon their categorical values. (See Below).

## Standardize Features

It is generally considered a good practice to standardize our features (convert them to have zero-mean and unit variance). Most of the times, the difference will be small, but in any case, it still never hurts to do so.

## Training and Test Data

We will separate our data into a training set (70%) and a test set (30%). This is a very standard approach in Machine Learning.

The stratify option ensures that the ratio of edible to poisonous mushrooms in our dataset remains the same in both training and test sets. The random_state parameter is simply a seed for the algorithm to use (if we didn't specify one, it would create different training and test sets every time we run it)

# **Logistic Regression**

Logistic Regression is a classification algorithm used where the response variable is categorical. The idea of Logistic Regression is to find a relationship between features and probability of particular outcome. This type of problem is referred to as Binomial Logistic Regression, where the response variable has two values 0 and 1 or pass and fail or true and false. 

Since this is a supervised learning binary classification problem, it makes sense to start running logistic regression.The models used here simply predicts the probability of an instance (row) belonging to the default class, which can then be snapped into a 0 or 1 classification. For this case, a 0 classification will signifies the mushroom is edible and a 1 classification signifies the mushroom is poisonous, thus inedible. 

The following three models are completed using Logistic regression, however they differ slightly. 

* The LR Normal model is a default logistic regression model using sklearn.
* The LR Tuned model imposes a penalty on the coefficients to prevent overfitting.
* The LR Lasso model performs variable selection by shrinking some coefficients. 

With the proceeding models you will see the following: AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between mushroom edibility.

## LR Lasso

L1 and L2 are regularization parameters and are used to avoid overfiting. Both L1 and L2 regularization prevents overfitting by shrinking (imposing a penalty) on the coefficients.

L2 (Ridge) shrinks the coefficients by the same proportions but eliminates none, while L1 (Lasso) can shrink some coefficients to zero, performing variable selection. If all the features are correlated with the label, ridge outperforms lasso, as the coefficients are never zero in ridge. If only a subset of features are correlated with the label, lasso outperforms ridge as in lasso model some coefficient can be shrunken to zero.

Grid search with GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the tuned_parameter. The GridSearchCV instance implements the usual estimator when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.

# **Support Vector Machine**

A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimensional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

**Parameters**:

* The *Regularization* parameter (often termed as C parameter in python’s sklearn library) tells the SVM optimization how much you want to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane mis-classifies more points.

* The learning of the hyperplane in linear SVM is done by transforming the problem using some linear algebra. This is where the *Kernel* plays role. Polynomial kernels calculates separation line in higher dimension. This is called kernel trick. 

* The *Gamma* parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. In other words, with low gamma, points far away from plausible seperation line are considered in calculation for the seperation line. Where as high gamma means the points close to plausible line are considered in calculation.

* *Margin* is the last but very importrant characteristic of SVM classifier. SVM to core tries to achieve a good margin. A margin is a separation of line to the closest class points. A good margin is one where this separation is larger for both the classes.

**RandomizedSearchCV** implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used. The parameters of the estimator used to apply these methods are optimized by cross-validated search over parameter settings.In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter. If all parameters are presented as a list, sampling without replacement is performed. If at least one parameter is given as a distribution, sampling with replacement is used.

# **Decision Trees**

In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. As the name goes, it uses a tree-like model of decisions. A decision tree is a tree where each node represents a feature(attribute), each link(branch) represents a decision(rule) and each leaf represents an outcome(categorical or continues value). The idea is to create a tree for the entire data and process a single outcome at every leaf, minimizing error.

There are couple of algorithms used to build decision trees. In this mushroom classification, we will use two different criterion for tuning parameters. These models use Gini indexing and Entropy:

* CART (Classification and Regression Trees) → uses Gini Index (Classification) as metric.
* ID3 (Iterative Dichotomiser 3) → uses Entropy function and information gain as metrics.


# **Random Forest**

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our models' prediction. The fundamental concept behind random forest is a simple but powerful one — the reason that the random forest model works so well is that s large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models. The trees protect each other from their individual errors (as long as they don’t constantly all err in the same direction). While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction.


# **Conclusion**

Though field experts determine that is that there is no simple set of rules to identify whether a mushroom is edible or not, it seems like with these algorithms we can get pretty close based upon numerical characteristics. The models above identified specific traits that seem to heavily influence the chance that a mushroom could be edible. Specifically, when tuning the logistic regression lasso model, there are a few coefficients identified that play an important role in the classification process. 

* **Gill Size**: If the gills are narrow, the mushroom is likly to be poisonous.  If they are broad, then it’s more likely to be edible. 

* **Ring Type**: Pendant ring types are very likely to be edible. Evanescent and large ring types are likely to be poisonous. 

* **Cap Surface**: If the mushroom has a more fibrous surface, it’s more likely to be edible. Smooth or scaly mushrooms are slightly more likely to be poisonous. 

* **Bruises**: If the mushroom has bruises, it’s likely to be edible. If the mushroom does not have bruises, it’s more likely to be poisonous. 

* **Spore Print Color**: Black and brown are highly likely to be edible. Tan and white colors are highly likely to be poisonous. 

**Best Models**: 

LR Lasso - performs variable selection by shrinking some coefficients.
    * Parameters: C=100, penalty='l1'
    * Weighted Avg Score: 0.94
    * 87 False Negatives
    * 10% False Positive rate for 96% accuracy
    
Tuned Random Forest - individual decision trees that operate as an ensemble.
    * Parameters: 'n_estimators': 20, 'min_samples_leaf': 10, 'max_features': 'auto'
    * Weighted Avg Score: 0.99
    * 87 False Negatives
    * 2% False Positive rate for 99% accuracy
