# Naive Bayes Classifier from Scratch

A Naive Bayes classifier from scartch trained on Mushroom dataset from UCI Machine Learning Repository.

## Dataset
This classifier is designed for datasets with categorical features and a discrete target variable. Each feature should have a limited set of unique values, allowing the model to compute conditional probabilities. This implementation is best suited for datasets such as the UCI Mushroom dataset or any dataset where the target variable is categorical and there are clear feature categories.

### Data Preparation

Before using the classifier, ensure that the data is cleaned, all features are categorical, and each sample is labeled correctly. No scaling or normalization is needed, as Naive Bayes relies on categorical feature counts rather than distances.

## Model Specification

The Naive Bayes classifier works by applying Bayes' Theorem:

$$
P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}
$$

Where:

- $P(y \mid X)$ is the posterior probability of class $ y $ given the input features $ X $.
- $P(X \mid y)$ is the likelihood of the features given class $ y $.
- $P(y)$ is the prior probability of class $ y $.
- $P(X)$ is the probability of the features $ X $, which we omit because it is constant across classes.

We approximate the probability of a sample belonging to a particular class using:

$$
\hat{y} = \arg\max_{y \in Y} \left( \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y) \right)
$$

This means we compute the probability scores for each class and select the class with the highest score.

## Training

### Calculating Class Priors
The class prior probabilities, $P(y)$, represent the relative frequency of each class in the dataset:

$$
P(y) = \frac{\text{Number of samples in class } y}{\text{Total number of samples}}
$$

```python
for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
```


### Calculating Likelihoods
For each feature $ i $, the likelihood $ P(x_i \mid y) $ is the probability of feature $ x_i $ taking a specific value, given that the sample belongs to class $ y $. We use Laplace smoothing to handle cases where a feature value might not be present in the class:

$$
P(x_i = v \mid y) = \frac{\text{Count of } x_i = v \text{ in class } y + \alpha}{\text{Number of samples in class } y + \alpha \cdot \text{Number of unique values of } x_i}
$$

Where:
- $ \alpha $ is the smoothing parameter. If $ \alpha = 1 $, it's called Laplace smoothing. If $ \alpha = 0 $, no smoothing is applied.


```python
for c in self.classes:
            X_with_c = X[y == c]
            num_of_X_with_c = X_with_c.shape[0]
            for f in range(n_features):
                feature_values = np.unique(X[:, f])
                for val in feature_values:
                    count = np.sum(X_with_c[:, f] == val)
                    self.likelihoods[c][f][val] = (count + self.smoothing) / (num_of_X_with_c + len(feature_values) * self.smoothing)
```

### Predicting

To predict the class of a new sample, we calculate the score for each class and select the class with the highest score. The score is the sum of the log prior probability and the log likelihood of each feature value in the sample:

```python
    def predict_sample(self, x):
        best_class, best_score = None, -np.inf

        for c in self.classes:
            score = np.log(self.class_priors[c])  # Log priors
            for f_index, val in enumerate(x):  # For each feature value in the sample
                score += np.log(self.likelihoods[c][f_index].get(val, self.smoothing / (len(self.likelihoods[c][f_index]) + self.smoothing)))

            if score > best_score:
                best_class, best_score = c, score

        return best_class
```

## Evaluation

Lastly, we use a simple accuracy metric to evaluate the model's performance. In this demo, we end up getting a near perfect accuracy as the dataset is perfect for naive bayes.

