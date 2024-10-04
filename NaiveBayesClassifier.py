import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, smoothing=0.0):
        self.smoothing = smoothing 
        self.class_priors = {}
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # {class: {feature_index: {feature_value: likelihood}}}
         

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Bayes Theorem: P(y | x) = P(x | y) * P(y) / P(x)
        # In this case, we don't care about P(x) because it's the same for all classes.
        
        # GOAL: Calculate P(y) and P(x_i | y) for all y and x_i
        
        # Calculate prior probabilities, which are the frequency of each class | P(y)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples

        # Calculate feature probabilities, which are the frequency of each feature value given the class | P(x_i | y)
        for c in self.classes:
            X_with_c = X[y == c]
            num_of_X_with_c = X_with_c.shape[0] 

            for f in range(n_features):
                feature_values = np.unique(X[:, f])
                for val in feature_values:
                    count = np.sum(X_with_c[:, f] == val)
                    self.likelihoods[c][f][val] = (count + self.smoothing) / (num_of_X_with_c + len(feature_values) * self.smoothing)

    def predict(self, X):
        return np.array([self.predict_feature(x) for x in X])

    def predict_feature(self, x):
        best_class, best_score = None, -np.inf

        for c in self.classes:
            score = np.log(self.class_priors[c])  # we use the log priors to start the score
            for f_index, val in enumerate(x): # for each feature value in the sample
                score += np.log(self.likelihoods[c][f_index].get(val, self.smoothing / (len(self.likelihoods[c][f_index]) + self.smoothing))) 
            
            if score > best_score:
                best_class, best_score = c, score

        return best_class

    def score(self, X, y):
        return np.mean(self.predict(X) == y)