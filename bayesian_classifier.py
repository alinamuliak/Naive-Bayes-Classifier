class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """

    def __init__(self):
        self.num_of_unique_words = 0
        self.unique_words = set()

        self.num_of_citations = {"Edgar Alan Poe": 0, "Mary Wollstonecraft Shelley": 0, "HP Lovecraft": 0}
        self.count_num_of_each_word = {"Edgar Alan Poe": {}, "Mary Wollstonecraft Shelley": {}, "HP Lovecraft": {}}
        self.count_all_words = {"Edgar Alan Poe": 0, "Mary Wollstonecraft Shelley": 0, "HP Lovecraft": 0}

    def fit(self, X: list, y: list) -> None:
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param X: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """

        for message in X:
            self.unique_words = self.unique_words.union(message.split())
        self.num_of_unique_words = len(self.unique_words)

        for idx, citation in enumerate(X):
            for word in citation.split():
                if word in self.count_num_of_each_word[y[idx]]:
                    self.count_num_of_each_word[y[idx]][word] += 1
                else:
                    self.count_num_of_each_word[y[idx]][word] = 1

        for key in self.count_all_words:
            self.count_all_words[key] = sum(self.count_num_of_each_word[key].values())

        for author in self.num_of_citations:
            self.num_of_citations[author] = y.count(author)

    def predict_prob(self, message: str, label: str) -> float:
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        message = message.split()

        # calculate the a priori probability of this author
        total_n_citations = sum(self.num_of_citations.values())
        p_this_author = self.num_of_citations[label] / total_n_citations

        # calculation P(word | author) for each word in message in that way,
        # so that any of probabilities won't be 0.
        probability = 1
        for word in message:
            if word in self.count_num_of_each_word[label]:
                num_of_this_word = self.count_num_of_each_word[label][word]
            else:
                num_of_this_word = 0
            num_of_authors_word_total = self.count_all_words[label]
            denominator = num_of_authors_word_total + len(set(self.unique_words).union(set(message)))
            probability *= (num_of_this_word + 1) / denominator

        return probability * p_this_author

    def predict(self, message: str) -> str:
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        authors_probabilities = {"Edgar Alan Poe": 0, "Mary Wollstonecraft Shelley": 0, "HP Lovecraft": 0}
        for author in authors_probabilities:
            probability = self.predict_prob(message, author)
            authors_probabilities[author] += probability

        # return author with highest probability
        max_prob = max(authors_probabilities.values())
        return list(authors_probabilities.keys())[list(authors_probabilities.values()).index(max_prob)]

    def score(self, X: list, y: list) -> float:
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        :return: float
        """
        num_true = 0
        for phrase in range(len(X)):
            if self.predict(X[phrase]) == y[phrase]:
                num_true += 1

        result = (num_true / len(X)) * 100
        return round(result, 2)
