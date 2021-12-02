from processing import process_data
from bayesian_classifier import BayesianClassifier

if __name__ == "__main__":
    train_X, train_y = process_data("data/train.csv", "data/stop_words.txt")
    test_X, test_y = process_data("data/test.csv", "data/stop_words.txt")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    # print prediction for the first test
    print(classifier.predict(test_X[0]))

    print("model score: ", classifier.score(test_X, test_y))
