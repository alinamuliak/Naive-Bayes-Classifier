# Naive Bayes Classifier

### Work-breakdown structure
[Karyna Volokhatiuk](https://github.com/karyna-volokhatiuk) - `process_data`, `fit`;</br>
[Alina Muliak](https://github.com/alinamuliak) - `predict_prob`, `predict`;</br>
[Olena Karaim](https://github.com/karakumm) - `score`, conclusions.

## Introduction
One of Bayes theorem applications is Naive Bayes classifier,
which is a probabilistic classifier whose aim is to determine which class some observation probably belongs by using the Bayes formula:

`P(class∣observation)=P(observation∣class)P(class)/P(observation)`

Under the strong independence assumption, one can calculate  `P(observation∣class)`  as
`P(observation)=∏_(i=1,..,n)P(feature_i)`, 
where  `n`  is the total number of features describing a given observation. Thus,  P(class|observation)  now can be calculated as

```P(class∣observation)=P(class)×∏_(i=1,..,n)P(feature_i∣class)/P(feature_i)```

## Data description
Data sets consists of citations of three famous writers: Edgar Alan Poe, Mary Wollstonecraft Shelley and HP Lovecraft. The task with this data set is to classify a piece of text with the author who was more likely to write it.

## Implementation
- data pre-processing
- implementation of BayesianClassifier
- testing

### A few sentences about implementation of classifier
In our classifier the following methods are realized: `fit`, `predict_prob`, `predict` and `score`.
In method fit using our databases we fill in all the necessary attributes for calculating
our probabilities, such as
- `num_of_citations` - the dictionary, where keys are authors and values are the total number
of citations for each author; 
- `count_num_of_each_word` - a dictionary, where again authors are keys and values are dictionaries with the numbers(value)of using each word(key) for each author;
- `count_all_words` - a dictionary which represents the total number of words each author uses.
</br></br>In `predict_prob` method we calculate the probability of a given author to be a true author for a given message.
This method returns the numerator for Bayes formula. In predict method we compare the probabilities for each author to
be an author of a given message and return the name of author with the highest calculated probability.
</br>And finally, method `score` returns the accuracy of our calculations on the given data and labels, represented as a percentage.

## Conclusions
In order to make predictions about the author, whose quotes are cited in out
given message, we use Naive Bayes classifier.
Firstly we calculate the probability that given author has written any quote from
database, using formula `P(author)`, where we divide the number of citations of this author by total number
of citations. To make our program more efficient we use dictionaries to save the number of times each author
has used each word. Then we calculate the conditional probability that the quote was written by a given author,
so that the probability cannot be equal to zero. So, as in Bayes formula,
in numerator we multiply `P(word|author)` by `P(author)`.
We ignore denominator for Bayes formula since for all authors they will be the same,
and it would be enough to compare only their numerators.
After this we create dictionary, where our authors are the keys and then in loop calculate
the probability for each author to have written the message and find the maximum one,
and it would be our wanted author.

### Pros and cons of the method
We find the implemented method quite efficient, moreover we improved our efficiency using dictionaries to save the number
of times each author has used each word. Firstly we tried the bag-of-words method with 0 and 1, representing them as vectors,
but unfortunately we refused to use this method because of long-time search, and also it needs a lot of memory,
while time complexity of search using dictionaries is O(1). But we also defined one drawback:
in our method similar words with common roots, or the same words but with different endings,
or plural and singular form of one word are considered to be different.
Unfortunately such improvement of our program needs more knowledge of artificial intelligence.

## Results
We have checked the efficiency of a trained model on the given test data and labels.
In our case the accuracy of the method is 83.3%, which is high enough.
So we have made a conclusion that the implementation of method of calculating probability using Naive Bayes classifier
is suitable and efficient, which frankly speaking exceeded our expectations.
