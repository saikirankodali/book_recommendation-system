from collections import namedtuple
import heapq
# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """

    pass


class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r_ui', 'est', 'details'])):
    """A named tuple for storing the results of a prediction.

    It's wrapped in a class, but only for documentation and printing purposes.

    Args:
        uid: The (raw) user id. See :ref:`this note<raw_inner_note>`.
        iid: The (raw) item id. See :ref:`this note<raw_inner_note>`.
        r_ui(float): The true rating :math:`r_{ui}`.
        est(float): The estimated rating :math:`\\hat{r}_{ui}`.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
    """

    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)

        return s


class ContentBased_Algo:
    def __init__(self, trainset, books, k=40, min_k=1):
        self.k = k
        self.min_k = min_k
        self.trainset = trainset
        self.books = books

    def estimate(self, u, i, ratings, skip):

        if not (self.trainset.knows_user(u) ):
            raise PredictionImpossible('User is unkown.')
        if skip:
            raise PredictionImpossible('User rated books not in book table.')


        neighbors = []
        for j, isbn in enumerate(ratings.keys()):
            item = self.trainset.to_inner_iid(isbn)
            neighbors.append((item, self.sim[-1, j], ratings[isbn]))

        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (item, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details

    def predict(self, uid, iid, ratings, skip, r_ui=None, clip=True, verbose=False):
        """Compute the rating prediction for given user and item.

                The ``predict`` method converts raw ids to inner ids and then calls the
                ``estimate`` method which is defined in every derived class. If the
                prediction is impossible (e.g. because the user and/or the item is
                unkown), the prediction is set according to :meth:`default_prediction()
                <surprise.prediction_algorithms.algo_base.AlgoBase.default_prediction>`.

                Returns:
                    A :obj:`Prediction\
                    <surprise.prediction_algorithms.predictions.Prediction>` object
                    containing:

                    - The (raw) user id ``uid``.
                    - The (raw) item id ``iid``.
                    - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
                    - The estimated rating (:math:`\\hat{r}_{ui}`).
                    - Some additional details about the prediction that might be useful
                      for later analysis.
                """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)

        details = {}

        try:
            est = self.estimate(iuid, iid, ratings, skip)
            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def get_item_attribute_similarity(self, uid, iid):

        # get user rated items
        book_ratings = {}
        skip = True
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
            return book_ratings, skip

        for item_rating in self.trainset.ur[self.trainset.to_inner_uid(uid)]:
            (item, r) = item_rating
            raw_iid = self.trainset.to_raw_iid(item)
            if raw_iid in self.books['ISBN'].unique():
                book_ratings[raw_iid] = r

        if book_ratings:
            compared_isbnlist = [isbn for isbn in book_ratings.keys()]
            compared_isbnlist.append(iid)

            compared_books = self.books.loc[self.books['ISBN'].isin(compared_isbnlist)]
            # fill rows with null values in bookAuthor and publisher columns
            if compared_books['bookAuthor'].isnull().sum() > 0:
                compared_books['bookAuthor'] = compared_books['bookAuthor'].fillna('xxxx')

            if compared_books['publisher'].isnull().sum() > 0:
                compared_books['publisher'] = compared_books['publisher'].fillna('xxx')

            book_title = compared_books['bookTitle']
            authors = compared_books['bookAuthor']
            publisher = compared_books['publisher']

            # get selected books tf-idf features and compare the similarity of features
            vectorizer = TfidfVectorizer(analyzer='word')
            # build book-title tfidf matrix
            tfidf_matrix = vectorizer.fit_transform(book_title)
            print(tfidf_matrix.shape)
            # creating cosine similarity matrix using linear_kernal of sklearn
            title_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

            # build book-author tfidf matrix
            tfidf_matrix = vectorizer.fit_transform(authors)
            print(tfidf_matrix.shape)
            # creating cosine similarity matrix using linear_kernal of sklearn
            author_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

            # build publisher tfidf matrix
            tfidf_matrix = vectorizer.fit_transform(publisher)
            print(tfidf_matrix.shape)
            # creating cosine similarity matrix using linear_kernal of sklearn
            publisher_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

            self.sim = (title_similarity+author_similarity+publisher_similarity)/3
            skip = False

        return book_ratings, skip

    def test(self, testset, verbose=False):
        predictions = []
        for (uid, iid, r_ui_trans) in testset:
            ratings, skip = self.get_item_attribute_similarity(uid, iid)
            pre = self.predict(uid, iid, ratings, skip, r_ui_trans, verbose=verbose)
            predictions.append(pre)

        return predictions