import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # Turn off warnings


# Import the dataset

# The dataset is comprised of three separate tables: BX-Users, BX-Books, and BX-Book-Ratings.
# Each is separated by semi-colons, and the first row contains the column headers.

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


#dropping last three columns containing image URLs which will not be required for analysis and Exploring books dataset and
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True)

# yearOfPublication should be set as having dtype as int
# checking the unique values of yearOfPublication
books.yearOfPublication.unique()

#as it can be seen from above that there are some incorrect entries in this field. It looks like Publisher names
#'DK Publishing Inc' and 'Gallimard' have been incorrectly loaded as yearOfPublication in dataset due to some errors in csv file
#Also some of the entries are strings and same years have been entered as numbers in some places
#making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)
#investigating the rows having 'DK Publishing Inc' as yearOfPublication
books.loc[books.yearOfPublication == 'DK Publishing Inc',:]

#it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
#rechecking
books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:]

#%%

#investigating the rows having 'Gallimard' as yearOfPublication
books.loc[books.yearOfPublication == 'Gallimard',:]

#%%

#making required corrections as above, keeping other fields intact
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"
#rechecking
books.loc[books.ISBN == '2070426769',:]

#%%

#Correcting the dtypes of yearOfPublication
books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')
print(sorted(books['yearOfPublication'].unique()))


import surprise
from surprise import Reader, Dataset, KNNWithMeans, KNNBasic, SVD, model_selection, accuracy, trainset
from surprise.model_selection import train_test_split
# First need to create a 'Reader' object to set the scale/limit of the ratings field
reader = Reader(rating_scale=(1, 10))

# # Load the data into a 'Dataset' object directly from the pandas df.
train_data = pd.read_csv('rating_trainingset.csv', encoding='latin-1')
test_data = pd.read_csv('rating_testset.csv', encoding='latin-1')
train_data.columns = ['user_id', 'rating', 'isbn']
test_data.columns = ['user_id', 'rating', 'isbn']
# switch 'isbn' with 'rating'
cols = list(train_data)
cols.insert(2,cols.pop(cols.index('rating')))
train_data = train_data.loc[:, cols]
cols = list(test_data)
cols.insert(2,cols.pop(cols.index('rating')))
test_data = test_data.loc[:, cols]

trainset = Dataset.load_from_df(train_data, reader)
trainset, raw2inner_id_users, raw2inner_id_items = trainset.construct_trainset(trainset.raw_ratings)
testset = Dataset.load_from_df(test_data, reader)
testset = testset.construct_testset(testset.raw_ratings)

print('%d users, %d items, %d ratings in training set\n'%(trainset.n_users, trainset.n_items, trainset.n_ratings))


from collections import defaultdict
from six import iteritems
from interval import Interval
from Content_based_Recommendation import ContentBased_Algo


unique_train_isbn = books['ISBN'].isin(train_data['isbn'].unique())
train_books = books.loc[unique_train_isbn]

#**********************************************************************************************************************
# hybrid-based algorithm
#**********************************************************************************************************************
# get training data for regression models from collaborative filtering and content-based algorithms
def load_user_CF_model(trainset, batch_size=15000, k=40):
       print('loading user-based collaboraive filtering model\n')
       batch_start = 0
       while batch_start < trainset.n_users:
              if (batch_start + batch_size) < trainset.n_users:
                     batch_end = batch_start + batch_size - 1
              else:
                     batch_end = trainset.n_users - 1
                     batch_size = batch_end - batch_start + 1

              batch_interval = Interval(batch_start, batch_end)
              ur_batch = defaultdict(list)
              for i in range(batch_start, batch_end + 1):
                     ur_batch[i - batch_start] = trainset.ur[i]

              # remove entries in trainset.ir whose users index is beyond user_batch_start and user_batch_end
              ir_batch = defaultdict(list)
              for y, y_ratings in iteritems(trainset.ir):
                     for xi, ri in y_ratings:
                            if xi in batch_interval:
                                   ir_batch[y].append((xi - batch_start, ri))

              trainset_batch = surprise.trainset.Trainset(ur_batch,
                                                          ir_batch,
                                                          batch_size,
                                                          trainset.n_items,
                                                          trainset.n_ratings,
                                                          (1, 10),
                                                          raw2inner_id_users, raw2inner_id_items)

              # user-based CF algorithm
              algo = KNNWithMeans(trainset.n_users, k, sim_options={'name': 'cosine', 'user_based': True},
                                  user_based=True)
              print('fitting batches between %d-%d, total:%d\n' % (batch_start, batch_end, trainset.n_users))
              algo.fit(trainset_batch, batch_start)

              batch_start = batch_end + 1
       return algo

def load_content_model(books, train_data, trainset, k=10):
       print('loading content-based model\n')
       unique_train_isbn = books['ISBN'].isin(train_data['isbn'].unique())
       train_books = books.loc[unique_train_isbn]
       algo = ContentBased_Algo(trainset, books, k)
       return algo

def get_train_test_data(trainset, testset, content_model, collaborative_model):
       train_set = trainset.build_testset()
       print('getting training data...\n')
       content_train_pred = content_model.test(train_set, verbose=True)
       collaborative_train_pred = collaborative_model.test(train_set)
       '''
       You need to figure out how to obtain training data from content_train_pred and collaborative_train_pred as training
       data, i.e. hybrid_train_x, hybrid_train_y, for hybrid recommendation algorithm here
       
       '''

       print('getting test data...\n')
       content_test_pred = content_model.test(testset, verbose=True)
       collaborative_test_pred = collaborative_model.test(testset)

       '''
       You need to figure out how to obtain test data from content_test_pred and collaborative_test_pred as test
       data, i.e. hybrid_test_x, hybrid_test_y, for hybrid recommendation algorithm here
       
       '''

       return hybrid_train_x, hybrid_train_y, hybrid_test_x, hybrid_test_y

def training_hybrid_model(model, train_x, train_y):
    '''
    You need to figure out how to train hybrid recommendation algorithm here

    '''
    return

def predict(model, test_x, test_y):
       '''
     You need to figure out how to train hybrid recommendation algorithm here

       '''
       return result

def rmse(predictions, true_r, verbose=True):
    '''
    Compute RMSE (Root Mean Squared Error).
    '''


    if predictions is None:
        print('Prediction list is empty.\n')
        exit()

    mse =[]
    for i, est in enumerate(predictions):
        mse.append((true_r[i] - est)**2)
    mse = np.mean(mse)
    rmse = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse))

    return rmse


#======================================================================================================================
models = []
model_name = []
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
models.append(model_DecisionTreeRegressor)
model_name.append('decision tree')


batch_size = 15000
k = 30
cf_model = load_user_CF_model(trainset, batch_size, k)
cb_model = load_content_model(books, train_data, trainset)
train_x, train_y, test_x, test_y = get_train_test_data(trainset, testset, cb_model, cf_model)

for i, model in enumerate(models):
       print('training %s model\n'%(model_name[i]))
       training_hybrid_model(model, train_x, train_y)
       print('test %s model\n'%(model_name[i]))
       result = predict(model, test_x, test_y)
       RMSE = rmse(result, test_y, verbose=True)
       plt.figure()
       plt.plot(np.arange(len(result)), test_y, 'go-', label='true value')
       plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
       plt.title('model %s' %(model_name[i]))
       plt.legend()
       plt.show()

