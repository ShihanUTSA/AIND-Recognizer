import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError
        pass


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #raise NotImplementedError
        pass


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        
        # initialize essential objects
        best_score= float("-inf") # initialize at lowest possible number
        best_component_num= 1 # intialize at first component
        best_model= None

        # outer loop iterating over components
        for component_num in range(self.min_n_components, self.max_n_components + 1):

            # initialize storage container for cv scores
            cv_scores= list()
            # grab essential objects
            word_sequences= self.sequences
            num_seqences= len(word_sequences) # count of word sequences
            n_splits= min(3, num_seqences) # 10 splits max, num_seqences splits min
            
            try:
                splitter= KFold(n_splits= n_splits)
            except:
                continue
            # inner loop where CV takes place
            try:
                for cv_train_idx, cv_test_idx in splitter.split(word_sequences):
                    # use indices to get train and test set array and length
                    cv_train_x, cv_train_length= combine_sequences(cv_train_idx, word_sequences)
                    cv_test_x, cv_test_length= combine_sequences(cv_test_idx, word_sequences)
                    # build a model using the cv traning data
                    cv_model= self.base_model(n_components=component_num).fit(cv_train_x, cv_train_length)
                    # get the model score (log likelihood) for the test fold
                    cv_score= cv_model.score(cv_test_x, cv_test_length)
                    cv_scores.append(cv_score)

            # get mean, update best score, extract best component number
            
                mean_scores= np.mean(cv_scores)
                if mean_scores > best_score:
                    mean_scores= best_score
                    best_component_num= component_num

            except:
                #print("failure on {} @ {}".format(self.this_word, component_num))
                continue

        # get the best model
        best_model= self.base_model(best_component_num)
        return best_model


"""
        '''Number of folds. Must be at least 2.'''
        n_splits = 3
        if len(self.sequences) == 2:
            n_splits=2
            
        split_method = KFold(n_splits=n_splits)
        
        ''' At the begining best_score and best_model '''
        best_score = float("-inf")
        best_model = None
        
        ''' Loop over number of hidden states '''
        for hidden_s in range(self.min_n_components, self.max_n_components + 1):
            if len(self.sequences) == 1:
                try :
                    hmm_model = GaussianHMM(n_components=hidden_s, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    new_score = hmm_model.score(self.X, self.lengths)
                except:
                    continue
            else:
                cv_score = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    trained_model = None
                    try:
                        '''Combining the train set with the sequence '''
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        '''Create the HMM model using training data set '''
                        trained_model = GaussianHMM(n_components=hidden_s, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        '''Combining the test set with the sequence '''
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        '''Calculate the score for the new model'''
                        state_score = trained_model.score(X_test, lengths_test)
                        cv_score.append(state_score)
                    except:
                        continue
                
                new_score = np.mean(cv_score)
                
            if new_score > best_score:
                best_score = new_score
                best_model = trained_model
                
        return best_model
"""
