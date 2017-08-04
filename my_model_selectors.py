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
        
        # initialize variables with score being the largest possible number
        best_score= float("inf")
        best_model= None

        # get other attributes we need for BIC calculation
        num_features= len(self.X[0])
        N= np.sum(self.lengths)
        logN= np.log(N)

        for component_num in range(self.min_n_components, self.max_n_components + 1):
            try:

                # get model and log likelihood
                model= self.base_model(component_num) #GaussianHMM
                logL= model.score(self.X, self.lengths)

                # calculate parameters
                p= (component_num**2) + 2 * num_features * component_num - 1

                # calculate BIC
                score= -2 * logL + p * logN

                # update score and model that generated score
                if score < best_score:
                    best_score= score
                    best_model= model

            except:
                #print("failure on {} @ {}".format(self.this_word, component_num))
                pass

        return best_model
        

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError
        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
            
        best_score = float("-inf")
        best_model = None
        num_features = self.X.shape[1]
        for i_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                new_model = self.base_model(i_components)
                logL = new_model.score(self.X, self.lengths)

                #Calculate all other words average score
                partial_score = 0
                count = 0
                for word in self.words:
                    if word != self.this_word:
                        new_X, new_lengths = self.hwords[word]
                        try:
                            partial_score += hmm_model.score(new_X, new_lengths)
                            count += 1
                        except:
                            pass
                if count > 0:
                    logAllButword = partial_score/count
                else:
                    logAllButword = 0

                #Calculate the total score
                new_score = logL - logAllButword
                #Only keep the max value
                best_score, best_model = max((best_score,best_model),(new_score,new_model))
            except:
                pass

        return best_model
'''
# initialize essential objects
        best_score= float("-inf") # initialize at lowest possible number
        best_model= None
        # outer loop iterating over components
        for component_num in range(self.min_n_components, self.max_n_components + 1):
            words_left_scores= list()

            try:
                model= self.base_model(component_num)
                score= model.score(self.X, self.lengths) # log(P(X(i)) for this word
                words= self.words # get all the words as dict with words as keys
                # generate a dict of all words except ith word
                words_left= words.copy() # copy the dict so we don't alter words dict
                words_left.pop(self.this_word) # remove this word
                # iterate over all the other words and sum up P (logL)
                for word in words_left:
                    X, lengths= self.hwords[word] # hwords is a dict with values of X and length for each key (word)
                    try:
                        words_left_scores.append(model.score(X, lengths)) # log(P(X(i)) for this word
                    except:
                        pass

                # put it all together
                M= len(words_left)
                words_left_score= np.sum(words_left_scores)
                normalized_words_left_score= words_left_score/M

                # update best score
                DIC= score - words_left_score
                if DIC > best_score:
                    best_score= DIC
                    best_model= model
            except:
                pass
        return best_model
'''
        # TODO implement model selection based on DIC scores
        #raise NotImplementedError
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        
        ''' At the begining best_score and best_hidden_state '''
        best_score = float("-inf")
        best_hidden_s = None
        
        '''Number of folds. Must be at least 2. If only 1 sequence provided, then
           that full dataset is used for both train and test'''
        n_splits=min(len(self.sequences),3)    
        split_method = KFold(n_splits=n_splits)
        
        
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
                
                if cv_score:
                    new_score = np.mean(cv_score)
                else:
                    new_score=0
                    
                
            if new_score > best_score:
                best_score = new_score
                best_hidden_s = hidden_s
                
        if best_hidden_s == 0:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_hidden_s) 
                


