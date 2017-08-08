import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    ''' Iterate over sequences to get X and length of X '''
    for index, sequence in test_set.get_all_Xlengths().items():
        X_test, Xlength_test = test_set.get_item_Xlengths(index)
        ''' Dictionary to store word and score '''
        words_score = dict()
        for word, model in models.items():
            try: 
                ''' Get the new score '''
                words_score[word] = model.score(X_test, Xlength_test)
            except:
                words_score[word]= float('-inf')
                       
        probabilities.append(words_score)
    
    for probs in probabilities:
        guesses.append(max(probs, key=probs.get))

    return probabilities,guesses
'''
############################################################################################################
    test_sequences = list(test_set.get_all_Xlengths().values())

    for test_X, test_Xlength in test_sequences:
        words_logL = dict()
        for word, hmm_model in models.items():
            try:
                words_logL[word] = hmm_model.score(test_X, test_Xlength)
            except:
                words_logL[word]= -1000000
                continue
        probabilities.append(words_logL)


    for probs in probabilities:
        guesses.append(max(probs, key=probs.get))

    return probabilities, guesses
####################################################################################
warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # get all sequences in the test set
    all_sequences= test_set.get_all_sequences()

    # iterate over sequences get X and lengths for each key
    for index, sequence in all_sequences.items():
        X, lengths= test_set.get_item_Xlengths(index)
        guess_probs= dict() # container to hold the word:probabilities for each sequnence for each model

      # iterate over each word:model pair where the model is the trained model for that word
        for word, word_model in models.items():
            try:
                prob= word_model.score(X, lengths)
                guess_probs[word]= prob
            except:
                guess_probs[word]= float('-inf') # lowest possible score

        probabilities.append(guess_probs) # add dictonary to list
        # get return key with highest score as best guess
        # thanks stack overflow! http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        best_guess= max(guess_probs, key=guess_probs.get)
        guesses.append(best_guess)

    return probabilities, guesses
    '''