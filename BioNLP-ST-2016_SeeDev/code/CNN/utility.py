##########################################################################################################
###                                            utility.py                                              ###
##########################################################################################################
# This is the utility function for this project

from sklearn.preprocessing import LabelEncoder


################   data matrix   ################
def labelEncoder(labels):
    '''
    Encode labels
    '''
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder

def getWordIdx(token, word2Idx): 
    '''
    Returns from the word2Idx table the word index for a given token
    '''      
    if token in word2Idx.keys():
        return word2Idx[token]
    elif token.lower() in word2Idx.keys():
        return word2Idx[token.lower()]
    else:
        return word2Idx["UNKNOWN_TOKEN"]

def mapDist(dist, min_dist=-30, max_dist=30):
    '''
    Measure distance of all words to flag word.
    '''
    if dist < min_dist:
        return min_dist
    elif dist > max_dist:
        return max_dist
    else:
        return dist

def labelType(loc, entity_locs, entity_type):
    '''
    Label entity and outer, inner-entity location.
    '''
    if loc < min(entity_locs) or loc > max(entity_locs):
        return -1
    elif loc in entity_locs:
        return entity_type
    else:
        return -2