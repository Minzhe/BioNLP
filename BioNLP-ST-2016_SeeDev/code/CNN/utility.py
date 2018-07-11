##########################################################################################################
###                                            utility.py                                              ###
##########################################################################################################
# This is the utility function for this project

import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


##############################   data matrix   ##############################
# >>>>>>>>>>>>>>>>>>       label       <<<<<<<<<<<<<<<<< #
class hLabelEncoder(object):
    '''
    Encode labels with hierarchy
    '''
    def __init__(self):
        self.__label_hierarchy = {'Exists_In_Genotype':    ('When_Where_01', 0, 0),
                                    'Occurs_In_Genotype':    ('When_Where_02', 1, 0),
                                    'Exists_At_Stage':       ('When_Where_03', 2, 0),
                                    'Occurs_During':         ('When_Where_04', 3, 0),
                                    'Is_Localized_In':       ('When_Where_05', 4, 0),
                                    'Is_Involved_In_Process':        ('Function_01', 5, 1),
                                    'Transcribes_Or_Translates_To':  ('Function_02', 6, 1),
                                    'Is_Functionally_Equivalent_To': ('Function_03', 7, 1),
                                    'Regulates_Accumulation':        ('Regulation_01', 8, 2),
                                    'Regulates_Development_Phase':   ('Regulation_02', 9, 2),
                                    'Regulates_Expression':          ('Regulation_03', 10, 2),
                                    'Regulates_Molecule_Activity':   ('Regulation_04', 11, 2),
                                    'Regulates_Process':             ('Regulation_05', 12, 2),
                                    'Regulates_Tissue_Development':  ('Regulation_06', 13, 2),
                                    'Composes_Primary_Structure':    ('Composition_01', 14, 3),
                                    'Composes_Protein_Complex':      ('Composition_02', 15, 3),
                                    'Is_Protein_Domain_Of':          ('Composition_03', 16, 3),
                                    'Is_Member_Of_Family':           ('Composition_04', 17, 3),
                                    'Has_Sequence_Identical_To':     ('Composition_05', 18, 3),
                                    'Binds_To':          ('Interaction_01', 19, 4),
                                    'Interacts_With':    ('Interaction_02', 20, 4),
                                    'Is_Linked_To':      ('Interaction_03', 21, 4)}
        self.__class_label = {value[1]: key for key, value in self.__label_hierarchy.items()}

    def label2class(self, labels):
        return np.array([self.__label_hierarchy[label][1] for label in labels])
    
    def label2hclass(self, labels):
        return np.array([self.__label_hierarchy[label][2] for label in labels])

    def class2label(self, classes):
        return np.array([self.__class_label[cls_] for cls_ in classes])

    def class2hierarchy(self, classes):
        labels = self.class2label(classes)
        return np.array([self.__label_hierarchy[label][0] for label in labels])
    

# >>>>>>>>>>>>>>>>>>       map encoding       <<<<<<<<<<<<<<<<< #
class encodeMapper(object):
    '''
    Map words, distance and entity type to index.
    '''
    def __init__(self):
        pass

    def mapWordIdx(self, token, word2Idx): 
        '''
        Returns from the word2Idx table the word index for a given token
        '''      
        if token in word2Idx.keys():
            return word2Idx[token]
        elif token.lower() in word2Idx.keys():
            return word2Idx[token.lower()]
        else:
            return word2Idx["UNKNOWN_TOKEN"]

    def mapDist(self, dist, min_dist=-30, max_dist=30):
        '''
        Measure distance of all words to flag word.
        '''
        if dist < min_dist:
            return min_dist
        elif dist > max_dist:
            return max_dist
        else:
            return dist

    def mapType(self, loc, entity_locs, entity_type):
        '''
        Label entity and outer, inner-entity location.
        '''
        if loc < min(entity_locs) or loc > max(entity_locs):
            return -1
        elif loc in entity_locs:
            return entity_type
        else:
            return -2


##############################   data sampler   ##############################
def bernRV(p):
    if p > 1:
        p = 1
    return bernoulli.rvs(p, loc=0, size=1)[0]

class dataSampler(object):
    '''
    Resample imbalanced data to balance different class instance. 
    '''
    def __init__(self, y, target_n=100):
        self.y = np.array(y)
        self.labels, self.label_counts = np.unique(y, return_counts=True)
        self.n_labels = len(self.labels)
        self.p_labels = {label:(target_n/self.n_labels/count) for label, count in zip(self.labels, self.label_counts)}
    
    def sample(self):
        y_bern = np.array([bernRV(inst) for inst in self.y])
        if (np.sum(y_bern) == 0):
            raise ValueError('No instance got sampled, consider enlarge target n.')
        return y_bern == 1
        


############################        metrics        #################################
def calMetrics(y_pred, y_true, label_encoder, y_train):
    acc = round(accuracy_score(y_pred=y_pred, y_true=y_true), 3)
    prec = calPrec(y_pred=y_pred, y_true=y_true)
    recall = calRecall(y_pred=y_pred, y_true=y_true)
    f1 = calF1(y_pred=y_pred, y_true=y_true)
    # get labels
    classes = np.unique(np.concatenate([y_pred, y_true]))
    classes = np.sort(classes)
    count_test = np.vectorize(lambda x: np.sum(y_true == x))
    count_train = np.vectorize(lambda x: np.sum(y_train == x))
    class_counts_test = count_test(classes)
    class_counts_train = count_train(classes)
    labels = label_encoder.class2label(classes)
    hierarchies = label_encoder.class2hierarchy(classes)
    # table
    metrics_table = pd.DataFrame({'class':classes, 'hierarchy':hierarchies, 'label':labels, 'train_count':class_counts_train, 'test_count':class_counts_test, 'precision':prec[3], 'recall':recall[3], 'f1':f1[3]})
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'hierarchy':None, 'label':'Micro', 'train_count':None, 'test_count':None, 'precision':prec[0], 'recall':recall[0], 'f1':f1[0]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'hierarchy':None, 'label':'Macro', 'train_count':None, 'test_count':None, 'precision':prec[1], 'recall':recall[1], 'f1':f1[1]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'hierarchy':None, 'label':'Weighted', 'train_count':None, 'test_count':None, 'precision':prec[2], 'recall':recall[2], 'f1':f1[2]}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.append(pd.DataFrame({'class':None, 'hierarchy':None, 'label':'Accuracy', 'train_count':None, 'test_count':None, 'precision':acc, 'recall':None, 'f1':None}, index=[len(metrics_table)+1]))
    metrics_table = metrics_table.sort_values(by=['class'])
    return metrics_table

def calPrec(y_pred, y_true):
    prec_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    prec_macro = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    prec_weighted = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    prec_all = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    return round(prec_micro, 3), round(prec_macro, 3), round(prec_weighted, 3), np.around(prec_all, 3)

def calRecall(y_pred, y_true):
    recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_weighted = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall_all = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    return round(recall_micro, 3), round(recall_macro, 3), round(recall_weighted, 3), np.around(recall_all,3)

def calF1(y_pred, y_true):
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_all = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    return round(f1_micro, 3), round(f1_macro, 3), round(f1_weighted, 3), np.around(f1_all, 3)

