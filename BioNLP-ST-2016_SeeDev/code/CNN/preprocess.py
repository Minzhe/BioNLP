##########################################################################################################
###                                           preprocess.py                                            ###
##########################################################################################################
'''
This script process and clean the row train and dev data.
'''
import os
import re
import numpy as np
import pandas as pd
import glob
pd.set_option('display.max_columns', 500)
from nltk.tokenize import word_tokenize


##########################   extract information  ################################
# >>>>>>>>>>> relation <<<<<<<<<<< #
def extractRelation(file):
    '''
    Extract relation from .a2 files.
    '''
    rel, e1, e1_role, e2, e2_role = [], [], [], [], [] 
    with open(file, encoding='utf-8') as f:
        for line in f:
            rel_, e1_, e2_ = tuple(line.strip().split('\t')[1].split(' '))
            e1_role_, e1_ = tuple(e1_.split(':'))
            e2_role_, e2_ = tuple(e2_.split(':'))
            rel.append(rel_)
            e1.append(e1_)
            e2.append(e2_)
            e1_role.append(e1_role_)
            e2_role.append(e2_role_)
    rel_data = pd.DataFrame({'rel':rel, 'e1':e1, 'e2':e2, 'e1_role':e1_role, 'e2_role':e2_role})
    return rel_data

# >>>>>>>>>>> entity <<<<<<<<<<< #
def extractEntity(file):
    '''
    Extract entities from .a1 files.
    '''
    entities = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line_ = unicodetoascii(line)
            idx, e_type_pos, e = tuple(line_.strip().split('\t'))
            # entity type
            e_type = e_type_pos.split(' ')[0]
            e_pos = ' '.join(e_type_pos.split(' ')[1:])
            # start and end position
            pos_start, pos_end = [], []
            for pos in e_pos.split(';'):
                pos_start.append(int(pos.split(' ')[0]))
                pos_end.append(int(pos.split(' ')[1]))
            # dict
            entities[idx] = {'type': e_type,
                             'pos_start': np.array(pos_start),
                             'pos_end': np.array(pos_end),
                             'entity': e}
    return entities

# >>>>>>>>>>> sentense <<<<<<<<<<< #
def extractSentence(file):
    '''
    Split text into sentenses
    '''
    ### read text
    text = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            text.append(line.strip())
    text = ' '.join(text)

    ### split sentense
    text = unicodetoascii(text)
    text = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!et al\.)(?<!Fig\.)(?<=\.|\?)\s", text)

    ### sentense and position
    sents = {}
    start_, end_ = 0, 0
    for idx, sent_ in enumerate(text):
        end_ = start_ + len(sent_) - 1
        sents[idx+1] = {'sent': sent_, 'start': start_, 'end': end_}
        start_ = end_ + 2               # prepare for the next, account for space
    
    return sents

# >>>>>>>>>>> find sentense <<<<<<<<<<< #
def findSent(sents, start, end):
    '''
    Find the sentense where start and end of an entity is located.
    '''
    for idx in sents.keys():
        if sents[idx]['start'] <= start and sents[idx]['end'] >= end:
            return sents[idx]['sent'], sents[idx]['start'], sents[idx]['end']
    # cross 2 sentense relation
    for idx in list(sents.keys())[:-1]:
        if sents[idx]['start'] <= start and sents[idx+1]['end'] >= end:
            print('.... Warning: Cross 2 sentense relation found.\tStart: {}\t End: {}'.format(start, end), flush=True)
            return ' '.join([sents[idx]['sent'], sents[idx+1]['sent']]), \
                   sents[idx]['start'], \
                   sents[idx+1]['end']
    # cross 3 sentense relation
    for idx in list(sents.keys())[:-2]:
        if sents[idx]['start'] <= start and sents[idx+2]['end'] >= end:
            print('.... Warning: Cross 3 sentense relation found.\tStart: {}\t End: {}'.format(start, end), flush=True)
            return ' '.join([sents[idx]['sent'], sents[idx+1]['sent'], sents[idx+2]['sent']]), \
                   sents[idx]['start'], \
                   sents[idx+2]['end']
    raise IndexError('Sentense not found!\nStart: {}\t End: {}'.format(start, end))

# >>>>>>>>>>> mark entity <<<<<<<<<<< #
def markEntity(sent, entity, start, end):
    '''
    Locate entities in the sentense.
    '''
    # check for entity correctness
    e = ''
    for pos in zip(start, end):
        e += ' ' + sent[pos[0]:pos[1]]
    e = e.strip()
    if e != entity:
        print('{},{}'.format(sent[start[0]:end[0]], sent[start[1]:end[1]]))
        raise IndexError('Entity not found.\nEntity:"{}"\nGet:"{}"\nSentence:{}'.format(entity, e, sent))
    
    sent_marked = sent
    entity_left = entity
    idx_move_stack = 0

    for (pos_start, pos_end) in zip(start, end):
        idx_start = pos_start + idx_move_stack
        idx_end = pos_end + idx_move_stack
        sent_marked, entity_left, idx_move = markWord(sent=sent_marked, entity=entity_left, start=idx_start, end=idx_end)
        idx_move_stack += idx_move
    
    return sent_marked

def markWord(sent, entity, start, end):
    '''
    Mark entity in the sentense.
    '''
    # check
    if sent[start:end] not in entity:
        raise IndexError('Entity location not correct.\nEntity:"{}"\nGet:"{}"\nSentence:{}'.format(entity, sent[start:end], sent))
    
    # substitute
    marker = re.sub(r'[^ ]+', ' **marker** ', sent[start:end])      # !!! add space to both end to ensure token identification
    sent_new = sent[0:start] + marker + sent[end:]
    entity_new = entity.replace(sent[start:end], '').strip()
    # moved index
    idx_move = len(sent_new) - len(sent)

    return sent_new, entity_new, idx_move


# >>>>>>>>>>> locate entity <<<<<<<<<<< #
def locateEntity(sent_marked):
    '''
    Find mark location in the sentense.
    '''
    tokens = word_tokenize(sent_marked)
    idxs = [idx for idx, token in enumerate(tokens) if token == '**marker**']
    if len(idxs) == 0:
        raise ValueError('No marker found in sentense:\n{}'.format(tokens))
    return idxs


# >>>>>>>>>>> tokenize sentense having entity <<<<<<<<<<< #
def entityTokenize(sent, start, end):
    '''
    Adding space to both end of two entities in a sentense, then tokenize.
    '''
    # padding
    sent = ' ' + sent + ' '
    start = [i+1 for i in start]
    end = [i+1 for i in end]

    # get entity
    e = []
    for i, j in zip(start, end):
        e.append(' ' + sent[i:j] + ' ')
    # other than entity
    o = []
    o_start = [0] + end
    o_end = start + [len(sent)]
    for i, j in zip(o_start, o_end):
        o.append(sent[i:j])
    
    # concat new sent
    if len(o) == len(e) + 1:
        sent_new = o[0]
        for idx in range(len(e)):
            sent_new += e[idx] + o[idx+1]
        return word_tokenize(sent_new)
    else:
        raise ValueError('Somthing goes wrong, need to check this function.')
    


############################   utility   ############################
def unicodetoascii(text):
    '''
    Replace some annoying character
    '''
    TEXT = (text.
            replace(u'\xa0', ' ').
    		replace(b'\xe2\x80\x99'.decode('utf-8'), "'").
            replace(b'\xc3\xa9'.decode('utf-8'), 'e').
            replace(b'\xe2\x80\x90'.decode('utf-8'), '-').
            replace(b'\xe2\x80\x91'.decode('utf-8'), '-').
            replace(b'\xe2\x80\x92'.decode('utf-8'), '-').
            replace(b'\xe2\x80\x93'.decode('utf-8'), '-').
            replace(b'\xe2\x80\x94'.decode('utf-8'), '-').
            replace(b'\xe2\x80\x94'.decode('utf-8'), '-').
            replace(b'\xe2\x80\x98'.decode('utf-8'), "'").
            replace(b'\xe2\x80\x9b'.decode('utf-8'), "'").
            replace(b'\xe2\x80\x9c'.decode('utf-8'), '"').
            replace(b'\xe2\x80\x9c'.decode('utf-8'), '"').
            replace(b'\xe2\x80\x9d'.decode('utf-8'), '"').
            replace(b'\xe2\x80\x9e'.decode('utf-8'), '"').
            replace(b'\xe2\x80\x9f'.decode('utf-8'), '"').
            replace(b'\xe2\x80\xa6'.decode('utf-8'), '...').
            replace(b'\xe2\x80\xb2'.decode('utf-8'), "'").
            replace(b'\xe2\x80\xb3'.decode('utf-8'), "'").
            replace(b'\xe2\x80\xb4'.decode('utf-8'), "'").
            replace(b'\xe2\x80\xb5'.decode('utf-8'), "'").
            replace(b'\xe2\x80\xb6'.decode('utf-8'), "'").
            replace(b'\xe2\x80\xb7'.decode('utf-8'), "'").
            replace(b'\xe2\x81\xba'.decode('utf-8'), "+").
            replace(b'\xe2\x81\xbb'.decode('utf-8'), "-").
            replace(b'\xe2\x81\xbc'.decode('utf-8'), "=").
            replace(b'\xe2\x81\xbd'.decode('utf-8'), "(").
            replace(b'\xe2\x81\xbe'.decode('utf-8'), ")"))
    return TEXT


def concatRegion(start1, end1, start2, end2):
    '''
    Concat differnt start and end positions to remove overlap.
    '''
    start, end = [], []
    sequence = np.zeros(max(max(end1), max(end2)))
    for i, j in zip(start1, end1):
        sequence[i:j] = 1
    for i, j in zip(start2, end2):
        sequence[i:j] = 1
    sequence = [0] + list(sequence) + [0]

    for idx in range(1, len(sequence)):
        if sequence[idx] == 1 and sequence[idx-1] == 0:
            start.append(idx)
    for idx in range(len(sequence)-1):
        if sequence[idx] == 1 and sequence[idx+1] == 0:
            end.append(idx+1)
    
    return [i-1 for i in start], [j-1 for j in end]



###########################  main prepare relation and entity table  ########################
def extractRelEnt(path, doc_id):
    '''
    Extract relation and entities data.
    '''
    print('Extracting information for {}'.format(doc_id), flush=True)
    entities = extractEntity(os.path.join(path, doc_id) + '.a1')
    rel_data = extractRelation(os.path.join(path, doc_id) + '.a2')
    rel_data['e1_type'] = rel_data.e1.apply(lambda x: entities[x]['type'])
    rel_data['e2_type'] = rel_data.e2.apply(lambda x: entities[x]['type'])
    rel_data['e1_pos_start'] = rel_data.e1.apply(lambda x: entities[x]['pos_start'])
    rel_data['e1_pos_end'] = rel_data.e1.apply(lambda x: entities[x]['pos_end'])
    rel_data['e2_pos_start'] = rel_data.e2.apply(lambda x: entities[x]['pos_start'])
    rel_data['e2_pos_end'] = rel_data.e2.apply(lambda x: entities[x]['pos_end'])
    # do these at last
    rel_data['e1'] = rel_data.e1.apply(lambda x: entities[x]['entity'])
    rel_data['e2'] = rel_data.e2.apply(lambda x: entities[x]['entity'])

    # find correspond sentense
    sents_all = extractSentence(os.path.join(path, doc_id) + '.txt')
    rel_data['sent'], rel_data['sent_start'], rel_data['sent_end'] = None, None, None
    for idx, row in rel_data.iterrows():
        try:
            rel_data['sent'][idx], \
            rel_data['sent_start'][idx], \
            rel_data['sent_end'][idx] = \
            findSent(sents=sents_all, 
                    start=min(row['e1_pos_start'][0], row['e2_pos_start'][0]), 
                    end=max(row['e1_pos_end'][-1], row['e2_pos_end'][-1]))
        except IndexError as e:
            print(e, '\nrow: {}'.format(idx+1))
            exit()

    # reindex each sentense
    rel_data['e1_pos_start'] = rel_data['e1_pos_start'] - rel_data['sent_start']
    rel_data['e1_pos_end'] = rel_data['e1_pos_end'] - rel_data['sent_start']
    rel_data['e2_pos_start'] = rel_data['e2_pos_start'] - rel_data['sent_start']
    rel_data['e2_pos_end'] = rel_data['e2_pos_end'] - rel_data['sent_start']
    rel_data['sent_end'] = rel_data['sent_end'] - rel_data['sent_start']
    # do this at last
    rel_data['sent_start'] = rel_data['sent_start'] - rel_data['sent_start']

    # Tokenize, convert character index to token index
    rel_data['e1_loc'], rel_data['e2_loc'] = None, None
    for idx, row in rel_data.iterrows():
        try:
            sent_marked_e1 = markEntity(row['sent'], entity=row['e1'], start=row['e1_pos_start'], end=row['e1_pos_end'])
            sent_marked_e2 = markEntity(row['sent'], entity=row['e2'], start=row['e2_pos_start'], end=row['e2_pos_end'])
            rel_data['e1_loc'][idx] = locateEntity(sent_marked_e1)
            rel_data['e2_loc'][idx] = locateEntity(sent_marked_e2)
            e12_start, e12_end = concatRegion(start1=row['e1_pos_start'], end1=row['e1_pos_end'], 
                                              start2=row['e2_pos_start'], end2=row['e2_pos_end'])
            tokens = entityTokenize(sent=row['sent'], start=e12_start, end=e12_end)
            rel_data['sent'][idx] = ' '.join(tokens)
        except IndexError as e:
            print(e, '\nrow: {}'.format(idx+1))
            exit()
    # locate entity
    rel_data = rel_data[['rel', 'e1_loc', 'e1_type', 'e2_loc', 'e2_type', 'sent']]
    rel_data['e1_loc'] = rel_data['e1_loc'].apply(lambda x: list(map(str, x))).apply(lambda x: ','.join(x))
    rel_data['e2_loc'] = rel_data['e2_loc'].apply(lambda x: list(map(str, x))).apply(lambda x: ','.join(x))

    return rel_data


if __name__ == '__main__':
    train_path = "/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/BioNLP-ST-2016_SeeDev-binary_train"
    dev_path = "/home/t-mizha/project/BioNLP/BioNLP-ST-2016_SeeDev/data/BioNLP-ST-2016_SeeDev-binary_dev"
    train_doc = [os.path.basename(f).strip('.txt') for f in glob.glob(os.path.join(train_path, '*.txt'))]
    dev_doc = [os.path.basename(f).strip('.txt') for f in glob.glob(os.path.join(dev_path, '*.txt'))]

    train_rel = []
    print('---------- trian data ----------')
    for doc in train_doc:
        train_rel.append(extractRelEnt(path=train_path, doc_id=doc))
    train_rel = pd.concat(train_rel)
    train_rel.to_csv(os.path.join(train_path, '..', 'train_relent.txt'), sep='\t', index=False)

    dev_rel = []
    print('\n---------- dev data ----------')
    for doc in dev_doc:
        dev_rel.append(extractRelEnt(path=dev_path, doc_id=doc))
    dev_rel = pd.concat(dev_rel)
    dev_rel.to_csv(os.path.join(dev_path, '..', 'dev_relent.txt'), sep='\t', index=False)
