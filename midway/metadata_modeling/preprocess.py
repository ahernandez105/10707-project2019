
# coding: utf-8

# In[1]:

# will parse a json.gzip file
def parse(path):
    g = gzip.open(path,'rb')
    for l in g:
        yield eval(l)

# will return src data as a df
def get_src(path):
    i = 0
    df = {}
    
    for d in parse(path):
        df[i] = d
        i += 1   
    dataframe = pd.DataFrame.from_dict(df,orient='index')
#     dataframe.columns = SRC_METAD_COLS
    
    return dataframe
    
'''
# will load a src dataset, clean it, add a target col
# and will dump .pickle of the df into cln_path
'''
def dump_clean(src_path,cln_path,target):
    df = get_src(src_path)
    df.dropna(subset = DROP_NAN_COLS,inplace=True)
    df.reset_index(inplace=True)
    df['target'] = target
    
    df.to_pickle(cln_path)
'''
# takes in a list of df paths and concats those dfs into one 
# df and takes a sub sample from that df and generates a 
# train and test from that sub sample and writes train
# and test as pickles
'''
def generate_rand_sample(seed,path_lst,sub_perc,train_perc):
    df = pd.DataFrame()
    
    # concat dfs in path_lst
    for i in path_lst:
        df = pd.concat([df,pd.read_pickle(i)],ignore_index=True)
        
    sub_df = df.sample(frac = sub_perc) # sub sample 
    train = sub_df.sample(frac = train_perc) # train of sub sample
    test = sub_df.drop(train.index) # test of sub sample
    
    train.to_pickle(os.getcwd() + '/data/rand_meta_spa_train_' + str(seed) + '.pickle')
    test.to_pickle(os.getcwd() + '/data/rand_meta_spa_test_' + str(seed) + '.pickle')

'''
# takes in array where each index in the array is a document
# and will return the vocabulary, as a dict, over all the docs
'''
def generate_vocab(array):
    unique_tokens = np.unique(' '.join(array).split(' '))
    d = {}
    
    for i in range(len(unique_tokens)):
        d[unique_tokens[i]] = i
    return d

def generate_one_hot_labels(array,label_dict):
    rows = len(array)
    cols = len(labels_dict.keys())
    out_array = np.zeros((rows,cols))
    
    count = 0
    for i in array:
        out_array[count,i] = 1
        count += 1
        
    return out_array

def is_greater_one_catg(lst):
    return len(lst[0]) > 1

def get_catg_subcatg(lst):
    catg = lst[0][0]
    sub = lst[0][1]
    
    return catg + ':' + sub

def generate_labels_dict(array):
    d = {} 
    
    for i in range(len(array)):
        d[array[i]] = i
    
    return d
