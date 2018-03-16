import time
import logging
import pandas as pd
import numpy as np
import implicit as im
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares as als
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

ALPHA=1
BETA=1
EPISILON=0.1
FACTORNUM=50
SAMPLE=1


# 读取数据
df = pd.read_csv('dataset_1.csv')
df = df.drop(['色相', 'Unnamed: 0', '品名', '色调', '款式流水号'], axis=1)
train_set, test_set, = train_test_split(df, test_size=0.3)

# 生成用户购买表
raw = train_set
raw["buys"] = 1
raw.groupby(["客户ID", "衣服ID"])["buys"].sum().to_csv('uib.csv')


def recoder(od_dataset):
    usr_item_dict = {usr: [i for i in groups['衣服ID']] for usr, groups in list(od_dataset)}
    return usr_item_dict


# 训练集、测试集重新编码
usr_train = train_set.groupby("客户ID")
usr_test = test_set.groupby("客户ID")
usr_item_dict_train = recoder(usr_train)
usr_item_dict_test = recoder(usr_test)

# CUI稀疏矩阵化，为预处理准备
def read_data_latent(filename):
    """ Reads in the last.fm dataset, and returns a tuple of a pandas dataframe
    and a sparse matrix of artist/user/playcount """
    # read in triples of user/artist/playcount from the input dataset
    # get a model based off the input params
    start = time.time()
    logging.debug("reading data from %s", filename)
    data = pd.read_csv(filename,
                       usecols=[0, 1, 2],
                       names=['user', 'item', 'buys'],
                       na_filter=False)
    # map each artist and user to a unique numeric value
    data['user'] = data['user'].astype("category")  # 降低内存使用不会改变显示,将object对象映射为int
    data['item'] = data['item'].astype("category")

    # create a sparse matrix of all the users/plays
    buys = coo_matrix((data['buys'].astype(np.float32),
                       (data['item'].cat.codes.copy(),  # Series.cat.codes 属性来返回 category 类型用来表示每个值的整型值
                        data['user'].cat.codes.copy())))
    logging.debug("read data file in %s", time.time() - start)
    return data, buys


# def get_Latent_Features():
# return item_factors

# 调用ALS模型生成因变量特征矩阵
# return user_map, latent_item_features

def gen_latent_item_features(filename):
    model = als(factors=FACTORNUM, dtype=np.float32, use_gpu=False)
    data, buys = read_data_latent(filename)
    buys = buys.tocsr()
    model.fit(buys)
    item_factors = model.item_factors

    # 建立User和np矩阵的逻辑序列映射
    userset = list(set(data['user']))
    user_map = {}
    for i in range(len(userset)):
        user_map[userset[i]] = i
    # 生成右表Item顺序和衣服ID的映射关系，无用
    latent_itemID = []
    artists = dict(enumerate(data['item'].cat.categories))
    for i in range(item_factors.shape[0]):
        latent_itemID.append(artists[i])
    items_factors_normalized = preprocessing.normalize(item_factors, norm='l2')  # L2正则化
    latent_item_features = pd.DataFrame(items_factors_normalized, index=latent_itemID)
    return data, user_map, latent_item_features


# item_feature.csv
def gen_explicit_item_features(filename):
    explicit_item_features = pd.read_csv(filename)
    explicit_item_features.index = explicit_item_features['衣服ID']
    explicit_item_features = explicit_item_features.drop(['衣服ID'], axis=1)
    return explicit_item_features


# paras: Explicits,Latent,Alpha,Beta
# return item_ID, merged_item_features
def mergeFeatures(explicit, latent, Alpha=1, Beta=1):
    explicit_itemID = explicit.index
    explicit = preprocessing.normalize(explicit, norm='l2')
    explicit = pd.DataFrame(explicit, index=explicit_itemID)
    merged = pd.merge(explicit * Alpha, latent * Beta, how='left', right_index=True, left_index=True)

    # Episilon填充空值
    merged = merged.fillna(EPISILON)

    # 建立item和np矩阵的逻辑序列映射
    itemID = explicit_itemID
    item_map = {}
    for i in range(len(itemID)):
        item_map[itemID[i]] = i

    merged_matrix = merged.as_matrix(columns=None)
    return item_map, merged_matrix


def gen_user_profile(item_matrix, user_map, item_map, data):
    user_profile = np.zeros((len(user_map), item_matrix.shape[1]))

    # 计算user_profile
    for user, item, buys in data.values:
        user_profile[user_map[user]] += item_matrix[item_map[item]] * buys

    # 正则化User Profile
    user_profile = preprocessing.normalize(user_profile, norm='l2')

    return user_profile


def gen_merit_matrix(user_profile, item_matrix):
    scores = user_profile.dot(item_matrix.T)
    return scores



def Rank(user, item_real, scores, user_map, item_map):
    items_bought = usr_item_dict_train[user]
    seq_score = pd.Series(scores[user_map[user]])
    items_score = pd.Series(item_map)
    items_score = items_score.map(seq_score)
    items_score = dict(items_score)
    # if item_real in items_score:
    #     print("origin contains")
    for item_bought in items_bought:
        try:
            items_score.pop(item_bought)
        except:
            pass
    lst = sorted(items_score.items(), key=lambda a: a[1], reverse=True)
    lst = [k for (k, v) in lst]
    # if not item_real in items_score:
    #     print("Later removed")
    Rank = (lst.index(item_real) + 1) / len(lst)
    return Rank




def CB_recommender(scores,user_map,item_map):
    rank=[]
    total=0
    for user in usr_item_dict_test:
        # print("########")
        # print("usr: ",user)
        if user in usr_item_dict_train:
            for item_real in usr_item_dict_test[user]:
                total+=1
                if item_real in usr_item_dict_train[user]:
                    total -= 1
                else:
                    rank.append(Rank(user, item_real, scores, user_map, item_map))
        else:
            rank.append(0.5)
    rank_score = sum(rank) / total
    # print("#####################################")
    # print("Rank Score:", rank_score)
    return rank_score


if __name__ == "__main__":
    def test():
        data, user_map, latent_item_features = gen_latent_item_features('uib.csv')

        explicit_item_features = gen_explicit_item_features('item_feature.csv')

        item_map, merged_matrix = mergeFeatures(explicit_item_features,latent_item_features,ALPHA,BETA)

        user_profile=gen_user_profile(merged_matrix, user_map, item_map, data)

        scores = gen_merit_matrix(user_profile, merged_matrix)

        result = CB_recommender(scores,user_map,item_map)
        return result

    stat=[]

    while SAMPLE>0:

        print("Round",SAMPLE)

        stat.append(test())

        SAMPLE-=1

    stat=np.array(stat)

    print("Average Rank Quality: ",stat.mean(),"StD: ",stat.std())