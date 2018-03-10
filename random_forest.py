import csv
import math
import numpy as np
import scipy.io as sio
import scipy.stats as sstats
from sklearn import preprocessing
from sklearn.utils import shuffle

class TreeNode:
    def __init__(self, feat_i=None, thr=None, is_cat=False, lab=None):
        self.leaf = False
        self.left = None
        self.right = None
        self.feat_i = feat_i
        self.thr = thr
        self.is_cat = is_cat
        self.lab = lab

    def setLeaf(self, label):
        self.leaf = True
        self.label = label

class DecisionTree:
    def __init__(self, max_depth, min_im):
        self.max_depth = max_depth
        self.min_im = min_im
        self.root = None

    def entropy(self, p_lst):
        r = [p for p in p_lst if p > 0.0]
        return np.sum([-p*math.log(p, 2) for p in r])

    def impurity(self, left_hist, right_hist):
        n_l = float(left_hist.sum())
        n_r = float(right_hist.sum())
        h_l = self.entropy(left_hist/n_l)
        h_r = self.entropy(right_hist/n_r)
        return (n_l*h_l+n_r*h_r) / (n_l+n_r)

    def segmentor(self, data, labels, ent):
        best_feat, best_im, best_split = None, ent, None
        best_left, best_right = None, None
        is_cat = False
        feats = np.random.choice(len(data[0]), self.feat_lim, replace=False)
        for feat_i in feats:
            data_col, labs = data[:,feat_i], labels[:]
            ind = np.lexsort((labs, data_col))
            s_data, s_lab = data[ind], labs[ind]
            s_col = s_data[:,feat_i]
            diff = np.diff(s_col)
            split_i = np.where(diff)[0]
            #splits = s_col[split_i][:-1]+diff[split_i]/2.0
            if split_i.size != 0:
                if self.cats is None or self.cats[feat_i] == -1:
                    # not categorical feature
                    for i in (split_i+1):
                        left_hist = np.histogram(s_lab[:i], bins=2)[0]
                        right_hist = np.histogram(s_lab[i:], bins=2)[0]
                        im = self.impurity(left_hist, right_hist)
                        if im < best_im:
                            best_split = (s_col[i] + s_col[i-1])/2.0
                            best_feat, best_im = feat_i, im
                            best_left = (s_data[:i], s_lab[:i])
                            best_right = (s_data[i:], s_lab[i:])
                else:
                    unk = self.cats[feat_i]
                    s_col[s_col==unk] = sstats.mode(s_col[s_col!=unk])[0][0]
                    prev = 0
                    for i in np.append(split_i+1, len(s_lab)):
                        left_lab, right_lab = s_lab[prev:i], np.append(s_lab[:prev],
                                s_lab[i:])
                        left_hist = np.histogram(left_lab, bins=2)[0]
                        right_hist = np.histogram(right_lab, bins=2)[0]
                        im = self.impurity(left_hist, right_hist)
                        if im < best_im:
                            best_split = s_col[prev]
                            best_feat, best_im = feat_i, im
                            best_left = (s_data[prev:i], left_lab)
                            best_right = (np.append(s_data[:prev], s_data[i:],
                                axis=0), right_lab)
                        prev = i
        if best_feat is None:
            return None
        split_node = TreeNode(best_feat, best_split, is_cat)
        return split_node, best_left, best_right

    def train(self, data, labels, cats=None):
        def train_helper(data, labels, depth=0):
            size = labels.size
            ones = np.count_nonzero(labels)
            zeros = size - ones
            ent = self.entropy(np.array([zeros/float(size), ones/float(size)]))
            if depth > self.max_depth or ent < self.min_im or size < 5:
                leaf = TreeNode()
                leaf.setLeaf(1 if ones > zeros else 0)
                return leaf
            split = self.segmentor(data, labels, ent)
            if split is None:
                leaf = TreeNode()
                leaf.setLeaf(1 if ones > zeros else 0)
                return leaf
            split_node, left_d, right_d = split
            split_node.lab = (1 if ones > zeros else 0)
            split_node.ones, split_node.zeros = ones, zeros

            left_node = train_helper(left_d[0], left_d[1], depth+1)
            right_node = train_helper(right_d[0], right_d[1], depth+1)
            left_node.parent, left_node.is_left = split_node, True
            right_node.parent, right_node.is_left = split_node, False
            split_node.left, split_node.right = left_node, right_node
            return split_node
        self.cats = cats
        self.root = train_helper(data, labels)

    def predict(self, data):
        preds = []
        for j, s in enumerate(data):
            node = self.root
            while not node.leaf:
                if node.is_cat:
                    if s[node.feat_i] == node.thr:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if s[node.feat_i] <= node.thr:
                        node = node.left
                    else:
                        node = node.right
            preds.append(node.label)
        return preds
    
    def prune(self, val_data, val_labs):
        result = []
        def post_order(root):
            if not root.leaf:
                post_order(root.left)
                post_order(root.right)
                result.append(root)

        post_order(self.root)
        result.pop()
        prune_list = result
        height = len(prune_list)
        while True:
            best_err = (self.predict(val_data) !=
                    val_labs).sum()/float(len(val_labs))
            best_node, best_leaf = None, None
            best_idx = -1
            for i, n in enumerate(prune_list):
                leaf = TreeNode()
                leaf.setLeaf(n.lab)
                if n.is_left:
                    n.parent.left = leaf
                else:
                    n.parent.right = leaf
                err = (self.predict(val_data) !=
                        val_labs).sum()/float(len(val_labs))
                if err < best_err:
                    best_err = err
                    best_node = n
                    best_leaf = leaf
                    best_idx = i
                if n.is_left:
                    n.parent.left = n
                else:
                    n.parent.right = n
            if best_node is not None:
                if best_node.is_left:
                    best_node.parent.left = best_leaf
                else:
                    best_node.parent.right = best_leaf
                best_leaf.parent = best_node.parent
                del prune_list[best_idx]
            else:
                break

class RandomForest:
    def __init__(self, num_tree):
        self.forest = [DecisionTree(25, 0.1) for _ in xrange(num_tree)]

    def train(self, data, labels, cats=None):
        num_sample = len(data)
        feat_lim = int(math.sqrt(len(data[0])))
        #feat_lim = len(data[0])
        for tree in self.forest:
            tree.feat_lim = feat_lim
            s_idx = np.random.randint(num_sample, size=num_sample)
            tree.train(data[s_idx], labels[s_idx], cats)
            #tree.train(data, labels, cats)

    def prune(self, val_data, val_labs):
        for tree in self.forest:
            tree.prune(val_data, val_labs)

    def predict(self, data):
        preds = [t.predict(data) for t in self.forest]
        return sstats.mode(np.array(preds))[0][0]

def spam():
    email_data = sio.loadmat("./spam-dataset/spam_data.mat")
    
    email_samples = email_data['training_data']
    email_labs = email_data['training_labels'][0]
    email_samples, email_labs = shuffle(email_samples, email_labs)
    email_test = email_data['test_data']
    val_samples, val_labs = email_samples[:500], email_labs[:500]
    prune_samples, prune_labs = email_samples[500:1500], email_labs[500:1500]
    email_samples, email_labs = email_samples[1500:], email_labs[1500:]
    
    classifier = RandomForest(25)
    classifier.train(email_samples, email_labs)
    classifier.prune(prune_samples, prune_labs)
    print (classifier.predict(val_samples) == val_labs).sum()/float(len(val_labs))

    #preds = classifier.predict(email_test)
    #with open('spam_preds.csv', 'w') as csvfile:
    #    fieldnames = ['Id', 'Category']
    #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #    
    #    writer.writeheader()
    #    index = 1
    #    for i in range(len(preds)):
    #        writer.writerow({'Id': str(i+1), 'Category': str(preds[i])})
    #        index += 1

def census():
    cats = ['workclass', 'education', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    raw_data = np.genfromtxt("./census-dataset/data.csv", dtype=str, delimiter=',')
    labs = raw_data[:,-1][1:].astype(int)
    raw_data = raw_data[:,:-1]
    header, samples = raw_data[0], raw_data[1:]
    idx_cats = np.array([i for i, v in enumerate(header) if v in cats])
    encoders = []
    for i, v in enumerate(header):
        if v in cats:
            le = preprocessing.LabelEncoder()
            samples[:,i] = le.fit_transform(samples[:,i])
            encoders.append(le)
        else:
            encoders.append(None)
    samples = samples.astype(int)

    cats = []
    for e in encoders:
        if e is None:
            cats.append(-1)
        elif '?' in list(e.classes_):
            cats.append(e.transform(['?'])[0])
        else:
            cats.append(-1)
        
    samples, labs = shuffle(samples, labs)
    val_samples, val_labs = samples[:3636], labs[:3636]
    m_samples, m_labs = samples[3636:], labs[3636:]

    test_data = np.genfromtxt("./census-dataset/test_data.csv", dtype=str, delimiter=',')
    header, test_samples = test_data[0], test_data[1:]
    for i, e in enumerate(encoders):
        if e is not None:
            test_samples[:,i] = e.transform(test_samples[:,i])
    test_samples = test_samples.astype(int)

    classifier = RandomForest(25)
    classifier.train(samples, labs, cats)
    preds = classifier.predict(val_samples)
    print (preds == val_labs).sum()/float(len(val_labs))

    #with open('census_preds.csv', 'w') as csvfile:
    #    fieldnames = ['Id', 'Category']
    #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #    
    #    writer.writeheader()
    #    index = 1
    #    for i in range(len(preds)):
    #        writer.writerow({'Id': str(i+1), 'Category': str(preds[i])})
    #        index += 1

spam()
#census()
