import numpy as np

#we create a class to represent each node of the tree
class Node:
    def __init__(self):
        self.is_leaf=False
        self.label=None
        self.branches=[]



#we create a class with some member methods to represent the tree itself
#it uses id3 to build the tree

class Decision_tree_id3:
    def __init__(self):
        self.root=None
        self.features=None
    
    def __compute_gain(self,X,y,index):
        values=X[...,index]
        unique_values,frequencies=np.unique(values,return_counts=True)
        subsets=[]
        for uv in unique_values :
            y_subset=[]
            for i,v in enumerate(values):
                if v==uv:
                    y_subset.append(y[i])
            subsets.append(y_subset)
        assert len(subsets)==len(frequencies)
        temp=[]
        for freq,subset in zip(frequencies,subsets):
            temp.append(freq / len(X) * self.__compute_entropy(subset))
        return self.__compute_entropy(y)-np.sum(np.array(temp))
    
    def __comput_best_feature(self, X, y, features):
        info_gains =[]
        for feature in features:
            temp=self.__compute_gain(X, y, feature)
            info_gains.append(temp)
                      
        best_feature= features[info_gains.index(max(info_gains))]
        return best_feature
    def __id3(self, X, y, feature_names):
        node = Node()

        if len(set(y)) == 1:
            node.is_leaf = True
            node.label = y[0]
            return node

        if len(feature_names) == 0:
            node.is_leaf = True
            unique_vals, counts = np.unique(y, return_counts=True)
            node.label = unique_vals[np.argmax(counts)]
            return node

        #  choose the feature that maximizes the information gain
        best_feature = self.__comput_best_feature(X, y, feature_names)
        node.label = best_feature

        #  of the chosen feature for each instance
        best_feature_id = best_feature
        feature_values = list(set(X[:, best_feature_id]))

        for feature_value in feature_values:
            branch = [feature_value, Node()]
            node.branches.append(branch)

            X_subset = X[X[:, best_feature_id] == feature_value]
            y_subset = y[X[:, best_feature_id] == feature_value]

            if len(X_subset) == 0:
                unique_vals, counts = np.unique(y, return_counts=True)
                branch[1].label = unique_vals[np.argmax(counts)]
            else:
                feature_names = [
                    a for a in feature_names if a != best_feature]
                branch[1] = self.__id3(X_subset, y_subset, feature_names)
        return node
    def fit(self, X, y):
        self.features = np.array(range(X.shape[1]))
        self.root = self.__id3(np.array(X), np.array(y), np.array(range(X.shape[1])))
    #rock solid
    def __compute_entropy (self,info: np.array):
        number_of_samples=len(info)
        values,frequencies=np.unique(info,return_counts=True)
        probs=frequencies/number_of_samples
        entropy=np.sum(np.log2(probs)*probs)
        return entropy
    

    def predict(self, X):
        y_pred = [self.__traverse(self.root, sample) for sample in X]
        return y_pred

    def __traverse(self, node, sample):
        if node.is_leaf:
            return node.label

        feature_name = node.label
        feature_id = self.features.tolist().index(feature_name)
        if node.branches:
            for b in node.branches:
                if b[0] == sample[feature_id]:
                    return self.__traverse(b[1], sample)

        return node.label
    
    
from sklearn import datasets,model_selection
#we only use sklearn to load the iris dataset and train test_split
iris = datasets.load_iris()
X,y=iris["data"],iris["target"]
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,train_size=0.8,random_state=342)


tree=Decision_tree_id3()
tree.fit(X_train,y_train)
pred=tree.predict(X_test)
#print (np.array(pred),"\n",np.array(y_test))
accurate=np.where (pred==np.array(y_test).astype("int32"))[0].shape[0]
acc=accurate/X_test.shape[0]
pred=tree.predict(X_train)
accurate=np.where (pred==np.array(y_train).astype("int32"))[0].shape[0]
acc_train=accurate/y_train.shape[0]
print (f"acc test:{acc} acc_train:{acc_train}")
