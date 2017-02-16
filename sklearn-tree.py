from sklearn import tree

APPLE = 0
ORANGE = 1

FEAT_SMOOTH = 0
FEAT_BUMPY = 1

features = [[140, FEAT_BUMPY], [130, FEAT_BUMPY], [150, FEAT_SMOOTH], [170, FEAT_SMOOTH]]
labels =   [APPLE, APPLE, ORANGE, ORANGE]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

assert ORANGE == clf.predict([[160, FEAT_SMOOTH]])
assert APPLE == clf.predict([[120, FEAT_BUMPY]])
