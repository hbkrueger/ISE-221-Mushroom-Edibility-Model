import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd

# initialize gini and p0, p1 vals
gini_vals = []
p0_vals = []
p1_vals = []

df = pd.read_csv('train.csv', quotechar='"')

# create tree model with depth 10 using gini measure
tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=7, 
                                    random_state=1) 

y_train = df["class_p"]
X_train = df.drop('class_p', axis=1)

tree_model.fit(X_train, y_train) # add training feature cols and class col to tree

feature_names = df.columns.drop('class_p').tolist() # gather names of feature cols

# plot tree
tree.plot_tree(tree_model, 
               feature_names=feature_names,
               filled=True)

# show tree
plt.savefig('dec_tree.pdf')
plt.show()

# calculate gini for each node
for i in range(tree_model.tree_.node_count):
    p = tree_model.tree_.value[i][0]
    gini = 1 - p[0]**2 - p[1]**2
    p0_vals.append(p[0])
    p1_vals.append(p[1])
    gini_vals.append(gini)

# show gini graph
plt.clf()
plt.scatter(p0_vals, gini_vals, label="Probability of Edible")
plt.scatter(p1_vals, gini_vals, label="Probability of Poisonous")
plt.title("Probability vs. Impurity (Gini)")
plt.xlabel('Probability of Class 0 or Class 1')
plt.ylabel('Impurity')
plt.grid(True)
plt.legend()
plt.savefig("gini_graph.pdf")
plt.show()




