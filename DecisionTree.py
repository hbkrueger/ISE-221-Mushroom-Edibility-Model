import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import pandas as pd


# initialize gini and p0, p1 vals
gini_vals = []
p0_vals = []
p1_vals = []

df = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')
#=================================== function to plot the confusion matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Edible", "Poisonous"],
                yticklabels=["Edible", "Poisonous"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
#====================================

# create tree model with depth 10 using gini measure
tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    random_state=1)

y_train = df["class_p"]
X_train = df.drop('class_p', axis=1)

X_val = val.drop("class_p", axis=1)
y_val = val["class_p"]

X_test = test.drop("class_p", axis=1)
y_test = test["class_p"]

tree_model.fit(X_train, y_train) # add training feature cols and class col to tree

#plot the confusion matrix
y_pred = tree_model.predict(X_test)
plot_conf_matrix(y_test, y_pred, "Confusion Matrix - Decision Tree")


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

#display decision tree metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


print("=== Decision Tree Metrics ===")
print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:   ", recall)
print("F1 Score: ", f1)
print("MSE:      ", mse)
print("MAE:      ", mae)

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
