import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

iris = load_iris()
x = iris.data
y = iris.target

x = pd.DataFrame(x,columns =  iris.feature_names[:])
y = pd.DataFrame(y,columns = ['Species']

#more depth cost to data overfitting

tree=DecisionTreeClassifier(max_depth=2)

tree.fit(x,y)

dot_data = export_graphviz(tree,class_names = iris.target_data,filled = True,rounded = True)

Graph = pydotplus.graph_from_dot_data(dot_data)

Image(Graph.create_png())


