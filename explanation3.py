import pandas as pd
import numpy as np
import sklearn
from six import StringIO
from warnings import warn


def pro_data(original_data, random_state=0):
    import numpy as np
    from sklearn.model_selection import train_test_split 
    from sklearn import metrics
    

    edu_mapping={"Bachelors":1,
                 "Masters":2,
                 "PHD":3}
    original_data["Edu"]=original_data["Education"].map(edu_mapping)

 
    city_mapping={"Bangalore":2,
                 "Pune":3,
                 "New Delhi":1}
    original_data["City_num"]=original_data["City"].map(city_mapping)

    gender_mapping={"Male":1,
                   "Female":0}
    everbenched_mapping={"No":0,
                        "Yes":1}
    original_data["Gender_num"]=original_data["Gender"].map(gender_mapping)
    original_data["EverBenched_num"]=original_data["EverBenched"].map(everbenched_mapping)

    pro_data=original_data[["PaymentTier","JoiningYear","Age","ExperienceInCurrentDomain","Edu","City_num","Gender_num","EverBenched_num","LeaveOrNot"]]
    pro_data = pro_data.astype(int)
    pro_data.rename(columns={"Edu":"Education",
                             "City_num":"City",
                             "Gender_num":"Gender",
                             "EverBenched_num":"EverBenched"},
                     inplace=True) 
    return pro_data

    

def data_process(original_data, random_state=0):
    import numpy as np
    from sklearn.model_selection import train_test_split 
    from sklearn import metrics
    
    
    X=original_data[["PaymentTier","JoiningYear","Age","ExperienceInCurrentDomain","Education","City","Gender","EverBenched"]]
    Y=original_data["LeaveOrNot"]
    X = X.astype(int)
    Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size=0.2,random_state=random_state)
    return Xtr, Xtest, Ytr, Ytest


def model_train(train,label,max_depth):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    dt = tree.DecisionTreeClassifier(criterion='entropy',max_depth=max_depth) 
    model=dt.fit(train,label)
    
    return model

def leaf_node(model):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    return is_leaves

def treeToJson_global(decision_tree,is_leaves,feature_names=None):
    
    from warnings import warn
    js = ""
    def node_to_str(tree, node_id, criterion):   
        if not isinstance(criterion, StringIO):
            criterion = "impurity"
        value = tree.tree_.value[node_id]
        if tree.n_outputs_ == 1:
            value = value[0, :]
        jsonValue = ', '.join([str(x) for x in value])
        if tree.tree_.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            leaf_name=" "
            if value[0]>value[1]:
                leaf_name="Not Leave"
            else:
                leaf_name="Leave"
            return '{"name":"%s","value": "%s"}' %(leaf_name,jsonValue)

        else:
            if feature_names is not None:
                feature = feature_names[tree.tree_.feature[node_id]]
            else:
                feature = tree.tree_.feature[node_id]
            if "=" in feature:
                ruleType = "="
                ruleValue = "false"
            else:
                ruleType = "<="
                ruleValue = "%.4f" % tree.tree_.threshold[node_id]
            return '"name": "%s %s %s"' % (feature,ruleType,ruleValue)
        
    def recurse(tree, node_id, criterion, parent=None, depth=0):
        tabs = "  " * depth
        js = ""
        left_child = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]
        if is_leaves[node_id]==True:
            js = js + "\n" + tabs  + node_to_str(tree, node_id, criterion)
        else:
            js = js + "\n" + tabs + "{\n" + tabs + "  " + node_to_str(tree, node_id, criterion)+','+'\n'+ '"children":['
        if left_child != sklearn.tree._tree.TREE_LEAF:

            js = js  + \
            tabs + \
            recurse(tree, \
                       left_child, \
                       criterion=criterion, \
                       parent=node_id, \
                       depth=depth + 1) + ",\n" + \
               tabs+ \
                recurse(tree, \
                   right_child, \
                   criterion=criterion, \
                   parent=node_id,
                   depth=depth + 1)
        if is_leaves[node_id]==True:
            js = js + tabs + "\n" 
        else:
            js = js + tabs + "\n" + tabs + "]}"
        
        return js
    if isinstance(decision_tree, sklearn.tree.BaseDecisionTree):
        js = js + recurse(decision_tree, 0, criterion="impurity")
    else:
        js = js + recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

    return js


def sample_path (model,test,n):
    sample=test.head(n)
    pred_label=model.predict(sample)
    node_indicator = model.decision_path(sample)
    leaf_id = model.apply(sample)
    path_node=[]
    for sample_id in range(n):
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]]
        path_node.append(node_index)
    
    return path_node,pred_label


def treeToJson_local(decision_tree,sample_node_path,is_leaves,feature_names=None):
    
    from warnings import warn
    js = ""
    def node_to_str(tree, node_id, criterion):   
        if not isinstance(criterion, StringIO):
            criterion = "impurity"
        value = tree.tree_.value[node_id]
        if tree.n_outputs_ == 1:
            value = value[0, :]
        jsonValue = ', '.join([str(x) for x in value])
        if tree.tree_.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            leaf_name=" "
            if value[0]>value[1]:
                leaf_name="Not Leave"
            else:
                leaf_name="Leave"
            if node_id in sample_node_path:
                return '{"name":"%s","value": "%s","lineStyle":{"color":"#0000ff","borderColor":"#0000ff"},"itemStyle":{"color":"#0000ff"}}' %(leaf_name,jsonValue)
            else:
                return '{"name":"%s","value": "%s"}' %(leaf_name,jsonValue)

        else:
            if feature_names is not None:
                feature = feature_names[tree.tree_.feature[node_id]]
            else:
                feature = tree.tree_.feature[node_id]
            if "=" in feature:
                ruleType = "="
                ruleValue = "false"
            else:
                ruleType = "<="
                ruleValue = "%.4f" % tree.tree_.threshold[node_id]
            if node_id in sample_node_path:
                return '"name": "%s %s %s","lineStyle":{"color":"#0000ff","borderColor":"#0000ff"},"itemStyle":{"color":"#0000ff"}' % (feature,ruleType,ruleValue)
            else:
                return '"name": "%s %s %s"' % (feature,ruleType,ruleValue)
        
    def recurse(tree, node_id, criterion, parent=None, depth=0):
        tabs = "  " * depth
        js = ""
        left_child = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]
        if is_leaves[node_id]==True:
            js = js + "\n" + tabs  + node_to_str(tree, node_id, criterion)
        else:
            js = js + "\n" + tabs + "{\n" + tabs + "  " + node_to_str(tree, node_id, criterion)+','+'\n'+ '"children":['
        if left_child != sklearn.tree._tree.TREE_LEAF:

            js = js  + \
            tabs + \
            recurse(tree, \
                       left_child, \
                       criterion=criterion, \
                       parent=node_id, \
                       depth=depth + 1) + ",\n" + \
               tabs+ \
                recurse(tree, \
                   right_child, \
                   criterion=criterion, \
                   parent=node_id,
                   depth=depth + 1)
        if is_leaves[node_id]==True:
            js = js + tabs + "\n" 
        else:
            js = js + tabs + "\n" + tabs + "]}"
        
        return js
    if isinstance(decision_tree, sklearn.tree.BaseDecisionTree):
        js = js + recurse(decision_tree, 0, criterion="impurity")
    else:
        js = js + recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

    return js


def conterfactual_explanation(decision_tree,initial,modified,is_leaves,feature_names=None):

    initial_path,pred_label_initial=sample_path(decision_tree,initial,1)
    modified_path,pred_label_modified=sample_path(decision_tree,modified,1)

    dif_path=np.array(list(set(modified_path[0])-set(initial_path[0])))
    
    from warnings import warn
    js = ""
    def node_to_str(tree, node_id, criterion):   
        if not isinstance(criterion, StringIO):
            criterion = "impurity"
        value = tree.tree_.value[node_id]
        if tree.n_outputs_ == 1:
            value = value[0, :]
        jsonValue = ', '.join([str(x) for x in value])
        if tree.tree_.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            leaf_name=" "
            if value[0]>value[1]:
                leaf_name="Not Leave"
            else:
                leaf_name="Leave"
                
            if node_id in initial_path[0]:
                return '{"name":"%s","value": "%s","lineStyle":{"color":"#0000ff","borderColor":"#0000ff"},"itemStyle":{"color":"#0000ff"}}' %(leaf_name,jsonValue)
            elif node_id in dif_path:
                    return '{"name":"%s","value": "%s","lineStyle":{"color":"green","borderColor":"green"},"itemStyle":{"color":"green"}}' %(leaf_name,jsonValue)
            else:
                return '{"name":"%s","value": "%s"}' %(leaf_name,jsonValue)

        else:
            if feature_names is not None:
                feature = feature_names[tree.tree_.feature[node_id]]
            else:
                feature = tree.tree_.feature[node_id]
            if "=" in feature:
                ruleType = "="
                ruleValue = "false"
            else:
                ruleType = "<="
                ruleValue = "%.4f" % tree.tree_.threshold[node_id]
                
            if node_id in initial_path[0]:
                return '"name": "%s %s %s","lineStyle":{"color":"#0000ff","borderColor":"#0000ff"},"itemStyle":{"color":"#0000ff"}' % (feature,ruleType,ruleValue)
            elif node_id in dif_path:
                return '"name": "%s %s %s","lineStyle":{"color":"#00A000","borderColor":"#00A000"},"itemStyle":{"color":"#00A000"}' % (feature,ruleType,ruleValue)
            else:
                return '"name": "%s %s %s"' % (feature,ruleType,ruleValue)
        
    def recurse(tree, node_id, criterion, parent=None, depth=0):
        tabs = "  " * depth
        js = ""
        left_child = tree.tree_.children_left[node_id]
        right_child = tree.tree_.children_right[node_id]
        if is_leaves[node_id]==True:
            js = js + "\n" + tabs  + node_to_str(tree, node_id, criterion)
        else:
            js = js + "\n" + tabs + "{\n" + tabs + "  " + node_to_str(tree, node_id, criterion)+','+'\n'+ '"children":['
        if left_child != sklearn.tree._tree.TREE_LEAF:

            js = js  + \
            tabs + \
            recurse(tree, \
                       left_child, \
                       criterion=criterion, \
                       parent=node_id, \
                       depth=depth + 1) + ",\n" + \
               tabs+ \
                recurse(tree, \
                   right_child, \
                   criterion=criterion, \
                   parent=node_id,
                   depth=depth + 1)
        if is_leaves[node_id]==True:
            js = js + tabs + "\n" 
        else:
            js = js + tabs + "\n" + tabs + "]}"
        
        return js
    if isinstance(decision_tree, sklearn.tree.BaseDecisionTree):
        js = js + recurse(decision_tree, 0, criterion="impurity")
    else:
        js = js + recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

    return js

def tree_visual (visual_pattern,decision_tree,is_leaves,feature_names=None,sample_node_path=None,initial=None,modified=None):

    model=decision_tree 
    is_leaves=is_leaves
    feature_names=feature_names
    vi_pattern=["Global_Explanation","Local_Explanation","Counterfactual_Explanation"]

    if visual_pattern==vi_pattern[0]:
        visual_data=treeToJson_global(model,is_leaves,feature_names)

    elif visual_pattern==vi_pattern[1]:
        if sample_node_path is not None:
            visual_data=treeToJson_local(model,sample_node_path,is_leaves,feature_names)

    elif visual_pattern==vi_pattern[2]:
        if initial is not None and modified is not None:
            visual_data=conterfactual_explanation(model,initial,modified,is_leaves,feature_names)
    

    import json
    json_data=[json.loads(visual_data)]
    
 
    return json_data
  