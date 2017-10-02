# _*_ coding: utf-8 _*_
# 预测北京房价
# Author: Kython Liao
# 2017.10.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, validation_curve, KFold, GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor


##################
# 第一步导入数据
##################
# 载入北京房屋的数据
df = pd.read_csv("bj_housing.csv")
prices = df["Value"]
features = df.drop("Value", axis=1)


##################
# 第二步预处理数据
##################
# 特征归一化
scaler = MinMaxScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
# 数据分割与重排
# 80%的数据用于训练，20%用于测试
X_train, X_test, y_train, y_test = train_test_split(features_scaled, prices, test_size=0.2, random_state=15)

##################
# 第三步模型衡量标准
##################
def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    score = r2_score(y_true, y_predict)
    return score

##################
# 第四步分析模型的表现
##################
# 根据不同的训练集大小，和最大深度，生成学习曲线
# def ModelLearning(X, y):
#     """ Calculates the performance of several models with varying sizes of training data.
#         The learning and validation scores for each model are then plotted. """
    
#     # Create 10 cross-validation sets for training and testing
#     cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)


#     # Generate the training set sizes increasing by 50
#     train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

#     # Create the figure window
#     fig = pl.figure(figsize=(10,7))

#     # Create three different models based on max_depth
#     for k, depth in enumerate([1,3,6,10]):
        
#         # Create a Decision tree regressor at max_depth = depth
#         regressor = DecisionTreeRegressor(max_depth = depth)

#         # Calculate the training and testing scores
#         sizes, train_scores, valid_scores = learning_curve(regressor, X, y, \
#             cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
#         # Find the mean and standard deviation for smoothing
#         train_std = np.std(train_scores, axis = 1)
#         train_mean = np.mean(train_scores, axis = 1)
#         valid_std = np.std(valid_scores, axis = 1)
#         valid_mean = np.mean(valid_scores, axis = 1)

#         # Subplot the learning curve 
#         ax = fig.add_subplot(2, 2, k+1)
#         ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
#         ax.plot(sizes, valid_mean, 'o-', color = 'g', label = 'Validation Score')
#         ax.fill_between(sizes, train_mean - train_std, \
#             train_mean + train_std, alpha = 0.15, color = 'r')
#         ax.fill_between(sizes, valid_mean - valid_std, \
#             valid_mean + valid_std, alpha = 0.15, color = 'g')
        
#         # Labels
#         ax.set_title('max_depth = %s'%(depth))
#         ax.set_xlabel('Number of Training Points')
#         ax.set_ylabel('r2_score')
#         ax.set_xlim([0, X.shape[0]*0.8])
#         ax.set_ylim([-0.05, 1.05])
    
#     # Visual aesthetics
#     ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
#     fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
#     fig.tight_layout()
#     fig.show()
#     fig.savefig('learning_curve_without_scaling.png', dpi=fig.dpi)

# ModelLearning(X_train, y_train)

# 根据不同的最大深度参数，生成复杂度曲线
# def ModelComplexity(X, y):
#     """ Calculates the performance of the model as model complexity increases.
#         The learning and validation errors rates are then plotted. """
    
#     # Create 10 cross-validation sets for training and testing
#     cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

#     # Vary the max_depth parameter from 1 to 10
#     max_depth = np.arange(1,11)

#     # Calculate the training and testing scores
#     train_scores, valid_scores = validation_curve(DecisionTreeRegressor(random_state = 15), X, y, \
#         param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

#     # Find the mean and standard deviation for smoothing
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#     valid_mean = np.mean(valid_scores, axis=1)
#     valid_std = np.std(valid_scores, axis=1)

#     # Plot the validation curve
#     fig = pl.figure(figsize=(7, 5))
#     pl.title('Decision Tree Regressor Complexity Performance')
#     pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
#     pl.plot(max_depth, valid_mean, 'o-', color = 'g', label = 'Validation Score')
#     pl.fill_between(max_depth, train_mean - train_std, \
#         train_mean + train_std, alpha = 0.15, color = 'r')
#     pl.fill_between(max_depth, valid_mean - valid_std, \
#         valid_mean + valid_std, alpha = 0.15, color = 'g')
    
#     # Visual aesthetics
#     pl.legend(loc = 'lower right')
#     pl.xlabel('Maximum Depth')
#     pl.ylabel('r2_score')
#     pl.ylim([-0.05,1.05])
#     pl.show()
#     fig.savefig('complexity_curve.png')

# ModelComplexity(X_train, y_train)


##################
# 第五步选择最优参数
##################
def fit_model(X, y):
    """ 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型"""
    # 交叉验证生成器
    cross_validator = KFold(n_splits=10, shuffle=True, random_state=15)
    # 决策树回归函数
    regressor = DecisionTreeRegressor(random_state=0)
    # 决策树待搜索参数空间
    params = {'max_depth': range(1,11)}
    # 评分函数
    scoring_fnc = make_scorer(performance_metric)
    # 网格搜索对象
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cross_validator)

    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(X, y)

    # 返回网格搜索后的最优模型
    return grid.best_estimator_

# 基于训练数据，获得最优模型
optimal_reg = fit_model(X_train, y_train)

# 输出最优模型的 'max_depth' 参数
print "max_depth: {} for the optimal model.".format(optimal_reg.get_params()['max_depth'])

##################
# 第六步做出预测
##################
# 用最优模型在整个测试数据上进行预测, 并计算相对于目标变量的决定系数R2的值
y_pred = optimal_reg.predict(X_test)
r2 = performance_metric(y_test, y_pred)

print "Optimal model has R^2 score {:,.2f} on test data".format(r2)

reg = DecisionTreeRegressor(max_depth=1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2 = performance_metric(y_test, y_pred)
print "Optimal model has R^2 score {:,.2f} on test data".format(r2)