#forestCover_GPU_1.py
# Celso Axelrud 
# 2019


# References:
# code: https://github.com/dmlc/xgboost/blob/master/demo/gpu_acceleration/cover_type.py
# http://archive.ics.uci.edu/ml/datasets/Covertype
# https://www.kaggle.com/uciml/forest-cover-type-dataset
# https://www.kaggle.com/c/forest-cover-type-prediction/leaderboard
# https://www.kaggle.com/uciml/forest-cover-type-dataset/kernels

# (*) https://www.kaggle.com/juzershakir/visualizing-predicting-type-of-forest-cover

# https://rstudio-pubs-static.s3.amazonaws.com/160297_f7bcb8d140b74bd19b758eb328344908.html
# https://shankarmsy.github.io/posts/forest-cover-types.html


# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/basic_walkthrough.py


# Classification Labels:
# Tree species: 
# Spruce/fir (type 1)
# Lodgepole pine (type 2)
# Ponderosa pine (type 3)
# Cottonwood/willow (type 4). 
# Aspen (type 5). 
# Douglas-fir (type 6)

# Classification Features: (Name / Data Type / Measurement / Description) 
# Elevation / quantitative /meters / Elevation in meters 
# Aspect / quantitative / azimuth / Aspect in degrees azimuth 
# Slope / quantitative / degrees / Slope in degrees 
# Horizontal_Distance_To_Hydrology / quantitative / meters / Horz Dist to nearest surface water features 
# Vertical_Distance_To_Hydrology / quantitative / meters / Vert Dist to nearest surface water features 
# Horizontal_Distance_To_Roadways / quantitative / meters / Horz Dist to nearest roadway 
# Hillshade_9am / quantitative / 0 to 255 index / Hillshade index at 9am, summer solstice 
# Hillshade_Noon / quantitative / 0 to 255 index / Hillshade index at noon, summer soltice 
# Hillshade_3pm / quantitative / 0 to 255 index / Hillshade index at 3pm, summer solstice 
# Horizontal_Distance_To_Fire_Points / quantitative / meters / Horz Dist to nearest wildfire ignition points 
# Wilderness_Area (4 binary columns) / qualitative / 0 (absence) or 1 (presence) / Wilderness area designation 
# Soil_Type (40 binary columns) / qualitative / 0 (absence) or 1 (presence) / Soil Type designation 
# Cover_Type (7 types) / integer / 1 to 7 / Forest Cover Type designation

# Datase Size: 581012

import os
import sys
os.path.dirname(sys.executable)

os.chdir("C:\\Users\\inter\\OneDrive\\Projects(Comp)\\Dev_2019\\XgBoost_2019")
os.getcwd()

import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time

# Fetch dataset using sklearn
cov = fetch_covtype()
X = cov.data
y = cov.target


# Create 0.75/0.25 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                    random_state=42)

# Specify sufficient boosting iterations to reach a minimum
num_round = 3000

# Leave most parameters as default
param = {'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 8, # Number of possible output classes
         'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
         }

# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

gpu_res = {} # Store accuracy result

# Train model
tmp = time.time()
bst1=xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))
#GPU Training Time: 784.006266117096 seconds
13/60
# 13 min
211/13
#16.23

# Prediction
preds = bst1.predict(dtest)
labels = dtest.get_label()
print('error=%f' % (sum(1 for i in range(len(preds)) if preds[i] == labels[i]) / float(len(preds))))
error=0.968565
# Kaggle competistion score rank: 7

bst1.save_model('CoverType_1.model')

# Repeat for CPU algorithm
tmp = time.time()
param['tree_method'] = 'hist'
cpu_res = {}
bst2=xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
#CPU Training Time: 2778.4804680347443 seconds
#1.487564659833857
2778/60
# 46.3 min
Kgpu=46.3/13
Kgpu
# 3.56

# Prediction
preds = bst2.predict(dtest)
labels = dtest.get_label()
print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
bst2.save_model('CoverType_2.model')
