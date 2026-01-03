
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

def get_ml_model(name):
    if name=="RandomForest":
        return RandomForestRegressor(n_estimators=300)
    if name=="XGBoost":
        return XGBRegressor(n_estimators=300)
    if name=="LightGBM":
        return LGBMRegressor(n_estimators=300)
    if name=="SVR":
        return SVR()
    if name=="LinearRegression":
        return LinearRegression()
