#1
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2
testing = pd.read_csv('testingdata2.csv')
training = pd.read_csv('trainingdata2.csv')
index_names = ['unit', 'cycle']
setting_names = ['temp','rh','pressure','ws','wd','ch','solrad']

# 3
def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit")
    max_cycle = grouped_by_unit["cycle"].max()
    
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit', right_index=True)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["cycle"]
    result_frame["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame
  
training = add_remaining_useful_life(training)
#training[index_names+['RUL']].head()

# This for converting dataframe to CSV
#training.to_csv('RULdata.csv')

# 4
settings_df = training[setting_names].copy()
# settings_df['setting_1'] = settings_df['setting_1'].round()
# settings_df['setting_2'] = settings_df['setting_2'].round(decimals=2)
settings_df.groupby(by=setting_names).size()

def plot_signal(df, signal_name, unit=None):
    plt.figure(figsize=(13,5))
    
    if unit:
        plt.plot('RUL', signal_name, 
                data=df[df['unit']==unit])
    else:
        for i in training['unit'].unique():
            if (i % 3 == 0):  # only ploting every 10th unit_nr
                plt.plot('RUL', signal_name, 
                         data=df[df['unit']==i])
    plt.xlim(35, 0)  # reverse the x-axis so RUL counts down to zero
    plt.ylabel(signal_name)
    plt.xlabel('Remaining Use fulLife')
    plt.show()

# specifically plotting setting name, I'm using this as an example throughout the rest of the analysis  
plot_signal(training, 'ws', unit=3)
# specifically plotting setting name, I'm using this as an example throughout the rest of the analysis  
plot_signal(training, 'rh', unit=3)

# 5
X_train = training[setting_names].copy()
y_train = training['RUL'].copy()
y_train_clipped = y_train.clip(upper=125)

# get last row of each engine
X_test = testing.drop('cycle', axis=1).groupby('unit').last().copy() 

# 6
def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))

lm = LinearRegression()
lm.fit(X_train, y_train)

#7
# predict and evaluate
y_hat_train = lm.predict(X_train)
evaluate(y_train, y_hat_train, 'train')

y_hat_test = lm.predict(X_test)
evaluate(y_test, y_hat_test)