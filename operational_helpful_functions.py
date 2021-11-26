### Import libraries ###
# data manipulation
import numpy as np
import pandas as pd
from math import sqrt

# data visualization
import matplotlib.pyplot as plt

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# model evaluation
from sklearn.metrics import mean_squared_error

# open models
from joblib import load

def plot_products(df, num):
    products = df.groupby('product_group')['dispatches_SellIn'].sum().sort_values(ascending=False).index.tolist()
    for product in products[:num]:
        single_product = df.loc[df['product_group']==product, :].copy()
        
        fig = plt.figure(facecolor = 'white', figsize = (16, 6))
        fig.suptitle(product, fontsize=16)
        fig.subplots_adjust(hspace=0.15, wspace=0.15)
        spec = fig.add_gridspec(ncols=1, nrows=4, height_ratios=[1, 1, 1, 1])
        
        # Plot Sell In Orders
        ax = fig.add_subplot(spec[0])
        ax.get_xaxis().set_visible(False)
        ax.plot(single_product['PERIOD_TAG'], single_product['orders_SellIn'], label = 'Sell In - Orders',
                color='green')
        ax.set_xlim(df['PERIOD_TAG'].min(), df['PERIOD_TAG'].max())
        ax.set_ylim(0, np.max([single_product['dispatches_SellIn'].max(),
                               single_product['orders_SellIn'].max(), 
                               single_product['SellOut'].max()]))
        plt.legend(loc='upper left')
        
        # Plot Sell In Orders + Sell Out
        ax = fig.add_subplot(spec[1])
        ax.get_xaxis().set_visible(False)
        ax.plot(single_product['PERIOD_TAG'], single_product['orders_SellIn'], label = 'Sell In - Orders',
                color='green')
        ax.plot(single_product['PERIOD_TAG'], single_product['SellOut'], label = 'Sell Out',
                color='black')
        
        ax.set_xlim(df['PERIOD_TAG'].min(), df['PERIOD_TAG'].max())
        ax.set_ylim(0, np.max([single_product['dispatches_SellIn'].max(),
                               single_product['orders_SellIn'].max(), 
                               single_product['SellOut'].max()]))
        plt.legend(loc='upper left')
        
        # Plot Sell In Orders + Sell In Dispatches + Sell Out
        ax = fig.add_subplot(spec[2])
        ax.get_xaxis().set_visible(False)
        ax.plot(single_product['PERIOD_TAG'], single_product['dispatches_SellIn'], label = 'Sell In - Dispatches',
                color='red')
        ax.plot(single_product['PERIOD_TAG'], single_product['orders_SellIn'], label = 'Sell In - Orders',
                color='green')
        ax.plot(single_product['PERIOD_TAG'], single_product['SellOut'], label = 'Sell Out',
                color='black')
        
        ax.set_xlim(df['PERIOD_TAG'].min(), df['PERIOD_TAG'].max())
        ax.set_ylim(0, np.max([single_product['dispatches_SellIn'].max(),
                               single_product['orders_SellIn'].max(), 
                               single_product['SellOut'].max()]))
        plt.legend(loc='upper left')
        
        # Plot Sell Out + Promo
        ax = fig.add_subplot(spec[3])
        ax.plot(single_product['PERIOD_TAG'], single_product['SellOut'], label = 'Sell Out',
                color='black')
        ax.set_xlim(df['PERIOD_TAG'].min(), df['PERIOD_TAG'].max())
        ax.set_ylim(0, np.max([single_product['dispatches_SellIn'].max(),
                               single_product['orders_SellIn'].max(), 
                               single_product['SellOut'].max()]))
        plt.legend(loc='upper left')
        ax2=ax.twinx()
        ax2.plot(single_product['PERIOD_TAG'], single_product['numeric_distribution_selling_promotion'],
                label = 'Distribution promo',
                color='blue')
        ax2.plot(single_product['PERIOD_TAG'], single_product['numeric_distribution_selling_promotion_hyperparmarkets'],
                label = 'Distribution promo - hypermarkets',
                color='orange')
        ax2.scatter(single_product.loc[single_product['type_promo_1']>0, 'PERIOD_TAG'],
                    single_product.loc[single_product['type_promo_1']>0, 'type_promo_1']*25,
                    label='type_promo_1', color='brown', marker='o')
        ax2.scatter(single_product.loc[single_product['type_promo_2']>0, 'PERIOD_TAG'],
                    single_product.loc[single_product['type_promo_2']>0, 'type_promo_2']*50,
                    label='type_promo_1', color='brown', marker='s')
        ax2.set_ylim(0, 100)
        plt.legend(loc='upper right')
        plt.show()
        print('\n\n')



def make_forecast(selected_Forecast, horizon_Forecast, start, end, num, df, pipep_prep, target_columns):
    dataset_transformed = pipep_prep.fit_transform(df)
    X_transformed = dataset_transformed.drop(target_columns+['product_group', 'customer_name'], axis = 1).copy()
    for horizon_prediction in target_columns:
        model = load(f'models/model_horizon_{horizon_prediction}.pkl')
        X_transformed[f'Prediction_{horizon_prediction}'] = model.predict(X_transformed.fillna(0))
    dataset_transformed[f'Prediction_{selected_Forecast}_{horizon_Forecast}W'] = X_transformed[
                                                                f'Prediction_{selected_Forecast}_{horizon_Forecast}W']
    for customer_name in dataset_transformed.loc[:, 'customer_name'].unique():
        for product_group_name in dataset_transformed.loc[dataset_transformed['customer_name']==customer_name,
                                                          'product_group'].unique():
            filter_unique_prod = ((dataset_transformed['customer_name']==customer_name)&\
                                  (dataset_transformed['product_group']==product_group_name))
            dataset_transformed.loc[filter_unique_prod,
                f'Prediction_{selected_Forecast}_{horizon_Forecast}W'] = dataset_transformed.loc[filter_unique_prod,
                    f'Prediction_{selected_Forecast}_{horizon_Forecast}W'].shift(horizon_Forecast)
    products = dataset_transformed.groupby('product_group')['dispatches_SellIn'].sum().sort_values(ascending=False).index.tolist()
    for product in products[:num]:
        single_product = dataset_transformed.loc[((dataset_transformed['product_group']==product)&\
                                                  (dataset_transformed.index>=start)&\
                                                  (dataset_transformed.index<=end)), :].copy()
        plt.figure(figsize=(16,7))
        plt.title(product)
        plt.plot(single_product.index,
                 single_product.loc[:, selected_Forecast],
                 color='black', label = selected_Forecast)
        plt.plot(single_product.index,
                 single_product.loc[:, f'Prediction_{selected_Forecast}_{horizon_Forecast}W'],
                 color='green', label= f'Prediction_{selected_Forecast}_{horizon_Forecast}W')
        plt.legend(loc='upper left')
        plt.xlim([single_product.index.min(), single_product.index.max()])
        plt.ylim([0, np.max([single_product.loc[:, selected_Forecast],
                             single_product.loc[:, f'Prediction_{selected_Forecast}_{horizon_Forecast}W']])*1.05])
        plt.show()
        pass



def Initial_Feature_Selection(name, df, scenario=1):
    ### Variables from the past ###
    # SellOut
    sell_out_list = ['SellOut'] + [f'SellOut_lag_{lag}' for lag in range(1, 6)]
    sell_out_return_list = ['SellOut_return'] + [f'SellOut_return_lag_{lag}' for lag in range(1, 6)]
    sell_out_diff_list = ['SellOut_diff'] + [f'SellOut_diff_lag_{lag}' for lag in range(1, 6)]
    distribution_promotion_list = ['numeric_distribution_selling_promotion'] + [
        f'numeric_distribution_selling_promotion_lag_{lag}' for lag in range(1, 6)]
    # SellOut Rolling values
    SellOut_window_mean_list = [f'SellOut_window_mean_{lag}' for lag in range(2, 6)]
    SellOut_window_std_list = [f'SellOut_window_std_{lag}' for lag in range(2, 6)]
    SellOut_window_min_list = [f'SellOut_window_min_{lag}' for lag in range(2, 6)]
    SellOut_window_max_list = [f'SellOut_window_max_{lag}' for lag in range(2, 6)]
    SellOut_window_ewm_list = [f'SellOut_window_ewm_{lag}' for lag in range(2, 6)]
    SellOut_window_Bollinger_Upper_list = [f'SellOut_window_Bollinger_Upper_{lag}' for lag in range(2, 6)]
    SellOut_window_Bollinger_Lower_list = [f'SellOut_window_Bollinger_Lower_{lag}' for lag in range(2, 6)]
    
    SellOut_return_window_mean_list = [f'SellOut_return_window_mean_{lag}' for lag in range(2, 6)]
    SellOut_return_window_std_list = [f'SellOut_return_window_std_{lag}' for lag in range(2, 6)]
    SellOut_return_window_min_list = [f'SellOut_return_window_min_{lag}' for lag in range(2, 6)]
    SellOut_return_window_max_list = [f'SellOut_return_window_max_{lag}' for lag in range(2, 6)]
    SellOut_return_window_ewm_list = [f'SellOut_return_window_ewm_{lag}' for lag in range(2, 6)]
    SellOut_return_window_Bollinger_Upper_list = [f'SellOut_return_window_Bollinger_Upper_{lag}' for lag in range(2, 6)]
    SellOut_return_window_Bollinger_Lower_list = [f'SellOut_return_window_Bollinger_Lower_{lag}' for lag in range(2, 6)]
    
    SellOut_diff_window_mean_list = [f'SellOut_diff_window_mean_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_std_list = [f'SellOut_diff_window_std_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_min_list = [f'SellOut_diff_window_min_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_max_list = [f'SellOut_diff_window_max_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_ewm_list = [f'SellOut_diff_window_ewm_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_Bollinger_Upper_list = [f'SellOut_diff_window_Bollinger_Upper_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_Bollinger_Lower_list = [f'SellOut_diff_window_Bollinger_Lower_{lag}' for lag in range(2, 6)]
    
    
    # SellIn orders
    orders_SellIn_list = ['orders_SellIn'] + [f'orders_SellIn_lag_{lag}' for lag in range(1, 6)]
    orders_SellIn_return_list = ['orders_SellIn_return'] + [f'orders_SellIn_return_lag_{lag}' for lag in range(1, 6)]
    orders_SellIn_diff_list = ['orders_SellIn_diff'] + [f'orders_SellIn_diff_lag_{lag}' for lag in range(1, 6)]
    # SellIn dispatches
    dispatches_SellIn_list = ['dispatches_SellIn'] + [f'dispatches_SellIn_lag_{lag}' for lag in range(1, 6)]
    dispatches_SellIn_return_list = ['dispatches_SellIn_return'] + [f'dispatches_SellIn_return_lag_{lag}' for lag in range(1, 6)]
    dispatches_SellIn_diff_list = ['dispatches_SellIn_diff'] + [f'dispatches_SellIn_diff_lag_{lag}' for lag in range(1, 6)]
    # SellIn difference 
    difference_list = ['difference'] + [f'difference_lag_{lag}' for lag in range(1, 6)]
    # SellIn orders Rolling values
    orders_SellIn_window_mean_list = [f'orders_SellIn_window_mean_{lag}' for lag in range(2, 6)]
    orders_SellIn_window_std_list = [f'orders_SellIn_window_std_{lag}' for lag in range(2, 6)]
    
    orders_SellIn_return_window_mean_list = [f'orders_SellIn_return_window_mean_{lag}' for lag in range(2, 6)]
    orders_SellIn_return_window_std_list = [f'orders_SellIn_return_window_std_{lag}' for lag in range(2, 6)]
    
    orders_SellIn_diff_window_mean_list = [f'orders_SellIn_diff_window_mean_{lag}' for lag in range(2, 6)]
    orders_SellIn_diff_window_std_list = [f'orders_SellIn_diff_window_std_{lag}' for lag in range(2, 6)]
    # SellIn dispatches Rolling values
    dispatches_SellIn_window_mean_list = [f'dispatches_SellIn_window_mean_{lag}' for lag in range(2, 6)]
    dispatches_SellIn_window_std_list = [f'dispatches_SellIn_window_std_{lag}' for lag in range(2, 6)]
    
    dispatches_SellIn_return_window_mean_list = [f'dispatches_SellIn_return_window_mean_{lag}' for lag in range(2, 6)]
    dispatches_SellIn_return_window_std_list = [f'dispatches_SellIn_return_window_std_{lag}' for lag in range(2, 6)]
    
    dispatches_SellIn_diff_window_mean_list = [f'dispatches_SellIn_diff_window_mean_{lag}' for lag in range(2, 6)]
    dispatches_SellIn_diff_window_std_list = [f'dispatches_SellIn_diff_window_std_{lag}' for lag in range(2, 6)]
    # SellIn difference orders and dispatches Rolling values
    difference_window_sum_list = [f'difference_window_sum_{lag}' for lag in range(2, 6)]
    difference_return_window_sum_list = [f'difference_return_window_sum_{lag}' for lag in range(2, 6)]
    difference_diff_window_sum_list = [f'difference_diff_window_sum_{lag}' for lag in range(2, 6)]
    
    date_list = ['week_sin', 'week_cos']
    customers_list = [f'customer_name_{customer}' for customer in list(df['customer_name'].unique())]
    products_list = [f'product_group_{prod}' for prod in list(df['product_group'].unique())]
    
    if name[:7] == 'SellOut':
        nr_horizon = int(name[8:-1])
        if nr_horizon > 1:
            predictions_sellout_list = [f'Prediction_SellOut_{horizon}W' for horizon in range(1, nr_horizon)]
        elif nr_horizon == 1:
            predictions_sellout_list = []
        promo = [f'numeric_distribution_selling_promotion_{nr_horizon}W', f'numeric_distribution_selling_promotion_hyperparmarkets_{nr_horizon}W',
                 f'type_promo_1_{nr_horizon}W', f'type_promo_2_{nr_horizon}W']
        variables = predictions_sellout_list + promo + date_list + customers_list + products_list + sell_out_list +\
            sell_out_return_list + sell_out_diff_list + distribution_promotion_list + SellOut_window_mean_list +\
            SellOut_window_std_list + SellOut_window_min_list + SellOut_window_max_list + SellOut_window_ewm_list +\
            SellOut_window_Bollinger_Upper_list + SellOut_window_Bollinger_Lower_list +\
            SellOut_return_window_mean_list + SellOut_return_window_std_list + SellOut_return_window_min_list +\
            SellOut_return_window_max_list + SellOut_return_window_ewm_list +\
            SellOut_return_window_Bollinger_Upper_list + SellOut_return_window_Bollinger_Lower_list +\
            SellOut_diff_window_mean_list + SellOut_diff_window_std_list + SellOut_diff_window_min_list +\
            SellOut_diff_window_max_list + SellOut_diff_window_ewm_list + SellOut_diff_window_Bollinger_Upper_list +\
            SellOut_diff_window_Bollinger_Lower_list
        return variables
    elif name[:13] == 'orders_SellIn':
        nr_horizon = int(name[14:-1])
        if scenario==1:
            predictions_sellout_list = [f'Prediction_SellOut_{horizon}W' for horizon in range(nr_horizon+1, nr_horizon+5)]
        elif scenario==2:
            predictions_sellout_list = [f'numeric_distribution_selling_promotion_hyperparmarkets_{horizon}W' for horizon in range(nr_horizon+1,
                                                                                                    21)] +\
                                       [f'type_promo_1_{horizon}W' for horizon in range(nr_horizon+1,
                                                                                                    21)] +\
                                       [f'type_promo_2_{horizon}W' for horizon in range(nr_horizon+1,
                                                                                                    21)]
        if nr_horizon > 1:
            predictions_sellin_list = [f'Prediction_orders_SellIn_{horizon}W' for horizon in range(1, nr_horizon)]
        elif nr_horizon == 1:
            predictions_sellin_list = []
        variables = predictions_sellout_list + predictions_sellin_list + date_list + customers_list + products_list +\
            orders_SellIn_list + orders_SellIn_return_list + orders_SellIn_diff_list + dispatches_SellIn_list +\
            dispatches_SellIn_return_list + dispatches_SellIn_diff_list + difference_list +\
            orders_SellIn_window_mean_list + orders_SellIn_window_std_list + orders_SellIn_return_window_mean_list +\
            orders_SellIn_return_window_std_list + orders_SellIn_diff_window_mean_list +\
            orders_SellIn_diff_window_std_list + dispatches_SellIn_window_mean_list +\
            dispatches_SellIn_window_std_list + dispatches_SellIn_return_window_mean_list +\
            dispatches_SellIn_return_window_std_list + dispatches_SellIn_diff_window_mean_list +\
            dispatches_SellIn_diff_window_std_list + difference_window_sum_list + difference_return_window_sum_list +\
            difference_diff_window_sum_list
        return variables



def buskets():
    ### Variables from the past ###
    # SellOut
    sell_out_list = ['SellOut'] + [f'SellOut_lag_{lag}' for lag in range(1, 6)]
    sell_out_return_list = ['SellOut_return'] + [f'SellOut_return_lag_{lag}' for lag in range(1, 6)]
    sell_out_diff_list = ['SellOut_diff'] + [f'SellOut_diff_lag_{lag}' for lag in range(1, 6)]
    distribution_promotion_list = ['numeric_distribution_selling_promotion'] + [
        f'numeric_distribution_selling_promotion_lag_{lag}' for lag in range(1, 6)]
    # SellOut Rolling values
    SellOut_window_mean_list = [f'SellOut_window_mean_{lag}' for lag in range(2, 6)]
    SellOut_window_std_list = [f'SellOut_window_std_{lag}' for lag in range(2, 6)]
    SellOut_window_min_list = [f'SellOut_window_min_{lag}' for lag in range(2, 6)]
    SellOut_window_max_list = [f'SellOut_window_max_{lag}' for lag in range(2, 6)]
    SellOut_window_ewm_list = [f'SellOut_window_ewm_{lag}' for lag in range(2, 6)]
    SellOut_window_Bollinger_Upper_list = [f'SellOut_window_Bollinger_Upper_{lag}' for lag in range(2, 6)]
    SellOut_window_Bollinger_Lower_list = [f'SellOut_window_Bollinger_Lower_{lag}' for lag in range(2, 6)]
    
    SellOut_return_window_mean_list = [f'SellOut_return_window_mean_{lag}' for lag in range(2, 6)]
    SellOut_return_window_std_list = [f'SellOut_return_window_std_{lag}' for lag in range(2, 6)]
    SellOut_return_window_min_list = [f'SellOut_return_window_min_{lag}' for lag in range(2, 6)]
    SellOut_return_window_max_list = [f'SellOut_return_window_max_{lag}' for lag in range(2, 6)]
    SellOut_return_window_ewm_list = [f'SellOut_return_window_ewm_{lag}' for lag in range(2, 6)]
    SellOut_return_window_Bollinger_Upper_list = [f'SellOut_return_window_Bollinger_Upper_{lag}' for lag in range(2, 6)]
    SellOut_return_window_Bollinger_Lower_list = [f'SellOut_return_window_Bollinger_Lower_{lag}' for lag in range(2, 6)]
    
    SellOut_diff_window_mean_list = [f'SellOut_diff_window_mean_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_std_list = [f'SellOut_diff_window_std_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_min_list = [f'SellOut_diff_window_min_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_max_list = [f'SellOut_diff_window_max_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_ewm_list = [f'SellOut_diff_window_ewm_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_Bollinger_Upper_list = [f'SellOut_diff_window_Bollinger_Upper_{lag}' for lag in range(2, 6)]
    SellOut_diff_window_Bollinger_Lower_list = [f'SellOut_diff_window_Bollinger_Lower_{lag}' for lag in range(2, 6)]
    
    
    # SellIn orders
    orders_SellIn_list = ['orders_SellIn'] + [f'orders_SellIn_lag_{lag}' for lag in range(1, 6)]
    orders_SellIn_return_list = ['orders_SellIn_return'] + [f'orders_SellIn_return_lag_{lag}' for lag in range(1, 6)]
    orders_SellIn_diff_list = ['orders_SellIn_diff'] + [f'orders_SellIn_diff_lag_{lag}' for lag in range(1, 6)]
    # SellIn dispatches
    dispatches_SellIn_list = ['dispatches_SellIn'] + [f'dispatches_SellIn_lag_{lag}' for lag in range(1, 6)]
    dispatches_SellIn_return_list = ['dispatches_SellIn_return'] + [f'dispatches_SellIn_return_lag_{lag}' for lag in range(1, 6)]
    dispatches_SellIn_diff_list = ['dispatches_SellIn_diff'] + [f'dispatches_SellIn_diff_lag_{lag}' for lag in range(1, 6)]
    # SellIn difference 
    difference_list = ['difference'] + [f'difference_lag_{lag}' for lag in range(1, 6)]
    # SellIn orders Rolling values
    orders_SellIn_window_mean_list = [f'orders_SellIn_window_mean_{lag}' for lag in range(2, 6)]
    orders_SellIn_window_std_list = [f'orders_SellIn_window_std_{lag}' for lag in range(2, 6)]
    
    orders_SellIn_return_window_mean_list = [f'orders_SellIn_return_window_mean_{lag}' for lag in range(2, 6)]
    orders_SellIn_return_window_std_list = [f'orders_SellIn_return_window_std_{lag}' for lag in range(2, 6)]
    
    orders_SellIn_diff_window_mean_list = [f'orders_SellIn_diff_window_mean_{lag}' for lag in range(2, 6)]
    orders_SellIn_diff_window_std_list = [f'orders_SellIn_diff_window_std_{lag}' for lag in range(2, 6)]
    # SellIn dispatches Rolling values
    dispatches_SellIn_window_mean_list = [f'dispatches_SellIn_window_mean_{lag}' for lag in range(2, 6)]
    dispatches_SellIn_window_std_list = [f'dispatches_SellIn_window_std_{lag}' for lag in range(2, 6)]
    
    dispatches_SellIn_return_window_mean_list = [f'dispatches_SellIn_return_window_mean_{lag}' for lag in range(2, 6)]
    dispatches_SellIn_return_window_std_list = [f'dispatches_SellIn_return_window_std_{lag}' for lag in range(2, 6)]
    
    dispatches_SellIn_diff_window_mean_list = [f'dispatches_SellIn_diff_window_mean_{lag}' for lag in range(2, 6)]
    dispatches_SellIn_diff_window_std_list = [f'dispatches_SellIn_diff_window_std_{lag}' for lag in range(2, 6)]
    # SellIn difference orders and dispatches Rolling values
    difference_window_sum_list = [f'difference_window_sum_{lag}' for lag in range(2, 6)]
    difference_return_window_sum_list = [f'difference_return_window_sum_{lag}' for lag in range(2, 6)]
    difference_diff_window_sum_list = [f'difference_diff_window_sum_{lag}' for lag in range(2, 6)]
    
    nr_busket = 1
    buskets = {}
    lists_buskets = [SellOut_window_mean_list, SellOut_window_std_list, SellOut_window_min_list,
                     SellOut_window_max_list, SellOut_window_ewm_list, SellOut_window_Bollinger_Upper_list,
                     SellOut_window_Bollinger_Lower_list, SellOut_return_window_mean_list,
                     SellOut_return_window_std_list, SellOut_return_window_min_list, SellOut_return_window_max_list,
                     SellOut_return_window_ewm_list, SellOut_return_window_Bollinger_Upper_list,
                     SellOut_return_window_Bollinger_Lower_list, SellOut_diff_window_mean_list,
                     SellOut_diff_window_std_list, SellOut_diff_window_min_list, SellOut_diff_window_max_list,
                     SellOut_diff_window_ewm_list, SellOut_diff_window_Bollinger_Upper_list,
                     SellOut_diff_window_Bollinger_Lower_list, orders_SellIn_window_mean_list,
                     orders_SellIn_window_std_list, orders_SellIn_return_window_mean_list,
                     orders_SellIn_return_window_std_list, orders_SellIn_diff_window_mean_list,
                     orders_SellIn_diff_window_std_list, dispatches_SellIn_window_mean_list,
                     dispatches_SellIn_window_std_list, dispatches_SellIn_return_window_mean_list,
                     dispatches_SellIn_return_window_std_list, dispatches_SellIn_diff_window_mean_list,
                     dispatches_SellIn_diff_window_std_list, difference_window_sum_list,
                     difference_return_window_sum_list, difference_diff_window_sum_list]
    for busket in lists_buskets:
        for variable in busket:
            buskets[variable] = f'Busket_{nr_busket}'
        nr_busket += 1
    return buskets



def model_evaluation(model, Xtrain, ytrain, Xtest, ytest):
    def fit_scatter_plot(X, y, set_name):
        y_fitted_values = model.predict(X)
        xmin = y.min()
        xmax = y.max()
        plt.scatter(x = y_fitted_values, y = y, alpha=0.25)
        x_line = np.linspace(xmin, xmax, 10)
        y_line = x_line
        plt.plot(x_line, y_line, 'r--')
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel('Prediction')
        plt.ylabel('True Value')
        plt.title(f'Plot of predicted values versus true values - {set_name} set')
    
    def plot_of_residuals(X, y, set_name):
        errors = model.predict(X) - np.reshape(np.array(y), (-1))
        plt.scatter(x = y, y = errors, alpha=0.25)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel('True Value')
        plt.ylabel('Residual')
        plt.title(f'Plot of residuals - {set_name} set')
        
    def hist_of_residuals(X, y, set_name):
        errors = model.predict(X) - np.reshape(np.array(y), (-1))
        plt.hist(errors, bins = 100)
        plt.axvline(errors.mean(), color='k', linestyle='dashed', linewidth=1)
        plt.title(f'Histogram of residuals - {set_name} set')
    
    def DPA(y_true, y_pred):
        dpa = 100 - (((np.sum(np.abs(y_pred - y_true)))/(np.sum(y_true)))*100)
        return dpa

    def BIAS(y_true, y_pred):
        bias = (((np.sum(y_pred - y_true))/(np.sum(y_true)))*100)
        return bias
    
    fig = plt.figure(figsize = (16, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    ax = fig.add_subplot(3, 2, 1)
    fit_scatter_plot(X = Xtrain, y = ytrain, set_name = 'train')
    
    ax = fig.add_subplot(3, 2, 2)
    fit_scatter_plot(X = Xtest, y = ytest, set_name = 'test')
    
    ax = fig.add_subplot(3, 2, 3)
    plot_of_residuals(X = Xtrain, y = ytrain, set_name = 'train')
    
    ax = fig.add_subplot(3, 2, 4)
    plot_of_residuals(X = Xtest, y = ytest, set_name = 'test')    

    ax = fig.add_subplot(3, 2, 5)
    hist_of_residuals(X = Xtrain, y = ytrain, set_name = 'train')   

    ax = fig.add_subplot(3, 2, 6)
    hist_of_residuals(X = Xtest, y = ytest, set_name = 'test')
    
    plt.show()
    
    y_pred_train = model.predict(Xtrain)
    print(f'RMSE train: {sqrt(mean_squared_error(ytrain, y_pred_train))}')
    print(f'DPA  train: {DPA(ytrain, y_pred_train)}')
    print(f'BIAS train: {BIAS(ytrain, y_pred_train)}')
    print()
    y_pred_test = model.predict(Xtest)
    print(f'RMSE test:  {sqrt(mean_squared_error(ytest, y_pred_test))}')
    print(f'DPA  test: {DPA(ytest, y_pred_test)}')
    print(f'BIAS test: {BIAS(ytest, y_pred_test)}')
    
    # Feature Importance
    importance = model._final_estimator.feature_importances_
    df_feature_importance=importance.argsort()
    df_feature_importance=pd.DataFrame({
        'column':Xtrain.columns[df_feature_importance],
        'importance':importance[df_feature_importance]
    })
    df_feature_importance = df_feature_importance[df_feature_importance['importance']>=0.01].copy().reset_index(drop=True)
    plt.figure(figsize=(16, 4))
    plt.barh(df_feature_importance['column'][-10:], df_feature_importance['importance'][-10:])
    plt.tick_params(axis='both', labelsize=10)
    plt.title('Model Feature Importance', size=20)
    plt.xlabel(' ', size=15)
    plt.tight_layout()
    plt.show()



# Select Date as index
class MakeTSTransformer(BaseEstimator, TransformerMixin):
    def __init__( self, date_column):
        self.date_column = date_column    
    
    def fit( self, X, y = None ):
        return self    
    
    def transform( self, X, y = None ):
        X_transformed = X.copy()
        X_transformed = X_transformed.set_index(self.date_column)
        X_transformed.index = pd.to_datetime(X_transformed.index)
        return X_transformed



# Feature Engineering
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, promo_variables=['numeric_distribution_selling_promotion', 'numeric_distribution_selling_promotion_hyperparmarkets', 'type_promo_1', 'type_promo_2']):
        self.promo_variables = promo_variables
    
    def fit( self, X, y = None ):
        def dummies_columns(X, column):
            dummies = pd.get_dummies(X[column], prefix=column)
            return dummies.columns
        
        # customer_name
        self.customer_name_columns = dummies_columns(X = X, column = 'customer_name')
        
        # new_group
        self.product_group_columns = dummies_columns(X = X, column = 'product_group')
        
        return self 
    
    def transform( self, X, y = None ):
        
        def encode_time(X, col, max_val):
            X[col + '_sin'] = np.sin(2 * np.pi * X[col]/max_val)
            X[col + '_cos'] = np.cos(2 * np.pi * X[col]/max_val)
            X.drop(col, axis='columns', inplace=True)
            return X
        
        def create_dummies(X, column, dummies_columns):
            dummies = pd.get_dummies(X[column], prefix=column)
            dummies = dummies.reindex(columns = dummies_columns, fill_value=0)
            X = pd.concat([X, dummies], axis=1)
            return X
        
        X_transformed = X.copy()
        
        future_variables_list = []
        
        variables_from_future = self.promo_variables
        
        # Promo variables to remove
        promo_to_remove = list(set(X_transformed.loc[:, 'numeric_distribution_selling_promotion_hyperparmarkets':].columns))
        
        # for each customer group dataset using name of customer 
        for customer_name in X_transformed['customer_name'].unique():
            print(customer_name)
            for product_group in X_transformed.loc[X_transformed['customer_name']==customer_name, 'product_group'].unique():
                filter_prod_customer = ((X_transformed['product_group']==product_group)&\
                                        (X_transformed['customer_name']==customer_name))
                
                # Future output
                for horizon in range(1, 21):
                    # SellOut
                    X_transformed.loc[filter_prod_customer,
                        f'SellOut_{horizon}W'] = X_transformed.loc[filter_prod_customer,
                            f'SellOut'].shift(-horizon)
                    future_variables_list.append(f'SellOut_{horizon}W')

                    # SellIn Orders
                    if horizon <= 15:
                        X_transformed.loc[filter_prod_customer,
                            f'orders_SellIn_{horizon}W'] = X_transformed.loc[filter_prod_customer,
                                f'orders_SellIn'].shift(-horizon)
                        future_variables_list.append(f'orders_SellIn_{horizon}W')

                    # Promo
                    for variable in variables_from_future:
                        X_transformed.loc[filter_prod_customer,
                            f'{variable}_{horizon}W'] = X_transformed.loc[filter_prod_customer,
                                f'{variable}'].shift(-horizon)
                        future_variables_list.append(f'{variable}_{horizon}W')

                # Variables from the past
                # Sell Out
                X_transformed['SellOut'] = np.where(X_transformed['SellOut']<1,1,
                                                    X_transformed['SellOut'])
                X_transformed.loc[filter_prod_customer,
                    'SellOut_return'] = np.log(X_transformed.loc[filter_prod_customer,
                        'SellOut']/X_transformed.loc[filter_prod_customer, 'SellOut'].shift(1))
                X_transformed.loc[filter_prod_customer,
                    'SellOut_diff'] = np.log(X_transformed.loc[filter_prod_customer,
                        'SellOut'])-np.log(X_transformed.loc[filter_prod_customer, 'SellOut'].shift(1))
                
                # Orders
                X_transformed['orders_SellIn'] = np.where(X_transformed['orders_SellIn']<1,1,
                                                          X_transformed['orders_SellIn'])
                X_transformed.loc[filter_prod_customer,
                    f'orders_SellIn_return'] = np.log(X_transformed.loc[filter_prod_customer,
                        'orders_SellIn']/X_transformed.loc[filter_prod_customer, 'orders_SellIn'].shift(1))
                X_transformed.loc[filter_prod_customer,
                    f'orders_SellIn_diff'] = np.log(X_transformed.loc[filter_prod_customer,
                        'orders_SellIn'])-np.log(X_transformed.loc[filter_prod_customer, 'orders_SellIn'].shift(1))
                
                # Dispatches
                X_transformed['dispatches_SellIn'] = np.where(X_transformed['dispatches_SellIn']<1,1,
                                                              X_transformed['dispatches_SellIn'])
                X_transformed.loc[filter_prod_customer,
                    f'dispatches_SellIn_return'] = np.log(X_transformed.loc[filter_prod_customer,
                        'dispatches_SellIn']/X_transformed.loc[filter_prod_customer, 'dispatches_SellIn'].shift(1))
                X_transformed.loc[filter_prod_customer,
                    f'dispatches_SellIn_diff'] = np.log(X_transformed.loc[filter_prod_customer,
                        'dispatches_SellIn'])-np.log(X_transformed.loc[filter_prod_customer,
                                                                       'dispatches_SellIn'].shift(1))
                
                # SellIn difference orders and dispatches
                X_transformed.loc[filter_prod_customer,
                    'difference'] = X_transformed.loc[filter_prod_customer,
                        'orders_SellIn']-X_transformed.loc[filter_prod_customer, 'dispatches_SellIn']
                
                
                # lag values
                for lag in range(1, 6):
                    # SellOut
                    X_transformed.loc[filter_prod_customer,
                        f'SellOut_lag_{lag}'] = X_transformed.loc[filter_prod_customer,
                                                                  'SellOut'].shift(lag)
                    X_transformed.loc[filter_prod_customer,
                        f'SellOut_return_lag_{lag}'] = np.log(X_transformed.loc[filter_prod_customer,
                            'SellOut'].shift(lag)/X_transformed.loc[filter_prod_customer,
                                                                                'SellOut'].shift(lag+1))
                    X_transformed.loc[filter_prod_customer,
                        f'SellOut_diff_lag_{lag}'] = np.log(X_transformed.loc[filter_prod_customer,
                            'SellOut'].shift(lag))-np.log(X_transformed.loc[filter_prod_customer,
                                                                                'SellOut'].shift(lag+1))
                    # Promo Distribution
                    X_transformed.loc[filter_prod_customer,
                        f'numeric_distribution_selling_promotion_lag_{lag}'] = X_transformed.loc[filter_prod_customer,
                                                                'numeric_distribution_selling_promotion'].shift(lag)
                    # SellIn orders
                    X_transformed.loc[filter_prod_customer,
                        f'orders_SellIn_lag_{lag}'] = X_transformed.loc[filter_prod_customer,
                                                                        'orders_SellIn_diff'].shift(lag)
                    X_transformed.loc[filter_prod_customer,
                        f'orders_SellIn_return_lag_{lag}'] = np.log(X_transformed.loc[filter_prod_customer,
                            'orders_SellIn'].shift(lag)/X_transformed.loc[filter_prod_customer,
                                                                               'orders_SellIn'].shift(lag+1))
                    X_transformed.loc[filter_prod_customer,
                        f'orders_SellIn_diff_lag_{lag}'] = np.log(X_transformed.loc[filter_prod_customer,
                            'orders_SellIn'].shift(lag))-np.log(X_transformed.loc[filter_prod_customer,
                                                                               'orders_SellIn'].shift(lag+1))
                    # SellIn dispatches
                    X_transformed.loc[filter_prod_customer,
                        f'dispatches_SellIn_lag_{lag}'] = X_transformed.loc[filter_prod_customer,
                                                                            'dispatches_SellIn'].shift(lag)
                    X_transformed.loc[filter_prod_customer,
                        f'dispatches_SellIn_return_lag_{lag}'] = np.log(X_transformed.loc[filter_prod_customer,
                            'dispatches_SellIn'].shift(lag)/X_transformed.loc[filter_prod_customer,
                                                                              'dispatches_SellIn'].shift(lag+1))
                    X_transformed.loc[filter_prod_customer,
                        f'dispatches_SellIn_diff_lag_{lag}'] = np.log(X_transformed.loc[filter_prod_customer,
                            'dispatches_SellIn'].shift(lag))-np.log(X_transformed.loc[filter_prod_customer,
                                                                              'dispatches_SellIn'].shift(lag+1))
                    # SellIn difference orders and dispatches
                    X_transformed.loc[filter_prod_customer,
                        f'difference_lag_{lag}'] = X_transformed.loc[filter_prod_customer,
                            'orders_SellIn'].shift(lag)-X_transformed.loc[filter_prod_customer,
                                                                          'dispatches_SellIn'].shift(lag)
                    
                    # rolling values
                    if lag >=2:
                        # SaleOut
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut'].rolling(window=lag).std()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_window_min_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut'].rolling(window=lag).min()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_window_max_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut'].rolling(window=lag).max()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_window_ewm_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut'].ewm(span=lag, adjust=False).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_window_Bollinger_Upper_{lag}'] = X_transformed.loc[filter_prod_customer,
                                f'SellOut_window_mean_{lag}'] + (X_transformed.loc[filter_prod_customer,
                                    f'SellOut_window_std_{lag}'] * 2)
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_window_Bollinger_Lower_{lag}'] = X_transformed.loc[filter_prod_customer,
                                f'SellOut_window_mean_{lag}'] - (X_transformed.loc[filter_prod_customer,
                                    f'SellOut_window_std_{lag}'] * 2)
                        # SellOut Return
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_return_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_return'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_return_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_return'].rolling(window=lag).std()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_return_window_min_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_return'].rolling(window=lag).min()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_return_window_max_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_return'].rolling(window=lag).max()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_return_window_ewm_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_return'].ewm(span=lag, adjust=False).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_return_window_Bollinger_Upper_{lag}'] = X_transformed.loc[filter_prod_customer,
                                f'SellOut_return_window_mean_{lag}'] + (X_transformed.loc[filter_prod_customer,
                                    f'SellOut_return_window_std_{lag}'] * 2)
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_return_window_Bollinger_Lower_{lag}'] = X_transformed.loc[filter_prod_customer,
                                f'SellOut_return_window_mean_{lag}'] - (X_transformed.loc[filter_prod_customer,
                                    f'SellOut_return_window_std_{lag}'] * 2)
                        # SellOut Diff
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_diff_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_diff'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_diff_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_diff'].rolling(window=lag).std()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_diff_window_min_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_diff'].rolling(window=lag).min()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_diff_window_max_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_diff'].rolling(window=lag).max()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_diff_window_ewm_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'SellOut_diff'].ewm(span=lag, adjust=False).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_diff_window_Bollinger_Upper_{lag}'] = X_transformed.loc[filter_prod_customer,
                                f'SellOut_diff_window_mean_{lag}'] + (X_transformed.loc[filter_prod_customer,
                                    f'SellOut_diff_window_std_{lag}'] * 2)
                        X_transformed.loc[filter_prod_customer,
                            f'SellOut_diff_window_Bollinger_Lower_{lag}'] = X_transformed.loc[filter_prod_customer,
                                f'SellOut_diff_window_mean_{lag}'] - (X_transformed.loc[filter_prod_customer,
                                    f'SellOut_diff_window_std_{lag}'] * 2)
                        # SellIn orders
                        X_transformed.loc[filter_prod_customer,
                            f'orders_SellIn_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'orders_SellIn_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn'].rolling(window=lag).std()
                        # SellIn orders Return
                        X_transformed.loc[filter_prod_customer,
                            f'orders_SellIn_return_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn_return'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'orders_SellIn_return_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn_return'].rolling(window=lag).std()
                        # SellIn orders Diff
                        X_transformed.loc[filter_prod_customer,
                            f'orders_SellIn_diff_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn_diff'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'orders_SellIn_diff_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn_diff'].rolling(window=lag).std()
                        
                        # SellIn dispatches
                        X_transformed.loc[filter_prod_customer,
                            f'dispatches_SellIn_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'dispatches_SellIn'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'dispatches_SellIn_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'dispatches_SellIn'].rolling(window=lag).std()
                        # SellIn dispatches Return
                        X_transformed.loc[filter_prod_customer,
                            f'dispatches_SellIn_return_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'dispatches_SellIn_return'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'dispatches_SellIn_return_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'dispatches_SellIn_return'].rolling(window=lag).std()
                        # SellIn dispatches Diff
                        X_transformed.loc[filter_prod_customer,
                            f'dispatches_SellIn_diff_window_mean_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'dispatches_SellIn_diff'].rolling(window=lag).mean()
                        X_transformed.loc[filter_prod_customer,
                            f'dispatches_SellIn_diff_window_std_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'dispatches_SellIn_diff'].rolling(window=lag).std()
                        
                        # SellIn difference orders and dispatches
                        X_transformed.loc[filter_prod_customer,
                            f'difference_window_sum_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn'].rolling(window=lag).sum()-X_transformed.loc[filter_prod_customer,
                                    'dispatches_SellIn'].rolling(window=lag).sum()
                        # SellIn difference orders and dispatches
                        X_transformed.loc[filter_prod_customer,
                            f'difference_return_window_sum_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn_return'].rolling(window=lag).sum()-X_transformed.loc[filter_prod_customer,
                                    'dispatches_SellIn_return'].rolling(window=lag).sum()
                        # SellIn difference orders and dispatches Diff
                        X_transformed.loc[filter_prod_customer,
                            f'difference_diff_window_sum_{lag}'] = X_transformed.loc[filter_prod_customer,
                                'orders_SellIn_diff'].rolling(window=lag).sum()-X_transformed.loc[filter_prod_customer,
                                    'dispatches_SellIn_diff'].rolling(window=lag).sum()
        
        # date
        X_transformed['date'] = X_transformed.index
        X_transformed['week'] = X_transformed['date'].dt.isocalendar().week
        X_transformed = encode_time(X_transformed, 'week', 53)
        
        # create dummies
        # customer_name
        X_transformed = create_dummies(X = X_transformed, column = 'customer_name',
                                       dummies_columns = self.customer_name_columns)
        # Product_group
        X_transformed = create_dummies(X = X_transformed, column = 'product_group',
                                       dummies_columns = self.product_group_columns)
            
        # drop missing values and unnecessary columns
        future_variables_list = list(set(future_variables_list))
        X_transformed.dropna(subset=[ele for ele in list(X_transformed.columns) if ele not in future_variables_list],
                             inplace=True)
        X_transformed.drop(promo_to_remove, axis=1, inplace=True)
        X_transformed.drop(['date'], axis=1, inplace=True)
        

        return X_transformed



class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, buskets=buskets()):
        self.buskets = buskets
        pass
        
        
    def fit( self, X, y):
        buskets = self.buskets
        corr = pd.concat([X, y], axis=1).corr()[y.name]
        corr = corr.iloc[:-1]
        corr = np.abs(corr)
        corr = corr.sort_values(ascending=False)
        corr = corr[corr>0.2]
        corr = pd.DataFrame({'corr': corr})
        corr['Busket'] = corr.index.map(buskets)
        corr = pd.concat([corr.loc[corr['Busket'].isnull(), :],
                          corr.loc[~corr['Busket'].isnull(), :].drop_duplicates(subset='Busket', keep='first')],
                         axis=0)
        corr = corr.sort_values(by='corr', ascending=False)
        self.selected_columns = list(corr.index)
        return self 
    

    def transform( self, X, y = None ):
        df_features = X.copy()
        
        df_features = df_features.loc[:, self.selected_columns]
        
        return df_features