# data manipulation
import numpy as np
import pandas as pd
from math import sqrt

# data visualization
import matplotlib.pyplot as plt

# pipeline
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# model evaluation
from sklearn.metrics import mean_squared_error

# plot products
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
                               single_product['Sellout'].max()]))
        plt.legend(loc='upper left')
        
        # Plot Sell In Orders + Sell Out
        ax = fig.add_subplot(spec[1])
        ax.get_xaxis().set_visible(False)
        ax.plot(single_product['PERIOD_TAG'], single_product['orders_SellIn'], label = 'Sell In - Orders',
                color='green')
        ax.plot(single_product['PERIOD_TAG'], single_product['Sellout'], label = 'Sell Out',
                color='black')
        
        ax.set_xlim(df['PERIOD_TAG'].min(), df['PERIOD_TAG'].max())
        ax.set_ylim(0, np.max([single_product['dispatches_SellIn'].max(),
                               single_product['orders_SellIn'].max(), 
                               single_product['Sellout'].max()]))
        plt.legend(loc='upper left')
        
        # Plot Sell In Orders + Sell In Dispatches + Sell Out
        ax = fig.add_subplot(spec[2])
        ax.get_xaxis().set_visible(False)
        ax.plot(single_product['PERIOD_TAG'], single_product['dispatches_SellIn'], label = 'Sell In - Dispatches',
                color='red')
        ax.plot(single_product['PERIOD_TAG'], single_product['orders_SellIn'], label = 'Sell In - Orders',
                color='green')
        ax.plot(single_product['PERIOD_TAG'], single_product['Sellout'], label = 'Sell Out',
                color='black')
        
        ax.set_xlim(df['PERIOD_TAG'].min(), df['PERIOD_TAG'].max())
        ax.set_ylim(0, np.max([single_product['dispatches_SellIn'].max(),
                               single_product['orders_SellIn'].max(), 
                               single_product['Sellout'].max()]))
        plt.legend(loc='upper left')
        
        # Plot Sell Out + Promo
        ax = fig.add_subplot(spec[3])
        ax.plot(single_product['PERIOD_TAG'], single_product['Sellout'], label = 'Sell Out',
                color='black')
        ax.set_xlim(df['PERIOD_TAG'].min(), df['PERIOD_TAG'].max())
        ax.set_ylim(0, np.max([single_product['dispatches_SellIn'].max(),
                               single_product['orders_SellIn'].max(), 
                               single_product['Sellout'].max()]))
        plt.legend(loc='upper left')
        ax2=ax.twinx()
        ax2.plot(single_product['PERIOD_TAG'], single_product['numeric_distribution_selling_promotion'],
                label = 'Distribution promo',
                color='blue')
        ax2.scatter(single_product.loc[single_product['type_promo_1']>0, 'PERIOD_TAG'],
                    single_product.loc[single_product['type_promo_1']>0, 'type_promo_1']*25,
                    label='type_promo_1', color='brown', marker='o')
        ax2.scatter(single_product.loc[single_product['type_promo_2']>0, 'PERIOD_TAG'],
                    single_product.loc[single_product['type_promo_2']>0, 'type_promo_2']*50,
                    label='type_promo_2', color='brown', marker='s')
        ax2.set_ylim(0, 100)
        plt.legend(loc='upper right')
        plt.savefig(f"prod_{product}.png", transparent=True, dpi=300)
        plt.show()
        print('\n\n')

# Select Date as index
class MakeTSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column    
    
    def fit(self, X, y = None):
        return self    
    
    def transform(self, X, y = None):
        X_transformed = X.copy()
        X_transformed = X_transformed.set_index(self.date_column)
        X_transformed.index = pd.to_datetime(X_transformed.index)
        return X_transformed


# Feature Engineering
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scenario, promo_variables=None, promo_sellin=None, cl4_col_name=None, mg1_col_name=None,
                 sellout_w=57, sellin_w = 52, nr_horizon=None, type_forecast=None, target=None, to_remove=True):
        self.scenario = scenario
        self.promo_variables = promo_variables
        self.promo_sellin = promo_sellin
        self.cl4_col_name = cl4_col_name
        self.mg1_col_name = mg1_col_name
        self.sellout_w = sellout_w
        self.sellin_w = sellin_w
        self.nr_horizon = nr_horizon
        self.type_forecast = type_forecast
        self.target = target
        self.to_remove = to_remove
    
    
    
    def fit(self, X, y = None):
        def dummies_columns(X, column):
            dummies = pd.get_dummies(X[column], prefix=column)
            return dummies.columns
        
        # customer_name
        if self.scenario != 5:
            if self.cl4_col_name != None:
                self.customer_name_columns = dummies_columns(X = X, column = self.cl4_col_name)
        
            print('Categoric features:')
            print(len(list(self.customer_name_columns)))
        
        ########################################################### SELECT VARIABLES ############################
        #################################################### Variables from the past #######################
        
        ##### lag values ##
        ### Sell Out
        sell_out_list = ['Sellout'] + [f'Sellout_lag_{lag}' for lag in range(1, 6)]
        ### Sell In 
        # Orders
        orders_SellIn_list = ['orders_SellIn'] + [f'orders_SellIn_lag_{lag}' for lag in range(1, 6)]
        # Dispatches
        dispatches_SellIn_list = ['dispatches_SellIn'] + [f'dispatches_SellIn_lag_{lag}' for lag in range(1, 6)]
        
        ##### rolling statistics ##
        ### Sell Out
        SellOut_window_mean_list = [f'Sellout_window_mean_{lag}' for lag in [5, 10]]
        SellOut_window_std_list = [f'Sellout_window_std_{lag}' for lag in range(2, 6)]
        SellOut_window_min_list = [f'Sellout_window_min_{lag}' for lag in range(2, 6)]
        SellOut_window_max_list = [f'Sellout_window_max_{lag}' for lag in range(2, 6)]
        ### Sell In 
        # Orders
        orders_SellIn_window_std_list = [f'orders_SellIn_window_std_{lag}' for lag in range(2, 6)]
        orders_SellIn_window_min_list = [f'orders_SellIn_window_min_{lag}' for lag in range(2, 6)]
        orders_SellIn_window_max_list = [f'orders_SellIn_window_max_{lag}' for lag in range(2, 6)]
        # Dispatches
        dispatches_SellIn_window_mean_list = [f'dispatches_SellIn_window_mean_{lag}' for lag in [5, 10]]
        dispatches_SellIn_window_std_list = [f'dispatches_SellIn_window_std_{lag}' for lag in range(2, 6)]
        dispatches_SellIn_window_min_list = [f'dispatches_SellIn_window_min_{lag}' for lag in range(2, 6)]
        dispatches_SellIn_window_max_list = [f'dispatches_SellIn_window_max_{lag}' for lag in range(2, 6)]
        # SellIn difference orders and dispatches
        SellIn_difference_list = [f'difference_window_sum_{lag}' for lag in range(2, 6)]
        
        ##################################################################### Date - Cyclic variable ##############
        date_list = ['week_sin', 'week_cos']
        
        ##################################################################### Categoric features ##############
        if self.scenario != 5:
            customers_list = [f'customer_name_{customer}' for customer in list(X['customer_name'].unique())]
        else:
            customers_list = []
        
        #####################################################################################################
        ##################################################################### Select variables ##############
        if self.scenario != 5:
            selected_variables = [f'{self.type_forecast}_{self.nr_horizon}W', 'customer_name', 'product_group'] +\
                                    date_list + customers_list
        else:
            selected_variables = [f'{self.type_forecast}_{self.nr_horizon}W', 'product_group'] +\
                                    date_list + customers_list
        
        ##### Sell Out ##
        if self.type_forecast == 'Sellout':
            if self.scenario == 1:
                future_or_predictions = [f'numeric_distribution_selling_promotion_{self.nr_horizon}W', f'type_promo_1_{self.nr_horizon}W',
                                         f'type_promo_2_{self.nr_horizon}W']
            elif self.scenario == 3:
                future_or_predictions = []
            
            if self.nr_horizon <= 3:
                selected_variables = selected_variables + future_or_predictions + sell_out_list +\
                    SellOut_window_mean_list + SellOut_window_std_list + SellOut_window_min_list +\
                    SellOut_window_max_list
            else:
                selected_variables = selected_variables + future_or_predictions +\
                    SellOut_window_mean_list + [f'Sellout_window_std_{10}', f'Sellout_window_min_{10}',
                                                f'Sellout_window_max_{10}']

        ##### Sell In ##
        elif self.type_forecast == 'orders_SellIn':
            if (self.scenario == 1)|(self.scenario == 3):
                future_or_predictions = [f'Prediction_Sellout_{horizon}W' for horizon in range(self.nr_horizon+1,
                                                                                               self.nr_horizon+5)]
            
            elif self.scenario == 2:
                future_or_predictions = [f'Distribution_{horizon}W' for horizon in range(self.nr_horizon+1,
                                                                                         self.nr_horizon+5)] +\
                                        [f'type_promo_1_{horizon}W' for horizon in range(self.nr_horizon+1,
                                                                               self.nr_horizon+5)] +\
                                        [f'type_promo_2_{horizon}W' for horizon in range(self.nr_horizon+1,
                                                                              self.nr_horizon+5)]
                
            
            if self.nr_horizon <= 3:
                selected_variables = selected_variables + future_or_predictions + orders_SellIn_list +\
                                    dispatches_SellIn_list + orders_SellIn_window_std_list +\
                                    orders_SellIn_window_min_list + orders_SellIn_window_max_list +\
                                    dispatches_SellIn_window_mean_list + dispatches_SellIn_window_std_list +\
                                    dispatches_SellIn_window_min_list + dispatches_SellIn_window_max_list
            else:
                selected_variables = selected_variables + future_or_predictions + dispatches_SellIn_window_mean_list +\
                    [f'orders_SellIn_window_std_{10}', f'orders_SellIn_window_min_{10}', 
                     f'orders_SellIn_window_max_{10}', f'dispatches_SellIn_window_std_{10}',
                     f'dispatches_SellIn_window_min_{10}', f'dispatches_SellIn_window_max_{10}',
                     f'difference_window_sum_{10}']
            
        print('Selected_variables:')
        print(selected_variables)
        print(f'Number: {len(selected_variables)}')
        self.selected_variables = selected_variables
        return self  
    
    
    
    def transform(self, X, y = None):
        
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
        
        # Promo variables to remove
        promo_to_remove = self.promo_variables+self.promo_sellin
        
        # Feature Engineering per cl4/mg1
        if self.scenario != 5:
            idxes = X_transformed[[self.cl4_col_name, self.mg1_col_name]].copy()
        else:
            idxes = X_transformed[[self.mg1_col_name]].copy()
        idxes.drop_duplicates(inplace=True)
        idxes.reset_index(drop=True, inplace=True)
        
        for idx in range(idxes.shape[0]):
            if self.scenario != 5:
                prod = ((X_transformed[[self.cl4_col_name, self.mg1_col_name]]==idxes.iloc[idx, :]).all(axis=1))
            else:
                prod = ((X_transformed[[self.mg1_col_name]]==idxes.iloc[idx, :]).all(axis=1))
            
            ##################################### Future ########################################################
            ################################# Future output ####
            ### SellOut
            X_transformed.loc[prod,
                              f'{self.type_forecast}_{self.nr_horizon}W'] = X_transformed.loc[prod,
                                                                                self.target].shift(-self.nr_horizon)
            future_variables_list.append(f'{self.type_forecast}_{self.nr_horizon}W')
            
            
            ### Promo Sell In
            if self.type_forecast == 'orders_SellIn':
                for variable in self.promo_sellin:
                    X_transformed.loc[prod,
                                      f'{variable}_{self.nr_horizon}W'] = X_transformed.loc[prod,
                                                                                f'{variable}'].shift(-self.nr_horizon)
                    future_variables_list.append(f'{variable}_{self.nr_horizon}W')
                    
                    
            if (self.scenario==1)|(self.scenario==2):
                ### Promo Sell Out
                if ((self.type_forecast == 'orders_SellIn')&(self.scenario==2)):
                    for horizon in range(self.nr_horizon+1, self.nr_horizon+5):
                        for variable in self.promo_variables:
                            X_transformed.loc[prod,
                                f'{variable}_{horizon}W'] = X_transformed.loc[prod,
                                    f'{variable}'].shift(-horizon)
                            future_variables_list.append(f'{variable}_{horizon}W')


                elif self.type_forecast == 'Sellout':
                    for variable in self.promo_variables:
                        X_transformed.loc[prod,
                            f'{variable}_{self.nr_horizon}W'] = X_transformed.loc[prod,
                                f'{variable}'].shift(-self.nr_horizon)
                        future_variables_list.append(f'{variable}_{self.nr_horizon}W')
                    
                
            #################################################### Variables from the past #######################
            for lag in range(1, 6):
                ##### lag values ##
                ### Sell Out
                if self.type_forecast == 'Sellout':
                    X_transformed.loc[prod,
                        f'Sellout_lag_{lag}'] = X_transformed.loc[prod,
                                                                  'Sellout'].shift(lag)
                ### Sell In
                if self.type_forecast == 'orders_SellIn':
                    # Orders
                    X_transformed.loc[prod,
                        f'orders_SellIn_lag_{lag}'] = X_transformed.loc[prod,
                                                                        'orders_SellIn'].shift(lag)
                    # Dispatches
                    X_transformed.loc[prod,
                        f'dispatches_SellIn_lag_{lag}'] = X_transformed.loc[prod,
                                                                            'dispatches_SellIn'].shift(lag)
                ##### rolling statistics ##
                if lag >= 2:
                    ### Sell Out
                    if self.type_forecast == 'Sellout':
                        X_transformed.loc[prod,
                            f'Sellout_window_std_{lag}'] = X_transformed.loc[prod,
                                'Sellout'].rolling(window=lag).std()
                        X_transformed.loc[prod,
                            f'Sellout_window_min_{lag}'] = X_transformed.loc[prod,
                                'Sellout'].rolling(window=lag).min()
                        X_transformed.loc[prod,
                            f'Sellout_window_max_{lag}'] = X_transformed.loc[prod,
                                'Sellout'].rolling(window=lag).max()
                    ### Sell In
                    if self.type_forecast == 'orders_SellIn':
                        # Orders
                        X_transformed.loc[prod,
                            f'orders_SellIn_window_std_{lag}'] = X_transformed.loc[prod,
                                'orders_SellIn'].rolling(window=lag).std()
                        X_transformed.loc[prod,
                            f'orders_SellIn_window_min_{lag}'] = X_transformed.loc[prod,
                                'orders_SellIn'].rolling(window=lag).min()
                        X_transformed.loc[prod,
                            f'orders_SellIn_window_max_{lag}'] = X_transformed.loc[prod,
                                'orders_SellIn'].rolling(window=lag).max()
                        # Dispatches
                        X_transformed.loc[prod,
                            f'dispatches_SellIn_window_std_{lag}'] = X_transformed.loc[prod,
                                'dispatches_SellIn'].rolling(window=lag).std()
                        X_transformed.loc[prod,
                            f'dispatches_SellIn_window_min_{lag}'] = X_transformed.loc[prod,
                                'dispatches_SellIn'].rolling(window=lag).min()
                        X_transformed.loc[prod,
                            f'dispatches_SellIn_window_max_{lag}'] = X_transformed.loc[prod,
                                'dispatches_SellIn'].rolling(window=lag).max()
                        # SellIn difference orders and dispatches
                        X_transformed.loc[prod,
                            f'difference_window_sum_{lag}'] = X_transformed.loc[prod,
                                'orders_SellIn'].rolling(window=lag).sum()-X_transformed.loc[prod,
                                    'dispatches_SellIn'].rolling(window=lag).sum()
                    
            #### Moving average for 5 and 10 shift weeks
            ### Sell Out
            if self.type_forecast == 'Sellout':
                X_transformed.loc[prod,
                    f'Sellout_window_mean_{5}'] = X_transformed.loc[prod,
                        'Sellout'].rolling(window=5).mean()
                X_transformed.loc[prod,
                    f'Sellout_window_mean_{10}'] = X_transformed.loc[prod,
                        'Sellout'].rolling(window=10).mean()
                
                if self.nr_horizon > 3:
                    X_transformed.loc[prod,
                        f'Sellout_window_std_{10}'] = X_transformed.loc[prod,
                            'Sellout'].rolling(window=10).std()
                    X_transformed.loc[prod,
                        f'Sellout_window_min_{10}'] = X_transformed.loc[prod,
                            'Sellout'].rolling(window=10).min()
                    X_transformed.loc[prod,
                        f'Sellout_window_max_{10}'] = X_transformed.loc[prod,
                            'Sellout'].rolling(window=10).max()
                
            
            ### Sell In
            if self.type_forecast == 'orders_SellIn':
                # Dispatches
                X_transformed.loc[prod,
                    f'dispatches_SellIn_window_mean_{5}'] = X_transformed.loc[prod,
                        'dispatches_SellIn'].rolling(window=5).mean()
                X_transformed.loc[prod,
                    f'dispatches_SellIn_window_mean_{10}'] = X_transformed.loc[prod,
                        'dispatches_SellIn'].rolling(window=10).mean()
                
                if self.nr_horizon > 3:
                    # Orders
                    X_transformed.loc[prod,
                        f'orders_SellIn_window_std_{10}'] = X_transformed.loc[prod,
                            'orders_SellIn'].rolling(window=10).std()
                    X_transformed.loc[prod,
                        f'orders_SellIn_window_min_{10}'] = X_transformed.loc[prod,
                            'orders_SellIn'].rolling(window=10).min()
                    X_transformed.loc[prod,
                        f'orders_SellIn_window_max_{10}'] = X_transformed.loc[prod,
                            'orders_SellIn'].rolling(window=10).max()
                    # Dispatches
                    X_transformed.loc[prod,
                        f'dispatches_SellIn_window_std_{10}'] = X_transformed.loc[prod,
                            'dispatches_SellIn'].rolling(window=10).std()
                    X_transformed.loc[prod,
                        f'dispatches_SellIn_window_min_{10}'] = X_transformed.loc[prod,
                            'dispatches_SellIn'].rolling(window=10).min()
                    X_transformed.loc[prod,
                        f'dispatches_SellIn_window_max_{10}'] = X_transformed.loc[prod,
                            'dispatches_SellIn'].rolling(window=10).max()
                    # SellIn difference orders and dispatches
                    X_transformed.loc[prod,
                        f'difference_window_sum_{10}'] = X_transformed.loc[prod,
                            'orders_SellIn'].rolling(window=10).sum()-X_transformed.loc[prod,
                                'dispatches_SellIn'].rolling(window=10).sum()
                
        
        ##################################################################### Date - Cyclic variable ##############
        X_transformed['date'] = X_transformed.index
        X_transformed['week'] = X_transformed['date'].dt.isocalendar().week
        X_transformed = encode_time(X_transformed, 'week', 53)
        
        ##################################################################### Categoric features ##############
        # CL4
        if self.scenario != 5:
            X_transformed = create_dummies(X = X_transformed, column = self.cl4_col_name,
                                           dummies_columns = self.customer_name_columns)
        
        ############################################### Drop missing values and unnecessary columns ##############
        future_variables_list = list(set(future_variables_list))
        X_transformed.dropna(subset=[ele for ele in list(X_transformed.columns) if ele not in future_variables_list],
                             inplace=True)
        
        if (self.scenario==1)|(self.scenario==2):
            X_transformed.drop(promo_to_remove, axis=1, inplace=True)
        X_transformed.drop(['date'], axis=1, inplace=True)
        
        ############################################## Select columns ###########################################
        X_transformed = X_transformed[self.selected_variables]
        if self.to_remove==True:
            X_transformed.dropna(inplace=True) # remove rows with Nans, because we do not have info for the future
        
        
        return X_transformed


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
    df_feature_importance = df_feature_importance[
                                            df_feature_importance['importance']>=0.01].copy().reset_index(drop=True)
    plt.figure(figsize=(16, 4))
    plt.barh(df_feature_importance['column'][-10:], df_feature_importance['importance'][-10:])
    plt.tick_params(axis='both', labelsize=10)
    plt.title('Model Feature Importance', size=20)
    plt.xlabel(' ', size=15)
    plt.tight_layout()
    plt.show()


def plot_true_pred(df, big_prod, number, target, type_pred, horizon, scenario):
    for idx in range(big_prod.shape[0])[:number]:
        prod = ((df[['customer_name', 'product_group']]==big_prod.iloc[idx, :2]).all(axis=1))
        fig = plt.figure(figsize=(16, 2))
        ax1 = fig.add_subplot(111)
        ax1.plot(df.loc[prod, 'PERIOD_TAG'], df.loc[prod, target],
                 color='green', label=type_pred, linewidth=3)
        ax1.plot(df.loc[prod, 'PERIOD_TAG'].shift(-horizon), 
                 df.loc[prod, f'Prediction_{type_pred}_{horizon}W'], 
                 color='black', label=f'{type_pred} -{horizon}W', linewidth=3)
        ### Add promo if Scenario 1 or 3 ##
        if (scenario==1)|(scenario==2):
            ax1.scatter(df.loc[(prod)&(df['type_promo_1']==1), 'PERIOD_TAG'], 
                        df.loc[(prod)&(df['type_promo_1']==1), target],
                        label='type_promo_1', alpha=0.5, s=100)
            ax1.scatter(df.loc[(prod)&(df['type_promo_2']==1), 'PERIOD_TAG'], 
                        df.loc[(prod)&(df['type_promo_2']==1), target],
                        label='type_promo_2', alpha=0.5, s=100)
            ax2 = ax1.twinx()
            ax2.plot(df.loc[prod, 'PERIOD_TAG'], df.loc[prod, 'numeric_distribution_selling_promotion'], 
                     color='blue', label='Promo Distribution')
            ax2.set_ylim([0, 100])
            ax2.legend(loc='upper right')

        ax1.set_xlim([df.loc[((prod)&(df['PERIOD_TAG']>='2019-01-01')), 'PERIOD_TAG'].min(),
                      df.loc[((prod)&(df['PERIOD_TAG']<'2020-01-01')), 'PERIOD_TAG'].max()])
        ax1.set_ylim(ymin=0)
        ax1.legend(loc='upper left')
        plt.title(f'CL4: {big_prod.iloc[idx, 0]}, MG1: {big_prod.iloc[idx, 1]}')
        plt.show()