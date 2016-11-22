# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:00:12 2016

@author: ZFang
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np

def rolling_coef(df):
    para_df = pd.DataFrame(columns=['ACWI', 'MSCI_World', 'Russell_3000', 'US_Dollar', 'Bond_Index'])
    for i in range(0,len(df.index)-35):
        df_r = df.iloc[i:i+36,:]
        my_ols = sm.ols(formula='Kroger ~ ACWI + MSCI_World + Russell_3000 + US_Dollar + Bond_Index - 1', data=df_r).fit()
        # concat each rolling param
        s = pd.DataFrame(my_ols.params).T
        para_df = pd.concat([para_df,s], axis=0)
    # chean format of index
    para_df = para_df.reset_index()
    para_df = para_df.set_index(df.index[35:])
    para_df = para_df.drop('index', axis=1)
    return para_df

    
def get_error(df, reg):
    my_ols = sm.ols(formula=reg, data=df).fit()
    resid = my_ols.resid
    return resid
    
    
    
def rolling_orth_coef(df):
    para_df = pd.DataFrame(columns=['ACWI', 'MSCI_World_err', 'Russell_3000_err', 'US_Dollar_err', 'Bond_Index_err'])
    for i in range(0,len(df.index)-35):
        df_r = df.iloc[i:i+36,:]
        # gen orthogonal error series
        df_r['MSCI_World_err'] = get_error(df_r, 'MSCI_World ~ ACWI + Russell_3000 + US_Dollar + Bond_Index').values
        df_r['Russell_3000_err'] = get_error(df_r, 'Russell_3000 ~ ACWI + MSCI_World_err + US_Dollar + Bond_Index').values
        df_r['US_Dollar_err'] = get_error(df_r, 'US_Dollar ~ ACWI + MSCI_World_err + Russell_3000_err + Bond_Index').values
        df_r['Bond_Index_err'] = get_error(df_r, 'Bond_Index ~ ACWI + MSCI_World_err + Russell_3000_err + US_Dollar_err').values
    
        my_ols = sm.ols(formula='Kroger ~ ACWI + MSCI_World_err + Russell_3000_err + US_Dollar_err + Bond_Index_err - 1', data=df_r).fit()
        # concat each rolling param
        s = pd.DataFrame(my_ols.params).T
        para_df = pd.concat([para_df,s], axis=0)
    # chean format of index
    para_df = para_df.reset_index()
    para_df = para_df.set_index(df.index[35:])
    para_df = para_df.drop('index', axis=1)
    return para_df
    
    
def plot_stack(df):
    date = np.arange(66)
    ACWI = df['ACWI'].values
    MSCI_World = df['MSCI_World'].values
    Russell_3000 = df['Russell_3000'].values
    US_Dollar = df['US_Dollar'].values
    Bond_Index = df['Bond_Index'].values
    
    fig, ax = plt.subplots()
    plt.plot([],[], label='ACWI', color='m')
    plt.plot([],[], label='MSCI_World', color='c')
    plt.plot([],[], label='Russell_3000', color='r')
    plt.plot([],[], label='US_Dollar', color='k')
    plt.plot([],[], label='Bond_Index', color='b')
    
    # Gen Stackplot
    plt.stackplot(date, ACWI, MSCI_World, Russell_3000, US_Dollar, Bond_Index, colors=['m','c','r','k','b'])
    plt.xlabel('Date')
    plt.ylabel('Composition')
    plt.legend(loc='upper right', prop={'size':10})
    plt.title('Evolution of Factor Exposure - Classical Method')
    plt.show()

def plot_orth_stack(df):
    date = np.arange(66)
    ACWI = df['ACWI'].values
    MSCI_World = df['MSCI_World_err'].values
    Russell_3000 = df['Russell_3000_err'].values
    US_Dollar = df['US_Dollar_err'].values
    Bond_Index = df['Bond_Index_err'].values
    
    fig, ax = plt.subplots()
    plt.plot([],[], label='ACWI', color='m')
    plt.plot([],[], label='MSCI_World', color='c')
    plt.plot([],[], label='Russell_3000', color='r')
    plt.plot([],[], label='US_Dollar', color='k')
    plt.plot([],[], label='Bond_Index', color='b')
    
    # Gen Stackplot
    plt.stackplot(date, ACWI, MSCI_World, Russell_3000, US_Dollar, Bond_Index, colors=['m','c','r','k','b'])
    plt.xlabel('Date')
    plt.ylabel('Composition')
    plt.legend(loc='upper right', prop={'size':10})
    plt.title('Evolution of Factor Exposure - Orthogonal Method')
    plt.show()

    
    
if __name__ == '__main__':
    # file path
    os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\return and risk attribution project\\')
    file_name = 'data_f.xlsx'
    df = pd.read_excel(file_name)
    df.index = df['Period']
    df = df.drop('Period', axis=1)
    
    # Correlation 
    corr_df = df.corr()
    
    # Rolling 36 months Correlation
    r_36_df = df.rolling(window=36).corr()
    corr_36_df = r_36_df[df.index[36:]].values
    corr_36_Kroger_df = r_36_df.iloc[:,5].T
    corr_36_Kroger_df = corr_36_Kroger_df.iloc[35:]

    # Plot graph
    plt.style.use('fivethirtyeight')
    corr_36_Kroger_df.plot()
    plt.title('36 Months Rolling Correlation')
    
    ### classical multi variable regerssion
    para_df = rolling_coef(df)
    para_df.plot()
    plt.title('Evolution of Coefficient - Classical Method')
    
    # Revise it into proportion
    para_por_df = para_df
    para_por_df = para_por_df.apply(lambda x: x/x.sum(), axis=1)
    
    para_por_abs_df = para_df.abs()
    para_por_abs_df = para_por_abs_df.apply(lambda x: x/x.sum(), axis=1)
    # plot
    plot_stack(para_por_df)
    plot_stack(para_por_abs_df)
    
    
    
    ### Orthogonal resid method
    para_orth_df = rolling_orth_coef(df)
    para_orth_df.plot()
    plt.title('Evolution of Coefficient - Orthogonal Method')

    # Revise it into proportion
    para_por_orth_df = para_orth_df
    para_por_orth_df = para_por_orth_df.apply(lambda x: x/x.sum(), axis=1)
    
    para_por_orth_abs_df = para_orth_df.abs()
    para_por_orth_abs_df = para_por_orth_abs_df.apply(lambda x: x/x.sum(), axis=1)
    # plot
    plot_orth_stack(para_por_orth_df)
    plot_orth_stack(para_por_orth_abs_df)  

    
    
    

    
    