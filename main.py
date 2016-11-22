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
import outliers_influence as ou

def rolling_coef(df):
    para_df = pd.DataFrame(columns=['ACWI', 'MSCI_World', 'Russell_3000', 'US_Dollar', 'Bond_Index', 'HFRI_World_Index', 'HFRI_Relative_Value', 'HFRI_Macro', 'HFRI_Macro_Total', 'HFRI_ED', 'HFRI_EH'])
    for i in range(0,len(df.index)-35):
        df_r = df.iloc[i:i+36,:]
        my_ols = sm.ols(formula='Kroger ~ ACWI + MSCI_World + Russell_3000 + US_Dollar + Bond_Index + HFRI_World_Index + HFRI_Relative_Value + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH - 1', data=df_r).fit()
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
    para_df = pd.DataFrame(columns=['ACWI_err', 'MSCI_World_err', 'Russell_3000_err', 'US_Dollar', 'Bond_Index','HFRI_World_Index_err','HFRI_Relative_Value_err','HFRI_Macro_err','HFRI_Macro_Total_err','HFRI_ED_err','HFRI_EH'])
    for i in range(0,len(df.index)-35):
        df_r = df.iloc[i:i+36,:]
        # gen orthogonal error series
        df_r['ACWI_err'] = get_error(df_r, 'ACWI ~ MSCI_World + Russell_3000 + US_Dollar + Bond_Index + HFRI_World_Index + HFRI_Relative_Value + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        df_r['MSCI_World_err'] = get_error(df_r, 'MSCI_World ~ ACWI_err + Russell_3000 + US_Dollar + Bond_Index + HFRI_World_Index + HFRI_Relative_Value + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        df_r['Russell_3000_err'] = get_error(df_r, 'Russell_3000 ~ ACWI_err + MSCI_World_err + US_Dollar + Bond_Index + HFRI_World_Index + HFRI_Relative_Value + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        df_r['HFRI_World_Index_err'] = get_error(df_r, 'HFRI_World_Index ~ ACWI_err + MSCI_World_err + Russell_3000_err + US_Dollar + Bond_Index + HFRI_Relative_Value + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        df_r['HFRI_Relative_Value_err'] = get_error(df_r, 'HFRI_Relative_Value ~ ACWI_err + MSCI_World_err + Russell_3000_err + US_Dollar + Bond_Index + HFRI_World_Index_err + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        df_r['HFRI_Macro_err'] = get_error(df_r, 'HFRI_Macro ~ ACWI_err + MSCI_World_err + Russell_3000_err + US_Dollar + Bond_Index + HFRI_World_Index_err + HFRI_Relative_Value_err + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        df_r['HFRI_Macro_Total_err'] = get_error(df_r, 'HFRI_Macro_Total ~ ACWI_err + MSCI_World_err + US_Dollar + Bond_Index + HFRI_World_Index_err + HFRI_Relative_Value_err + HFRI_Macro_err + HFRI_ED + HFRI_EH').values
        df_r['HFRI_ED_err'] = get_error(df_r, 'HFRI_ED ~ ACWI_err + MSCI_World_err + US_Dollar + Bond_Index + HFRI_World_Index_err + HFRI_Relative_Value_err + HFRI_Macro_err + HFRI_Macro_Total_err + HFRI_EH').values
        
    
        my_ols = sm.ols(formula='Kroger ~ ACWI_err + MSCI_World_err + Russell_3000_err + US_Dollar + Bond_Index + HFRI_World_Index_err + HFRI_Relative_Value_err + HFRI_Macro_err + HFRI_Macro_Total_err + HFRI_ED_err + HFRI_EH - 1', data=df_r).fit()
        # concat each rolling param
        s = pd.DataFrame(my_ols.params).T
        para_df = pd.concat([para_df,s], axis=0)
    # chean format of index
    para_df = para_df.reset_index()
    para_df = para_df.set_index(df.index[35:])
    para_df = para_df.drop('index', axis=1)
    return para_df
    
def rolling_v_coef(df):
    para_df = pd.DataFrame(columns=['ACWI_err', 'MSCI_World_err', 'Russell_3000', 'US_Dollar', 'Bond_Index','HFRI_World_Index','HFRI_Relative_Value', 'HFRI_Macro_Total','HFRI_ED','HFRI_EH'])
    for i in range(0,len(df.index)-35):
        df_r = df.iloc[i:i+36,:]
        # gen orthogonal error series
        df_r['ACWI_err'] = get_error(df_r, 'ACWI ~ MSCI_World + Russell_3000 + US_Dollar + Bond_Index + HFRI_World_Index + HFRI_Relative_Value + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        df_r['MSCI_World_err'] = get_error(df_r, 'MSCI_World ~ ACWI_err + Russell_3000 + US_Dollar + Bond_Index + HFRI_World_Index + HFRI_Relative_Value + HFRI_Macro + HFRI_Macro_Total + HFRI_ED + HFRI_EH').values
        
    
        my_ols = sm.ols(formula='Kroger ~ ACWI_err + MSCI_World_err + Russell_3000 + US_Dollar + Bond_Index + HFRI_World_Index + HFRI_Relative_Value + HFRI_Macro_Total + HFRI_ED + HFRI_EH - 1', data=df_r).fit()
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
    HFRI_World_Index = df['HFRI_World_Index'].values
    HFRI_Relative_Value = df['HFRI_Relative_Value'].values
    HFRI_Macro = df['HFRI_Macro'].values
    HFRI_Macro_Total = df['HFRI_Macro_Total'].values
    HFRI_ED = df['HFRI_ED'].values
    HFRI_EH = df['HFRI_EH'].values
    
    fig, ax = plt.subplots()
    plt.plot([],[], label='ACWI', color='m')
    plt.plot([],[], label='MSCI_World', color='c')
    plt.plot([],[], label='Russell_3000', color='r')
    plt.plot([],[], label='US_Dollar', color='k')
    plt.plot([],[], label='Bond_Index', color='b')
    plt.plot([],[], label='HFRI_World_Index', color='#E24A33')
    plt.plot([],[], label='HFRI_Relative_Value', color='#92C6FF')
    plt.plot([],[], label='HFRI_Macro', color='#0072B2')
    plt.plot([],[], label='HFRI_Macro_Total', color='#001C7F')
    plt.plot([],[], label='HFRI_ED', color='.15')
    plt.plot([],[], label='HFRI_EH', color='#30a2da')
    
    # Gen Stackplot
    plt.stackplot(date, ACWI, MSCI_World, Russell_3000, US_Dollar, Bond_Index, HFRI_World_Index, 
                  HFRI_Relative_Value, HFRI_Macro, HFRI_Macro_Total, HFRI_ED, HFRI_EH, 
                  colors=['m','c','r','k','b','#E24A33','#92C6FF','#0072B2','#001C7F','.15','#30a2da'])
    plt.xlabel('Date')
    plt.ylabel('Composition')
    plt.legend(loc='upper right', prop={'size':10})
    plt.title('Evolution of Factor Exposure - Classical Method')
    plt.show()

def plot_orth_stack(df):
    date = np.arange(66)
    ACWI = df['ACWI_err'].values
    MSCI_World = df['MSCI_World_err'].values
    Russell_3000 = df['Russell_3000_err'].values
    US_Dollar = df['US_Dollar'].values
    Bond_Index = df['Bond_Index'].values
    HFRI_World_Index = df['HFRI_World_Index_err'].values
    HFRI_Relative_Value = df['HFRI_Relative_Value_err'].values
    HFRI_Macro = df['HFRI_Macro_err'].values
    HFRI_Macro_Total = df['HFRI_Macro_Total_err'].values
    HFRI_ED = df['HFRI_ED_err'].values
    HFRI_EH = df['HFRI_EH'].values
    
    fig, ax = plt.subplots()
    plt.plot([],[], label='ACWI', color='m')
    plt.plot([],[], label='MSCI_World', color='c')
    plt.plot([],[], label='Russell_3000', color='r')
    plt.plot([],[], label='US_Dollar', color='k')
    plt.plot([],[], label='Bond_Index', color='b')
    plt.plot([],[], label='HFRI_World_Index', color='#E24A33')
    plt.plot([],[], label='HFRI_Relative_Value', color='#92C6FF')
    plt.plot([],[], label='HFRI_Macro', color='#0072B2')
    plt.plot([],[], label='HFRI_Macro_Total', color='#001C7F')
    plt.plot([],[], label='HFRI_ED', color='.15')
    plt.plot([],[], label='HFRI_EH', color='#30a2da')
    
    # Gen Stackplot
    plt.stackplot(date, ACWI, MSCI_World, Russell_3000, US_Dollar, Bond_Index, HFRI_World_Index, 
                  HFRI_Relative_Value, HFRI_Macro, HFRI_Macro_Total, HFRI_ED, HFRI_EH, 
                  colors=['m','c','r','k','b','#E24A33','#92C6FF','#0072B2','#001C7F','.15','#30a2da'])
    plt.xlabel('Date')
    plt.ylabel('Composition')
    plt.legend(loc='upper right', prop={'size':10})
    plt.title('Evolution of Factor Exposure - Orthogonal Method')
    plt.show()

    
    
def plot_v_stack(df):
    date = np.arange(66)
    ACWI = df['ACWI_err'].values
    MSCI_World = df['MSCI_World_err'].values
    Russell_3000 = df['Russell_3000'].values
    US_Dollar = df['US_Dollar'].values
    Bond_Index = df['Bond_Index'].values
    HFRI_World_Index = df['HFRI_World_Index'].values
    HFRI_Relative_Value = df['HFRI_Relative_Value'].values
    HFRI_Macro_Total = df['HFRI_Macro_Total'].values
    HFRI_ED = df['HFRI_ED'].values
    HFRI_EH = df['HFRI_EH'].values
    
    fig, ax = plt.subplots()
    plt.plot([],[], label='ACWI', color='m')
    plt.plot([],[], label='MSCI_World', color='c')
    plt.plot([],[], label='Russell_3000', color='r')
    plt.plot([],[], label='US_Dollar', color='k')
    plt.plot([],[], label='Bond_Index', color='b')
    plt.plot([],[], label='HFRI_World_Index', color='#E24A33')
    plt.plot([],[], label='HFRI_Relative_Value', color='#92C6FF')
    plt.plot([],[], label='HFRI_Macro_Total', color='#001C7F')
    plt.plot([],[], label='HFRI_ED', color='.15')
    plt.plot([],[], label='HFRI_EH', color='#30a2da')
    
    # Gen Stackplot
    plt.stackplot(date, ACWI, MSCI_World, Russell_3000, US_Dollar, Bond_Index, HFRI_World_Index, 
                  HFRI_Relative_Value, HFRI_Macro_Total, HFRI_ED, HFRI_EH, 
                  colors=['m','c','r','k','b','#E24A33','#92C6FF','#0072B2','#001C7F','.15','#30a2da'])
    plt.xlabel('Date')
    plt.ylabel('Composition')
    plt.legend(loc='upper right', prop={'size':10})
    plt.title('Evolution of Factor Exposure - VIF Adjusted Method')
    plt.show()
    
    
    
    
    
def cal_vif(df):
    df = df.drop('Kroger', axis=1)
    VIF_df = df.corr()
    for i in range(0,11):
        for j in range(0,11):
            df_ = df.iloc[:,(i,j)]
            VIF_df.iloc[i,j] = ou.variance_inflation_factor(df_.values,0)
    return VIF_df
    
    
if __name__ == '__main__':
    # file path
    os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\return and risk attribution project\\')
    file_name = 'data_f.xlsx'
    df = pd.read_excel(file_name)
    df.index = df['Period']
    df = df.drop('Period', axis=1)
    
    # Correlation 
    corr_df = df.corr()
    
    ### Calculate VIF Matrix
    VIF_df = cal_vif(df)
    
    
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
    # para_df.plot()
    # plt.title('Evolution of Coefficient - Classical Method')
    
    # Revise it into proportion
    para_por_df = para_df
    para_por_df = para_por_df.apply(lambda x: x/x.sum(), axis=1)
    
    para_por_abs_df = para_df.abs()
    para_por_abs_df = para_por_abs_df.apply(lambda x: x/x.sum(), axis=1)
    # plot
    # plot_stack(para_por_df)
    plot_stack(para_por_abs_df)
    
    
    
    ### Orthogonal resid method
    para_orth_df = rolling_orth_coef(df)
    # para_orth_df.plot()
    # plt.title('Evolution of Coefficient - Orthogonal Method')

    # Revise it into proportion
    para_por_orth_df = para_orth_df
    para_por_orth_df = para_por_orth_df.apply(lambda x: x/x.sum(), axis=1)
    
    para_por_orth_abs_df = para_orth_df.abs()
    para_por_orth_abs_df = para_por_orth_abs_df.apply(lambda x: x/x.sum(), axis=1)
    # plot
    # plot_orth_stack(para_por_orth_df)
    plot_orth_stack(para_por_orth_abs_df)  
    
    
    ### VIF adjusted method
    para_v_df = rolling_v_coef(df)
    # para_v_df.plot()
    # plt.title('Evolution of Coefficient - VIF adjusted Method')

    # Revise it into proportion
    para_por_v_df = para_v_df
    para_por_v_df = para_por_v_df.apply(lambda x: x/x.sum(), axis=1)
    
    para_por_v_abs_df = para_v_df.abs()
    para_por_v_abs_df = para_por_v_abs_df.apply(lambda x: x/x.sum(), axis=1)
    # plot
    # plot_v_stack(para_por_v_df)
    plot_v_stack(para_por_v_abs_df)      
    
    
    
    
'''
    my_ols = sm.ols(formula='HFRI_Macro ~ HFRI_Macro_Total', data=df).fit()
    VIF_l = []
    for i in range(0,12):
        VIF = ou.variance_inflation_factor(df.values,i)
        VIF_l.append(VIF)
'''
    
    

    
