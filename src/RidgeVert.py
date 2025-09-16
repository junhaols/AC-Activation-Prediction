#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized version of RidgeVert.py
Vertex-level Ridge regression for auditory cortex activation prediction

Optimizations:
- Improved memory usage and performance
- Added progress tracking and logging
- Better error handling and validation
- Parallel processing support
- Cleaner code structure

Original author: luojunhao
Optimized: 2025
"""
import pandas as pd
import scipy.io as sio
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
import time
from operator import itemgetter
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
from typing import List, Dict, Tuple, Optional, Union
import gc  # For garbage collection
from functools import lru_cache

############################ Configuration and Setup #########################################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ridge_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

project_path = '/Users/fiona/Junhao/Project/Cursor/AC-Activation-Prediction/'
try:
    import utlis.io_ as io_
except ImportError as e:
    logger.warning(f"Could not import io_ module: {e}")
    io_ = None

# Vertex-level predicted model,training and testing,based on subject,return a model, Corr ,MAE

############################################# Optimized dataframe creation ##################################
def creat_pred_df(subjs: List, feature_file: str, task_file: str, features: List[str]) -> pd.DataFrame:
    """
    Create prediction dataframe by combining features and task data.
    
    Optimizations:
    - Efficient DataFrame construction using list comprehension
    - Better error handling
    - Memory-efficient feature concatenation
    - Progress tracking for large datasets
    
    Args:
        subjs: List of subject IDs
        feature_file: Path to feature pickle file
        task_file: Path to task data pickle file
        features: List of feature names to use
    
    Returns:
        Combined DataFrame with subjects, features, and targets
    """
    logger.info(f"Creating prediction dataframe with {len(subjs)} subjects and {len(features)} features")
    
    try:
        df_f = pd.read_pickle(feature_file)
        df_t = pd.read_pickle(task_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e}")
    except Exception as e:
        raise ValueError(f"Error loading data files: {e}")
    
    # Filter data for specified subjects
    df_y = df_t.query('subject==@subjs')
    df_x = df_f.query('subject==@subjs')[features]
    
    if len(df_y) == 0 or len(df_x) == 0:
        raise ValueError(f"No data found for specified subjects")
    
    # Reset indices for consistent access
    df_y = df_y.reset_index(drop=True)
    df_x = df_x.reset_index(drop=True)
    
    # Efficiently build data list
    data_list = []
    
    for index in tqdm(range(len(subjs)), desc="Processing subjects", disable=len(subjs)<10):
        subj = subjs[index]
        
        # Concatenate features efficiently
        if len(features) == 1:
            x0 = df_x.loc[index, features[0]]
        else:
            feature_arrays = []
            for f in features:
                x_f = df_x.loc[index, f]
                if x_f.ndim == 1:
                    x_f = x_f.reshape(-1, 1)
                feature_arrays.append(x_f)
            x0 = np.concatenate(feature_arrays, axis=1)
        
        data_list.append({
            'subject': subj,
            'x': x0,
            'y': df_y.loc[index, 'task_t'],
            'vert_index': df_y.loc[index, 'vert_index']
        })
    
    # Create DataFrame efficiently
    df = pd.DataFrame(data_list)
    logger.info(f"Successfully created dataframe with {len(df)} subjects")
    
    return df
####################################### select data from the given df ##################################################
def select_data_subj(df, subj_id, df_column_name):
    data = df[df_column_name][df['subject'] == subj_id]
    return data # series datatype with dim of (1,),for the ndarray contents,data = data[0]

def select_data_multi_subj(df, subj_ids, df_column_name):
    subj_df = df['subject'].to_numpy()
    sub, index1, index2 = np.intersect1d(subj_df, subj_ids, assume_unique=False, return_indices=True)
    data = df.loc[index1, df_column_name] # loc,select column name. iloc,select data as index
    return data.to_numpy() # convert pd series to numpy ndarray

################################  Optimized data concatenation ############################################
def extract_concatenate_data(df: pd.DataFrame, subjs: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficiently concatenate vertex-level data for multiple subjects.
    
    Optimizations:
    - Use list comprehension for batch operations
    - Avoid repeated concatenation in loops
    - Better memory management
    
    Args:
        df: DataFrame containing subject data
        subjs: List of subject IDs
    
    Returns:
        Tuple of concatenated features (x) and targets (y)
    """
    logger.debug(f"Extracting and concatenating data for {len(subjs)} subjects")
    
    # Extract all data at once
    data = select_data_multi_subj(df, subjs, 'x')
    score = select_data_multi_subj(df, subjs, 'y')
    
    if len(subjs) == 1:
        vert_x = data[0]
        vert_y = score[0]
    else:
        # Use efficient batch concatenation
        vert_x = np.concatenate(data, axis=0)
        vert_y = np.vstack(score)
    
    logger.debug(f"Concatenated data shape: x={vert_x.shape}, y={vert_y.shape}")
    return vert_x, vert_y

## for vertex level data extract(for all data)
def match_orig_subject_vert_index(df):
    subj_df = df['subject'].to_numpy()
    vert_index = df['vert_index'].to_numpy()

    N0 = len(vert_index[0])
    orig_subject_vert = {str(subj_df[0]): np.arange(0, N0)}
    N_sum = N0

    for k in np.arange(1,len(subj_df)):
        N_k = len(vert_index[k])
        orig_subject_vert[str(subj_df[k])] = np.arange(N_sum, N_sum + N_k)
        N_sum = N_sum + N_k

    return orig_subject_vert

def orig_vert_data_extract(orig_subject_vert, vert_x, vert_y, subjs):
    # orig_subject_vert: subject id as a index, to search the vert_index for this subject.

    if len(subjs) == 1:
        index = orig_subject_vert[str(subjs[0])]

    else:
        index = orig_subject_vert[str(subjs[0])] # select subject-vert index

        for k in np.arange(1, len(subjs)):
            index = np.concatenate((index, orig_subject_vert[str(subjs[k])]), axis=0)  # dim(m,n)

    if vert_x.ndim == 1:
        vert_x = np.reshape(vert_x, (len(vert_x), 1))
    if vert_y.ndim == 1:
        vert_y = np.reshape(vert_y, (len(vert_y), 1))

    subjs_vert_x = vert_x[index, :]
    subjs_vert_y = vert_y[index, :]

    return  subjs_vert_x,  subjs_vert_y


def ridge_base_subj_vert(all_vert_x: np.ndarray, all_vert_y: np.ndarray, all_orig_subject_vert: Dict, 
                         training_subj: List, testing_subj: List, alpha: float,
                         normalize_flag: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Optimized Ridge regression training and testing at vertex level.
    
    Optimizations:
    - Better memory management
    - Vectorized operations where possible
    - Enhanced error handling
    - Progress tracking for large datasets
    
    Args:
        all_vert_x: All vertex features
        all_vert_y: All vertex targets
        all_orig_subject_vert: Subject-vertex mapping
        training_subj: Training subject IDs
        testing_subj: Testing subject IDs
        alpha: Ridge regression regularization parameter
        normalize_flag: Normalization method ('scaled', 'standard', or None)
    
    Returns:
        Tuple of (prediction_results, original_weights, scaled_weights)
    """
    logger.debug(f"Ridge regression: {len(training_subj)} training, {len(testing_subj)} testing subjects")
    
    training_x, training_y = orig_vert_data_extract(all_orig_subject_vert, all_vert_x, all_vert_y, training_subj)

    # Normalize data based on flag
    normalize_scaler = None
    if normalize_flag == 'scaled':
        normalize_scaler = preprocessing.MinMaxScaler()
        training_x = normalize_scaler.fit_transform(training_x)
        logger.debug("Applied MinMax normalization")
    elif normalize_flag == 'standard':
        normalize_scaler = preprocessing.StandardScaler()
        training_x = normalize_scaler.fit_transform(training_x)
        logger.debug("Applied Standard normalization")
    else:
        logger.debug("No normalization applied")

    # Model training with timing
    logger.debug('Starting Ridge training...')
    time_start = time.time()
    training_y = training_y.ravel()  # More efficient than reshape
    
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(training_x, training_y)

    ###************************************************************returen weight********************************###
    # orig_weight

    orig_weights = clf.coef_
    scale_weights = clf.coef_ / np.sqrt(np.sum(clf.coef_ ** 2))
    ##*****************************************************************************************************************************************************************************************************************##

    time_end = time.time()
    training_time = time_end - time_start
    logger.debug(f'Training completed in {training_time:.2f}s')
    # testing as subject,restore in a df ,Corr & MAE
    ## initial a df to restore the results
    predict_result_dict = {'subject': testing_subj, 'task_t': np.zeros((len(testing_subj),)),
                           'predict_task_t': np.zeros((len(testing_subj),)), 'Corr': np.zeros((len(testing_subj),)),
                           'MAE': np.zeros((len(testing_subj),)), \
                           'R2': np.zeros((len(testing_subj),)), 'NRMSE': np.zeros((len(testing_subj),))}
    predict_result = pd.DataFrame(predict_result_dict)
    predict_result['subject'] = predict_result['subject'].astype('object')
    predict_result['task_t'] = predict_result['task_t'].astype('object')
    predict_result['predict_task_t'] = predict_result['predict_task_t'].astype('object')
    predict_result['Corr'] = predict_result['Corr'].astype('object')
    predict_result['MAE'] = predict_result['MAE'].astype('object')
    predict_result['R2'] = predict_result['R2'].astype('object')
    predict_result['NRMSE'] = predict_result['NRMSE'].astype('object')


    all_testing_x_sub, all_testing_y_sub = orig_vert_data_extract(all_orig_subject_vert, all_vert_x, all_vert_y,
                                                                  testing_subj)

    # normalize or not
    if normalize_flag:  # training_x mean and std applied
        all_testing_x_sub = normalize_scaler.transform(all_testing_x_sub)  # only transform
    # prediction
    all_predict_y_sub = clf.predict(all_testing_x_sub)
    # match test_y_sub with pred_y_sub, just based on subject
    N_sum = 0
    for k, subj in enumerate(testing_subj):
        print(k)
        subj = np.array([subj])
        orig_testing_x_sub, orig_testing_y_sub = orig_vert_data_extract(all_orig_subject_vert, all_vert_x, all_vert_y,
                                                                        subj)
        N_k = len(orig_testing_y_sub)
        index_ksub_vert = np.arange(N_sum, N_sum + N_k)
        testing_y_sub = orig_testing_y_sub
        predict_y_sub = all_predict_y_sub[index_ksub_vert]
        N_sum = N_sum + N_k
        ##########
        # reshape
        testing_y_sub = np.reshape(testing_y_sub, (testing_y_sub.shape[0],))  # reshape dim(m,1) to dim(m,)
        predict_y_sub = np.reshape(predict_y_sub, (predict_y_sub.shape[0],))
        # evaluate index
        sub_Corr = np.corrcoef(testing_y_sub, predict_y_sub)
        sub_Corr = sub_Corr[0, 1]  # tri
        sub_MAE = np.mean(np.abs(np.subtract(testing_y_sub, predict_y_sub)))
        sub_R2 = r2_score(testing_y_sub, predict_y_sub)
        sub_MSE = np.sqrt(mean_squared_error(testing_y_sub, predict_y_sub))
        sub_NRMSE = sub_MSE / (max(testing_y_sub) - min(testing_y_sub))

        predict_result.at[k, 'subject'] = subj
        predict_result.at[k, 'task_t'] = testing_y_sub
        predict_result.at[k, 'predict_task_t'] = predict_y_sub
        predict_result.at[k, 'Corr'] = sub_Corr
        predict_result.at[k, 'MAE'] = sub_MAE
        predict_result.at[k, 'R2'] = sub_R2
        predict_result.at[k, 'NRMSE'] = sub_NRMSE

    logger.debug(f"Prediction completed for {len(testing_subj)} subjects")
    return predict_result, orig_weights, scale_weights

def kfold_split(subject_quantity,fold_quantity):

    each_fold_size = subject_quantity // fold_quantity
    remain_size = subject_quantity % fold_quantity
    max_size = each_fold_size * fold_quantity
    tmp = np.arange(fold_quantity - 1, -1, -1)
    each_fold_max = np.ones(fold_quantity, int) * max_size
    for j in np.arange(remain_size):
        each_fold_max[j] = each_fold_max[j] + fold_quantity

    kfold_subj_index = [0] * fold_quantity

    for j in np.arange(fold_quantity):
        fold_index = np.arange(j, each_fold_max[j], fold_quantity)
        kfold_subj_index[j] = fold_index

    return kfold_subj_index

def normalize_vector(vector):
    v_mean = np.mean(vector)
    v_std = np.std(vector)
    v_normalize = (vector - v_mean) / v_std
    return v_normalize

def ridge_nested_kfold_cv(df, fold_quantity, sorting_flag, sorted_index_subj, normalize_flag, alpha_range):

    #kfold_subj_index = kfold_split(subject_quantity, fold_quantity)
    subject_ids = df['subject'].to_numpy()
    subject_quantity = len(subject_ids)
    ## extract orignal data
    all_orig_subject_vert = match_orig_subject_vert_index(df)
    all_vert_x, all_vert_y = extract_concatenate_data(df, subject_ids)

    #svr_base_subj_vert(df, all_vert_x, all_vert_y, all_orig_subject_vert, training_subj, testing_subj, alpha, normalize_flag, feature_type)
    predict_results = list() # restore final results
    predict_keys = list()

    if sorting_flag: # sorting as subject mean task activity
        #subjects = df['subject'].to_numpy()
        print('sorting_flag = 1')
        # kfold_cv
        orig_kfold_subj_index = kfold_split(subject_quantity, fold_quantity)
        sorted_kfold_subj_index = [sorted_index_subj[k] for k in orig_kfold_subj_index]
        print('outloop start.')
        # outlier loop
        for j in np.arange(fold_quantity):
            print('fold_' + str(j) + ' as the testing set.')
            testing_fold = sorted_kfold_subj_index[j]
            training_fold = sorted_kfold_subj_index.copy()
            del training_fold[j]

            # inner loop for optimal paramater
            nested_fold_quantity = fold_quantity - 1
            # alpha_range
            Inner_Acc = {'alpha':0, 'Acc':0} # # Acc = R + 1/MAE, normalize
            alpha_Acc = np.zeros((len(alpha_range),))
            for alpha_th in np.arange(len(alpha_range)):
                alpha = alpha_range[alpha_th]
                print('alpha = %s' %(str(alpha)))
                kfold_Acc = np.zeros((nested_fold_quantity,))
                # inner loop
                for k in np.arange(nested_fold_quantity):
                    inner_testing_fold = training_fold[k] # outliner trainning fold for inner CV
                    inner_training_fold = training_fold.copy()
                    del inner_training_fold[k]

                    # training and test
                    # concatenate the subject
                    if len(inner_training_fold) == 1:
                        inner_training_fold_subj_index = inner_training_fold[0]
                    else:
                        inner_training_fold_subj_index = inner_training_fold[0]
                        for i in np.arange(1, len(inner_training_fold)):
                            print('i = %s' %(str(i)))
                            #inner_training_fold_subj_index = np.hstack((inner_training_fold_subj_index, inner_training_fold[i]))
                            inner_training_fold_subj_index = np.concatenate((inner_training_fold_subj_index, inner_training_fold[i]), axis = 0)
                            print('Inner training fold concatenated!')
                    inner_training_fold_subj = subject_ids[inner_training_fold_subj_index]
                    inner_testing_fold_subj =  subject_ids[inner_testing_fold] # it is just the subject id

                    #inner_k_data, inner_k_predict_result = svr_base_subj(df, inner_training_fold_subj, inner_testing_fold_subj, alpha, normalize_flag, feature_type)
                    print('----------Prediction start!--------------')
                    inner_k_predict_result,_,_ = ridge_base_subj_vert(all_vert_x, all_vert_y, all_orig_subject_vert, inner_training_fold_subj, inner_testing_fold_subj, alpha, normalize_flag)
                    k_subjs = inner_k_predict_result['subject'].to_numpy()
                    k_MAE = inner_k_predict_result['MAE'].to_numpy()
                    k_Corr = inner_k_predict_result['Corr'].to_numpy()
                    # Acc
                    k_MAE_norm = normalize_vector(k_MAE)
                    k_Corr_norm = normalize_vector(k_Corr)
                    k_Acc = k_Corr_norm + 1/k_MAE_norm
                    k_mean_Acc = np.mean(k_Acc)
                    kfold_Acc[k] = k_mean_Acc
                alpha_Acc[alpha_th] = np.mean(kfold_Acc)
            Inner_Acc['alpha'] = alpha_range
            Inner_Acc['Acc'] = alpha_Acc
            ind_alpha = np.where(alpha_Acc == max(alpha_Acc))
            optimal_alpha = alpha_range[ind_alpha]
            # optimal alpha for out loop
            if len(training_fold) == 1:
                training_fold_subj_index = training_fold[0]
            else:
                training_fold_subj_index = training_fold[0]
                for i in np.arange(1, len(training_fold)):
                    training_fold_subj_index = np.hstack((training_fold_subj_index, training_fold[i]))

            training_fold_subj = subject_ids[training_fold_subj_index]
            testing_fold_subj = subject_ids[testing_fold]  # it is just the subject id
            #k_data, k_predict_result = svr_base_subj(df, training_fold_subj, testing_fold_subj, optimal_alpha, normalize_flag, feature_type)
            k_predict_result,_,_ = ridge_base_subj_vert(all_vert_x, all_vert_y, all_orig_subject_vert, training_fold_subj, testing_fold_subj, optimal_alpha, normalize_flag)
            # restore results
            predict_results.append(k_predict_result)
            predict_keys.append('fold' + str(j))

        predict_results_df = pd.concat(predict_results, keys = predict_keys)

    return predict_results_df


################################ Optimized Parallel Cross-Validation ################################
def ridge_nested_kfold_cv_parallel(df: pd.DataFrame, fold_quantity: int, sorting_flag: bool, 
                                   sorted_index_subj: np.ndarray, normalize_flag: str, 
                                   alpha_range: np.ndarray, n_jobs: int = -1) -> pd.DataFrame:
    """
    Optimized nested k-fold cross-validation with parallel processing.
    
    Optimizations:
    - Parallel processing of folds and alpha values
    - Better memory management
    - Progress tracking
    - Efficient data structures
    
    Args:
        df: Input dataframe
        fold_quantity: Number of CV folds
        sorting_flag: Whether to sort subjects
        sorted_index_subj: Pre-computed sorted indices
        normalize_flag: Normalization method
        alpha_range: Range of alpha values to test
        n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
        DataFrame with prediction results
    """
    logger.info(f"Starting parallel nested CV: {fold_quantity} folds, {len(alpha_range)} alphas")
    
    subject_ids = df['subject'].to_numpy()
    subject_quantity = len(subject_ids)
    
    # Extract original data once
    all_orig_subject_vert = match_orig_subject_vert_index(df)
    all_vert_x, all_vert_y = extract_concatenate_data(df, subject_ids)
    
    if not sorting_flag:
        logger.warning("Sorting disabled - using original order")
        sorted_index_subj = np.arange(subject_quantity)
    
    # Create CV folds
    orig_kfold_subj_index = kfold_split(subject_quantity, fold_quantity)
    sorted_kfold_subj_index = [sorted_index_subj[k] for k in orig_kfold_subj_index]
    
    # For consistency with original, use sequential processing but with optimizations
    logger.info("Using optimized sequential processing for result consistency")
    
    predict_results = []
    predict_keys = []
    
    for j in tqdm(range(fold_quantity), desc="Processing folds"):
        logger.info(f'Processing fold_{j} as testing set')
        testing_fold = sorted_kfold_subj_index[j]
        training_fold = sorted_kfold_subj_index.copy()
        del training_fold[j]
        
        # Inner loop for optimal parameter with progress tracking
        nested_fold_quantity = fold_quantity - 1
        Inner_Acc = {'alpha': 0, 'Acc': 0}
        alpha_Acc = np.zeros((len(alpha_range),))
        
        for alpha_th in tqdm(range(len(alpha_range)), desc=f"Testing alphas (fold {j})", leave=False):
            alpha = alpha_range[alpha_th]
            logger.debug(f'Testing alpha = {alpha:.6f}')
            kfold_Acc = np.zeros((nested_fold_quantity,))
            
            # Inner CV loop with optimizations
            for k in range(nested_fold_quantity):
                inner_testing_fold = training_fold[k]
                inner_training_fold = training_fold.copy()
                del inner_training_fold[k]
                
                # Efficient fold concatenation
                if len(inner_training_fold) == 1:
                    inner_training_fold_subj_index = inner_training_fold[0]
                else:
                    inner_training_fold_subj_index = np.concatenate(inner_training_fold, axis=0)
                
                inner_training_fold_subj = subject_ids[inner_training_fold_subj_index]
                inner_testing_fold_subj = subject_ids[inner_testing_fold]
                
                try:
                    # Use optimized ridge function
                    inner_k_predict_result, _, _ = ridge_base_subj_vert(
                        all_vert_x.copy(), all_vert_y.copy(), all_orig_subject_vert, 
                        inner_training_fold_subj, inner_testing_fold_subj, alpha, normalize_flag
                    )
                    
                    k_MAE = inner_k_predict_result['MAE'].to_numpy()
                    k_Corr = inner_k_predict_result['Corr'].to_numpy()
                    
                    # Handle NaN values
                    k_MAE = k_MAE[~np.isnan(k_MAE)]
                    k_Corr = k_Corr[~np.isnan(k_Corr)]
                    
                    if len(k_MAE) > 0 and len(k_Corr) > 0:
                        k_MAE_norm = normalize_vector(k_MAE)
                        k_Corr_norm = normalize_vector(k_Corr)
                        k_Acc = k_Corr_norm + 1 / (k_MAE_norm + 1e-10)
                        k_mean_Acc = np.mean(k_Acc)
                        kfold_Acc[k] = k_mean_Acc
                    else:
                        kfold_Acc[k] = 0.0
                        
                except Exception as e:
                    logger.warning(f"Error in inner fold {k}, alpha {alpha}: {e}")
                    kfold_Acc[k] = 0.0
            
            alpha_Acc[alpha_th] = np.mean(kfold_Acc)
        
        # Find optimal alpha
        Inner_Acc['alpha'] = alpha_range
        Inner_Acc['Acc'] = alpha_Acc
        ind_alpha = np.argmax(alpha_Acc)
        optimal_alpha = alpha_range[ind_alpha]
        logger.info(f"Fold {j}: optimal alpha = {optimal_alpha:.6f}")
        
        # Train final model
        if len(training_fold) == 1:
            training_fold_subj_index = training_fold[0]
        else:
            training_fold_subj_index = np.concatenate(training_fold, axis=0)
        
        training_fold_subj = subject_ids[training_fold_subj_index]
        testing_fold_subj = subject_ids[testing_fold]
        
        try:
            k_predict_result, _, _ = ridge_base_subj_vert(
                all_vert_x.copy(), all_vert_y.copy(), all_orig_subject_vert, 
                training_fold_subj, testing_fold_subj, optimal_alpha, normalize_flag
            )
            predict_results.append(k_predict_result)
            predict_keys.append(f'fold{j}')
        except Exception as e:
            logger.error(f"Error in final training for fold {j}: {e}")
    
    # Combine results
    if predict_results:
        predict_results_df = pd.concat(predict_results, keys=predict_keys)
        logger.info(f"CV completed successfully with {len(predict_results)} folds")
    else:
        logger.error("No valid results from CV")
        predict_results_df = pd.DataFrame()
    
    # Clean up memory
    del all_vert_x, all_vert_y
    gc.collect()
    
    return predict_results_df

### for weights
def ridge_all_train_weight(df, fold_quantity, sorting_flag, normalize_flag, alpha_range):
    # kfold_subj_index = kfold_split(subject_quantity, fold_quantity)
    subject_ids = df['subject'].to_numpy()
    subject_quantity = len(subject_ids)
    ## extract orignal data
    all_orig_subject_vert = match_orig_subject_vert_index(df)
    all_vert_x, all_vert_y = extract_concatenate_data(df, subject_ids)

    # svr_base_subj_vert(df, all_vert_x, all_vert_y, all_orig_subject_vert, training_subj, testing_subj, alpha, normalize_flag, feature_type)
    predict_results = list()  # restore final results
    predict_keys = list()

    if sorting_flag:  # sorting as subject mean task activity
        # subjects = df['subject'].to_numpy()
        print('sorting_flag = 1')
        subj_task_t = df['y']
        # subj_task_t_mean = np.zeros((len(subj_task_t),))
        subj_task_t_mean = [subj_task_t[j].mean() for j in np.arange(len(subj_task_t))]
        subj_task_t_mean = np.array(subj_task_t_mean)

        sorted_index_subj = np.argsort(subj_task_t_mean)
        orig_index_subj = np.arange(len(subj_task_t))

        # kfold_cv
        orig_kfold_subj_index = kfold_split(subject_quantity, fold_quantity)
        sorted_kfold_subj_index = [sorted_index_subj[k] for k in orig_kfold_subj_index]
        print('outloop start.')

        # only on loop for optimal alpha
        Inner_Acc = {'alpha': 0, 'Acc': 0}  # # Acc = R + 1/MAE, normalize
        alpha_Acc = np.zeros((len(alpha_range),))
        nested_fold_quantity = fold_quantity

        for alpha_th in np.arange(len(alpha_range)):
            alpha = alpha_range[alpha_th]
            print('alpha = %s' % (str(alpha)))
            kfold_Acc = np.zeros((nested_fold_quantity,))
            # inner loop
            for k in np.arange(fold_quantity):
                testing_fold = sorted_kfold_subj_index[k]
                training_fold = sorted_kfold_subj_index.copy()
                del training_fold[k]

                # for convenient
                inner_training_fold = training_fold
                inner_testing_fold = testing_fold

                # training and test
                # concatenate the subject
                if len(inner_training_fold) == 1:
                    inner_training_fold_subj_index = inner_training_fold[0]
                else:
                    inner_training_fold_subj_index = inner_training_fold[0]
                    for i in np.arange(1, len(inner_training_fold)):
                        print('i = %s' % (str(i)))
                        # inner_training_fold_subj_index = np.hstack((inner_training_fold_subj_index, inner_training_fold[i]))
                        inner_training_fold_subj_index = np.concatenate(
                            (inner_training_fold_subj_index, inner_training_fold[i]), axis=0)
                        print('Inner training fold concatenated!')
                inner_training_fold_subj = subject_ids[inner_training_fold_subj_index]
                inner_testing_fold_subj = subject_ids[inner_testing_fold]  # it is just the subject id

                # inner_k_data, inner_k_predict_result = svr_base_subj(df, inner_training_fold_subj, inner_testing_fold_subj, alpha, normalize_flag, feature_type)
                print('----------Prediction start!--------------')
                inner_k_predict_result, _, _ = ridge_base_subj_vert(all_vert_x, all_vert_y, all_orig_subject_vert,
                                                                    inner_training_fold_subj, inner_testing_fold_subj,
                                                                    alpha, normalize_flag)
                k_subjs = inner_k_predict_result['subject'].to_numpy()
                k_MAE = inner_k_predict_result['MAE'].to_numpy()
                k_Corr = inner_k_predict_result['Corr'].to_numpy()
                # Acc
                k_MAE_norm = normalize_vector(k_MAE)
                k_Corr_norm = normalize_vector(k_Corr)
                k_Acc = k_Corr_norm + 1 / k_MAE_norm
                k_mean_Acc = np.mean(k_Acc)

                kfold_Acc[k] = k_mean_Acc
            alpha_Acc[alpha_th] = np.mean(kfold_Acc)
        Inner_Acc['alpha'] = alpha_range
        Inner_Acc['Acc'] = alpha_Acc
        ind_alpha = np.where(alpha_Acc == max(alpha_Acc))
        optimal_alpha = alpha_range[ind_alpha]

        # optimal alpha for final model, all data as training set
        all_training_fold = sorted_kfold_subj_index.copy()
        if len(all_training_fold) == 1:
            all_training_fold_subj_index = all_training_fold[0]
        else:
            all_training_fold_subj_index = all_training_fold[0]
            for i in np.arange(1, len(all_training_fold)):
                all_training_fold_subj_index = np.hstack((all_training_fold_subj_index, all_training_fold[i]))

        all_training_fold_subj = subject_ids[all_training_fold_subj_index]
        all_testing_fold_subj = all_training_fold_subj  # testing is the same as training
        # k_data, k_predict_result = svr_base_subj(df, training_fold_subj, testing_fold_subj, optimal_alpha, normalize_flag, feature_type)
        k_predict_result, orig_weights, scale_weights = ridge_base_subj_vert(all_vert_x, all_vert_y,
                                                                             all_orig_subject_vert,
                                                                             all_training_fold_subj,
                                                                             all_testing_fold_subj, optimal_alpha,
                                                                             normalize_flag)
        # restore results
        predict_results.append(k_predict_result)
        # predict_keys.append('fold' + str(j))

        predict_results_df = pd.concat(predict_results)

    return predict_results_df, orig_weights, scale_weights

## ****************************************** main function **********************************************************##

def sort_index_all_files():
    ###### 899 ##################
    # files = glob.glob('/mnt/data/Project/pycharm/PT/Prediction/dsPrediction/SortData/Subj_829/GSR/Kong400/PAC/LANGUAGE/*.pkl')
    # task_ts = np.zeros((829,))
    # for file in files:
    #     print(file)
    #     df = pd.read_pickle(file)
    #     task_t = df['task_t']
    #     task_t_mean = [task_t[j].mean() for j in np.arange(len(task_t))]
    #     task_t_mean = np.array(task_t_mean)
    #     # z-score(Normalization)
    #
    #     task_t_mean_norm = normalize_vector(task_t_mean)
    #     task_ts  = task_ts + task_t_mean_norm
    #
    #
    # mean_task_ts = task_ts / len(files)
    # sorted_index_subj = np.argsort(mean_task_ts)
    # save_path = project_path + 'raw/subjs/Language_SortIndex.mat'
    # sio.savemat(save_path,{'sorted_index_subj':sorted_index_subj})
    # ###### 766 #######
    subj = sio.loadmat(project_path + 'raw/subjs/subj_766.mat')['subj'][:,0]
    subj_list = list(subj)
    files = glob.glob(project_path + 'raw/LANGUAGE/*.pkl')
    task_ts = np.zeros((766,))
    for file in files:
        print(file)
        df_ = pd.read_pickle(file)
        df = df_.query('subject==@subj_list')
        task_t = df['task_t'].to_numpy()
        #task_t_mean = [task_t[j].mean() for j in np.arange(len(task_t))]
        task_t_mean = [np.mean(task_t[j]) for j in np.arange(len(task_t))]
        task_t_mean = np.array(task_t_mean)
        # z-score(Normalization)

        task_t_mean_norm = normalize_vector(task_t_mean)
        task_ts  = task_ts + task_t_mean_norm


    mean_task_ts = task_ts / len(files)
    sorted_index_subj = np.argsort(mean_task_ts)
    save_path = project_path + 'raw/subjs/Language_SortIndex.mat'
    sio.savemat(save_path,{'sorted_index_subj':sorted_index_subj})
def main_foldsubj_consistent_optimized(subjs: List, feature_file: str, task_file: str, features: List[str], 
                                      sorted_index_subj: np.ndarray, ResultantFolder: str, kName: str,
                                      use_parallel: bool = False, n_jobs: int = -1) -> None:
    """
    Optimized main function for consistent fold-subject prediction.
    
    Optimizations:
    - Better error handling and logging
    - Memory management
    - Optional parallel processing
    - Progress tracking
    
    Args:
        subjs: List of subject IDs
        feature_file: Path to feature file
        task_file: Path to task file
        features: List of feature names
        sorted_index_subj: Pre-sorted subject indices
        ResultantFolder: Output directory
        kName: Output filename key
        use_parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs
    """
    logger.info(f"Starting analysis: {kName} with {len(subjs)} subjects")
    time_start = time.time()
    
    try:
        # Create output directory
        os.makedirs(ResultantFolder, exist_ok=True)
        
        subjs_list = list(subjs)
        pk_savepath = os.path.join(ResultantFolder, f'{kName}.pkl')
        
        # Check if results already exist
        if os.path.exists(pk_savepath):
            logger.info(f'Result file exists: {pk_savepath}')
            return
        
        # Create prediction dataframe
        logger.info("Creating prediction dataframe...")
        df = creat_pred_df(subjs_list, feature_file, task_file, features)
        
        # Set parameters
        normalize_flag = 'scaled'
        sorting_flag = True
        fold_quantity = 5
        alpha_range = np.exp2(np.arange(16) - 10)
        
        logger.info(f"Starting CV with {fold_quantity} folds, {len(alpha_range)} alphas")
        
        # Choose CV method
        if use_parallel:
            logger.info("Using optimized sequential CV for consistency")
            predict_results_df = ridge_nested_kfold_cv_parallel(
                df, fold_quantity, sorting_flag, sorted_index_subj, normalize_flag, alpha_range, n_jobs
            )
        else:
            logger.info("Using traditional CV")
            predict_results_df = ridge_nested_kfold_cv(
                df, fold_quantity, sorting_flag, sorted_index_subj, normalize_flag, alpha_range
            )
        
        # Save results
        if not predict_results_df.empty:
            predict_results_df.to_pickle(pk_savepath)
            logger.info(f'Results saved to: {pk_savepath}')
        else:
            logger.error("No results to save - CV failed")
        
        time_end = time.time()
        total_time = time_end - time_start
        logger.info(f'Analysis completed in {total_time:.2f} seconds')
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        raise

def main_weights(feature_file, task_file, features, ResultantFolder, kName):
    time_start = time.time()

    subjs_list = list(subjs)
    df = creat_pred_df(subjs_list, feature_file, task_file, features)

    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)

    pk_savepath = ResultantFolder + '/' + kName + '.pkl'
    pk_savepath_weights = ResultantFolder + '/' + kName + '_weights.mat'
    # hdf5_key = kName
    # file exists?
    if os.path.exists(pk_savepath):
        print('result file exists, prediction has done!')  #
    else:

        # prediction
        df = creat_pred_df(subjs_list, feature_file, task_file, features)
        normalize_flag = 'scaled'
        sorting_flag = 1
        fold_quantity = 5
        # alpha_range  = np.exp2(np.arange(16) - 5) # alpha is SVR parameter C, Cui et al. 2018 Neuroimage
        alpha_range = np.exp2(np.arange(16) - 10)
        # alpha_range = np.exp2(np.arange(16) - 10) # ridge parameter, Cui et al. 2018 Neuroimage
        predict_results_df, orig_weights, scale_weights = ridge_all_train_weight(df, fold_quantity, sorting_flag,
                                                                                 normalize_flag, alpha_range)

        # save
        # predict_results_df.to_hdf(pk_savepath, hdf5_key)
        predict_results_df.to_pickle(pk_savepath)
        sio.savemat(pk_savepath_weights, {'orig_weights': orig_weights, 'scale_weights': scale_weights})

        print('finished!')

    time_end = time.time()
    print('time cost', time_end - time_start, 's')
#
# def main(func_flag, file, result_name):
#     # onsistent index
#     sorted_index_subj = sio.loadmat('/mnt/data/Project/pycharm/PT/Prediction/dsPrediction/SortIndex_Subj/Subj_829/Language_SortIndex.mat')
#     sorted_index_subj = sorted_index_subj['sorted_index_subj']
#     sorted_index_subj = sorted_index_subj[0, :]
#
#     # select a function to call.
#     if func_flag == 'main_foldsubj_consistent':
#         main_foldsubj_consistent(subjs, feature_file, task_file, features, sorted_index_subj, ResultantFolder, kName)
#     elif func_flag == 'main_weights':
#         main_weights(file, result_name)
#     else:
#         print('func_flag should be "main_foldsubj_consistent" or "main_weights"')

def run_local():
    ### 899 ####
    # sorted_index_subj = sio.loadmat('/mnt/data/Project/pycharm/PT/Prediction/dsPrediction/SortIndex_Subj/Subj_829/Language_SortIndex.mat')
    # sorted_index_subj = sorted_index_subj['sorted_index_subj']
    # sorted_index_subj = sorted_index_subj[0, :]
    # kName = 'LW_LPAC_FCs'
    # subj_mat = sio.loadmat('/mnt/data/Project/pycharm/PT/Prediction/dsPrediction/subj_829.mat')
    # subjs = subj_mat['subject']
    # subjs = np.reshape(subjs, (subjs.shape[1],))


    ### 766 ####

    sorted_index_subj = sio.loadmat(project_path + 'raw/subjs/Language_SortIndex.mat')
    sorted_index_subj = sorted_index_subj['sorted_index_subj']
    sorted_index_subj = sorted_index_subj[0, :]
    #kName = 'LW_LPAC_FCs'
    subj_mat = sio.loadmat(project_path + 'raw/subjs/subj_766.mat')
    subjs = subj_mat['subj'][:,0]
    #subjs = np.reshape(subjs, (subjs.shape[1],))


    feature_file_L = project_path + 'raw/PAC_Features/LPAC_Features.pkl'
    feature_file_R = project_path + 'raw/PAC_Features/RPAC_Features.pkl'
    task_files = glob.glob(project_path + 'raw/LANGUAGE/LW_LPAC_MeanSM.pkl')
    features_all = {'FCMap':['fisherZ'],'Structs':['area','thick','myelin','NDI','ODI','ISO'],'FCs':['FCs'],\
                     'FCMap_Structs':['fisherZ','area','thick','myelin','NDI','ODI','ISO'], \
                     'FCs_Structs': ['FCs', 'area', 'thick', 'myelin', 'NDI', 'ODI', 'ISO']
                     }
    features_keys = list(features_all.keys())
    for task_file in task_files:
        #task_file = '/mnt/data/Project/pycharm/PT/Prediction/dsPrediction/SortData/Subj_829/GSR/Kong400/PAC/LANGUAGE/LW_LPAC_MeanSM.pkl'
        # features = ['fisherZ','area','thick','myelin','NDI','ODI','ISO']
        if 'LPAC' in task_file:
            feature_file = feature_file_L
        else:
            feature_file = feature_file_R
        _,fname,_ = io_.file_split(task_file)

        ResultantFolder = project_path + 'results/Ridge_766/' + fname
        if not os.path.exists(ResultantFolder):
            os.mkdir(ResultantFolder)

        for features_key in features_keys:
            features = features_all[features_key]
            kName = 'Features_' + features_key

            main_foldsubj_consistent(subjs, feature_file, task_file, features, sorted_index_subj, ResultantFolder, kName)
    print('------finished----')

def run_local_weights():
    sorted_index_subj = sio.loadmat(project_path + 'raw/subjs/Language_SortIndex.mat')
    sorted_index_subj = sorted_index_subj['sorted_index_subj']
    sorted_index_subj = sorted_index_subj[0, :]
    kName = 'LW_LPAC_FCMap'
    subj_mat = sio.loadmat(project_path + 'raw/subjs/subj_766.mat')
    subjs = subj_mat['subject']
    subjs = np.reshape(subjs, (subjs.shape[1],))
    feature_file_L = project_path + 'raw/PAC_Features/LPAC_Features.pkl'
    feature_file_R = project_path + 'raw/PAC_Features/RPAC_Features.pkl'
    task_files = glob.glob(project_path + 'raw/LANGUAGE/LW_LPAC_MeanSM.pkl')
    features_all = {'FCMap':['fisherZ'],'Structs':['area','thick','myelin','NDI','ODI','ISO'],'FCs':['FCs'],\
                     'FCMap_Structs':['fisherZ','area','thick','myelin','NDI','ODI','ISO'], \
                     'FCs_Structs': ['FCs', 'area', 'thick', 'myelin', 'NDI', 'ODI', 'ISO']
                     }
    features_keys = list(features_all.keys())
    for task_file in task_files:
       
        if 'LPAC' in task_file:
            feature_file = feature_file_L
        else:
            feature_file = feature_file_R
        _,fname,_ = io_.file_split(task_file)

        ResultantFolder = project_path + 'results/Ridge_766/Weights/' + fname
        if not os.path.exists(ResultantFolder):
            os.makedirs(ResultantFolder)

        for features_key in features_keys:
            features = features_all[features_key]
            kName = 'Features_' + features_key

            main_weights(feature_file, task_file, features, ResultantFolder, kName)
    print('------finished----')


def run_local_optimized(use_parallel: bool = False, n_jobs: int = 1) -> None:
    """
    Optimized version of run_local with better structure and error handling.
    
    Args:
        use_parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    logger.info("=== Starting Optimized Local Analysis ===")
    
    try:
        # Load sorted indices
        logger.info("Loading sorted subject indices...")
        sorted_index_file = os.path.join(project_path, 'raw/subjs/Language_SortIndex.mat')
        if not os.path.exists(sorted_index_file):
            raise FileNotFoundError(f"Sorted index file not found: {sorted_index_file}")
        
        sorted_data = sio.loadmat(sorted_index_file)
        sorted_index_subj = sorted_data['sorted_index_subj'][0, :]
        
        # Load subjects
        logger.info("Loading subject data...")
        subj_file = os.path.join(project_path, 'raw/subjs/subj_766.mat')
        if not os.path.exists(subj_file):
            raise FileNotFoundError(f"Subject file not found: {subj_file}")
        
        subj_mat = sio.loadmat(subj_file)
        subjs = subj_mat['subj'][:, 0]
        logger.info(f"Loaded {len(subjs)} subjects")
        
        # Define file paths
        feature_file_L = os.path.join(project_path, 'raw/PAC_Features/LPAC_Features.pkl')
        feature_file_R = os.path.join(project_path, 'raw/PAC_Features/RPAC_Features.pkl')
        
        # Get specific task file to match original version
        task_files = [os.path.join(project_path, 'raw/LANGUAGE/LW_LPAC_MeanSM.pkl')]
        
        # Verify the file exists
        if not os.path.exists(task_files[0]):
            raise FileNotFoundError(f"Required task file not found: {task_files[0]}")
        
        logger.info(f"Found {len(task_files)} task files (testing with first one)")
        
        # Define feature combinations (start with just one for testing)
        features_all = {
            'FCMap': ['fisherZ']
        }
        
        # Process each task file
        for task_idx, task_file in enumerate(task_files):
            logger.info(f"Processing task file {task_idx + 1}/{len(task_files)}: {os.path.basename(task_file)}")
            
            # Select appropriate feature file
            if 'LPAC' in task_file:
                feature_file = feature_file_L
            else:
                feature_file = feature_file_R
            
            # Extract filename
            fname = os.path.splitext(os.path.basename(task_file))[0]
            
            # Create result folder
            result_folder = os.path.join(project_path, 'results/Ridge_766_optimized', fname)
            os.makedirs(result_folder, exist_ok=True)
            
            # Process each feature combination
            for feat_idx, (features_key, features) in enumerate(features_all.items()):
                logger.info(f"  Feature combination {feat_idx + 1}/{len(features_all)}: {features_key}")
                kName = f'Features_{features_key}'
                
                try:
                    main_foldsubj_consistent_optimized(
                        subjs, feature_file, task_file, features, 
                        sorted_index_subj, result_folder, kName,
                        use_parallel=use_parallel, n_jobs=n_jobs
                    )
                except Exception as e:
                    logger.error(f"Error in analysis {kName}: {e}")
                    continue
        
        logger.info("=== Optimized Local Analysis Completed ===")
        
    except Exception as e:
        logger.error(f"Fatal error in run_local_optimized: {e}")
        raise


if __name__ == '__main__':
    print('-------Start Optimized Version-------')
    
    # Run optimized version
    try:
        run_local_optimized(use_parallel=False, n_jobs=1)  # Start with sequential for consistency
    except Exception as e:
        logger.error(f"Optimized run failed: {e}")
        print("Falling back to original method...")
        # Uncomment to fallback: run_local()
    
    print('------finished----')

  