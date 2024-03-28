import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import pickle
import os,sys
import matplotlib.pyplot as plt
import argparse
import lightgbm as lgb
import evaluate
import time

#################################################################################################
def make_val_dataset(val_set,
                     cols_to_drop,
                     probwind_test=False,
                     probhail_test=False,
                     probtor_test=False):
             
    ph = val_set['probhail']
    pw = val_set['probwind']
    pt = val_set['probtor']
    psv1 = val_set['prob_severe']
    psv2 = np.max([ph,pw,pt],axis=0)
    X_val = val_set.drop(columns=cols_to_drop)

    hail = np.array(val_set['hail'])
    wind = np.array(val_set['wind'])
    torn = np.array(val_set['torn'])
    if(probtor_test):
      val_labels = torn
    elif(probhail_test):
      val_labels = hail
    elif(probwind_test):
      val_labels = wind
    else:
      val_labels = np.max([hail,wind,torn],axis=0)
    val_labels[val_labels >= 0] = 1
    val_labels[val_labels < 0] = 0
    print(f'val yes: {np.sum(val_labels)}')
    print(f'val total: {len(val_labels)}')
    print('val y/total ratio: ' + "{:.4f}".format(np.sum(val_labels)/len(val_labels)))

    return {'X_val':X_val,
            'val_labels':val_labels,
            'ph':ph, 'pw':pw, 'pt':pt, 'psv1':psv1, 'psv2':psv2} 
###################################################################################################
def get_best_params(probhail_test,probwind_test,probtor_test):
  if(probhail_test):
    params = {'num_leaves': 190,
              'n_estimators': 360,
              'min_data_in_leaf': 20720,
              'max_bin': 255,
              'learning_rate': 0.012574334296829354,
              'lambda_l2': 0.6326530612244897,
              'lambda_l1': 0.5714285714285714,
              'feature_fraction': 0.5555555555555556,
              'bagging_fraction': 1.0}

  elif(probwind_test):
    params =  {'num_leaves': 71,
               'n_estimators': 500,
               'min_data_in_leaf': 23820,
               'max_bin': 255,
               'learning_rate': 0.02325385717596329,
               'lambda_l2': 0.18367346938775508,
               'lambda_l1': 0.1020408163265306,
               'feature_fraction': 0.6111111111111112,
               'bagging_fraction': 0.7777777777777778}

  elif(probtor_test):
     params = {'num_leaves': 124,
               'n_estimators': 680,
               'min_data_in_leaf': 36600,
               'max_bin': 255,
               'learning_rate': 0.010495581875524821,
               'lambda_l2': 0.0,
               'lambda_l1': 0.12244897959183673,
               'feature_fraction': 0.4,
               'bagging_fraction': 0.533333333333333}
  

  else: #prob any severe
      params = {'num_leaves': 156,
                'n_estimators': 700,
                'min_data_in_leaf': 32400,
                'max_bin': 255,
                'learning_rate': 0.017320508075688777,
                'lambda_l2': 0.16326530612244897,
                'lambda_l1': 0.7142857142857142,
                'feature_fraction': 0.5333333333333333,
                'bagging_fraction': 0.5333333333333333}


  params['objective'] = 'binary'

  return params, params['n_estimators']

#####################################################################################################
def main(train_df,
         val_df,
         outdir,
         probtor_test=False,
         probwind_test=False,
         probhail_test=False,
         best_params=False,
         early_stopping_rounds=5,
):


  os.makedirs(outdir, exist_ok=True)

  print(f'Reading training DF, {train_df}...')
  train_set = pd.read_pickle(train_df)

  print(f'Reading validation DF, {val_df}...')
  val_set = pd.read_pickle(val_df)

  print(f"# of train: {len(train_set)}; # of val: {len(val_set)}")
  print(f"val/train: {len(val_set)/(len(train_set)+len(val_set))}")
 
  all_preds=list(train_set)

  #any severe
  good_predictors = ['max_mesh','max_compref','max_vil','max_llazshear','max_mlazshear','p98_llazshear','p98_mlazshear',
               'icp','maxrc_emiss','flash_rate','max_tltg_density',
               'LAPSERATE_03KM_merged_smoothed','MAX_LAPSERATE_26KM_merged_smoothed','PWAT_merged_smoothed','MLCAPE_merged_smoothed',
               'EBS_merged_smoothed','MEANWIND_1-3kmAGL_merged_smoothed','WETBULB_0C_HGT_merged_smoothed','SFC_LCL_merged_smoothed',
               'SRW46KM_merged_smoothed','SRW02KM_merged_smoothed']

  #hail
  if(probhail_test):
    good_predictors = ['max_mesh','max_llazshear','max_mlazshear','p98_mlazshear','max_ref20','max_compref','max_etop50','max_h50a0c',
                 'maxrc_emiss','flash_rate','max_tltg_density','icp','max_fed',
                 'EBS_merged_smoothed','LAPSERATE_03KM_merged_smoothed','MAX_LAPSERATE_26KM_merged_smoothed','MEANWIND_1-3kmAGL_merged_smoothed',
                 'MUCAPE_merged_smoothed','MLCAPE_merged_smoothed','PWAT_merged_smoothed','WETBULB_0C_HGT_merged_smoothed','SRW46KM_merged_smoothed']

  #wind
  if(probwind_test):
    good_predictors = ['max_vil','max_llazshear','p98_llazshear','max_mlazshear','max_compref','max_ref10','max_ref20',
                 'maxrc_emiss','icp','flash_rate','max_tltg_density','avg_group_area',
                 'MEANWIND_1-3kmAGL_merged_smoothed','LAPSERATE_03KM_merged_smoothed','MAX_LAPSERATE_26KM_merged_smoothed',
                 'PWAT_merged_smoothed','WETBULB_0C_HGT_merged_smoothed','SRH_01KM_merged_smoothed','SRW02KM_merged_smoothed',
                 'MLCAPE_merged_smoothed','DCAPE_merged_smoothed','SRW46KM_merged_smoothed','CAPE_M10M30_merged_smoothed']

  #tornado
  if(probtor_test):
    good_predictors = ['max_llazshear','p98_llazshear','max_mlazshear','p98_mlazshear','max_mesh','max_compref','icp','flash_rate',
               'SFC_LCL_merged_smoothed','EBS_merged_smoothed','MEANWIND_1-3kmAGL_merged_smoothed',
               'SRH_01KM_merged_smoothed','SRW02KM_merged_smoothed','SRW46KM_merged_smoothed',
               'MAX_LAPSERATE_26KM_merged_smoothed','LAPSERATE_03KM_merged_smoothed','MLCAPE_merged_smoothed','WETBULB_0C_HGT_merged_smoothed',
               'MLCIN_merged_smoothed']
  
  # Drop non-predictors
  cols_to_drop = []
  for elem in all_preds:
    if(elem not in good_predictors): cols_to_drop.append(elem)
  # Ensure that all of our predictors are in the dataframe
  for good_pred in good_predictors:
    if(good_pred not in all_preds): print(f'{good_pred} not found in DF.'); sys.exit(1)

  # Labels
  print('Making training and validation X and y...')

  X_train = train_set.drop(columns=cols_to_drop)
 
  hail = np.array(train_set['hail'])
  wind = np.array(train_set['wind'])
  torn = np.array(train_set['torn'])
  if(probtor_test):
    train_labels = torn
  elif(probhail_test):
    train_labels = hail
  elif(probwind_test):
    train_labels = wind
  else:
    train_labels = np.max([hail,wind,torn],axis=0)
  train_labels[train_labels >= 0] = 1
  train_labels[train_labels < 0] = 0
  print(f'train yes: {np.sum(train_labels)}')
  print(f'train total: {len(train_labels)}')
  print('train y/total ratio: ' + "{:.4f}".format(np.sum(train_labels)/len(train_labels)))
  predictors = list(X_train)
  print('predictors:',predictors)

  # Get the parameters  
 
  if(best_params):
    # Parameters from Cintineo et al. 2024
    params, num_rounds = get_best_params(probhail_test,probwind_test,probtor_test)
  else:
    params = {
        #parameters to tune      #defaults
        'max_depth':-1,          #-1
        'min_child_weight':1e-3, #1e-3
        'learning_rate':0.1,     #0.1
        'bagging_fraction':1,    #1
        'feature_fraction':1,    #1 
        'num_leaves':31,         #31
        #other parameters
        'objective':'binary',
        'min_data_in_leaf':20,  #20 #min number of samples in leaf
        'lambda_l1':0,           #0
        'lambda_l2':0,            #0
        'is_unbalance':False,
        'scale_pos_weight':1, #1
    }
    num_rounds = 500
  
  params['num_threads'] = 40
  params['metric'] = 'average_precision'

  # Save params
  pickle.dump(params,open(f"{outdir}/params.pkl","wb"))

  es = lgb.early_stopping(stopping_rounds=early_stopping_rounds)

  # Train one model
  print('Training a model...')

  # Create the val set
  result_dict = make_val_dataset(val_set,
                                 cols_to_drop,
                                 probtor_test=probtor_test,
                                 probwind_test=probwind_test,
                                 probhail_test=probhail_test)
  val_labels = result_dict['val_labels']
  X_val = result_dict['X_val']
                         
  # Model object
  clf = lgb.LGBMClassifier(objective='binary',importance_type="gain",n_jobs=params['num_threads']) 
  clf.set_params(**params) 

  # Fit the model
  clf.fit(X_train,
          train_labels,
          callbacks=[es],
          feature_name=list(X_train),
          eval_metric=['average_precision'],
          eval_set=[(X_val,val_labels)])
  pickle.dump(clf,open(f'{outdir}/lgb_classifier.pkl','wb'))

  # Compute predictions on validation set
  preds = clf.predict_proba(X_val,num_iteration=clf.best_iteration_)
  preds=preds[:,1]
  
  # Save info, including val labels and val predictions.
  pickle.dump({'train_df':train_df,
               'val_df':val_df,
               'val_labels':val_labels,
               'val_pred':preds},
               open(f"{outdir}/val_lab_pred.pkl","wb"))

  scores_dict = evaluate.evaluate(val_labels, preds, clf, outdir, feature_names=list(X_val))

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train and validate a LGBM GBDT. Default is to train against 'any severe' report.")
  parser.add_argument('train',help="Training DataFrame",type=str)
  parser.add_argument('val',help="Validation DataFrame", type=str)
  parser.add_argument('-o','--outdir',help="Root output directory for model and figures. Default=$PWD/OUTPUT/",
                       default=os.path.join(os.environ['PWD'],'OUTPUT/'),type=str)
  parser.add_argument('-pt','--probtor_test',help="Train and test a tornado only model",action="store_true")
  parser.add_argument('-ph','--probhail_test',help="Train and test a hail only model",action="store_true")
  parser.add_argument('-pw','--probwind_test',help="Train and test a wind only model",action="store_true")
  parser.add_argument('-bp','--best_params',help="Train a model with pre-defined optimal parameters (hard-coded in function). \
                      Use in conjuction with -pt, -ph, -pw or none (which would be prob any severe)",action="store_true")
  parser.add_argument('-es','--early_stopping_rounds',help="Number of early stopping rounds. Default = 5. Make this a huge number \
                      if you don't want early stopping.",default=5,type=int)

  args = parser.parse_args()

  main(args.train,
       args.val,
       args.outdir,
       probtor_test=args.probtor_test,
       probhail_test=args.probhail_test,
       probwind_test=args.probwind_test,
       best_params=args.best_params,
       early_stopping_rounds=args.early_stopping_rounds)
