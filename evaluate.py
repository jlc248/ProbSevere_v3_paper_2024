import sys,os
import numpy as np
import pandas as pd
import performance_diagrams
import attributes_diagrams
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score
import lightgbm as lgb
import argparse
import collections
######################################################################################################################

def evaluate(labels, preds, model, outdir, feature_names=None):
  
  os.makedirs(outdir, exist_ok=True)

  # Feature importances
  for it in ['gain','split']:
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax = lgb.plot_importance(model,importance_type=it,xlabel=it,ax=ax,precision=0,ignore_zero=False)
    #for tick in ax.yaxis.get_major_ticks():
    #  tick.label.set_fontsize(10)

    plt.savefig(f"{outdir}/feature_importance_{it}.png", bbox_inches="tight")
    plt.close()

  # AUC, AUPD, BSS
  model_auc = roc_auc_score(labels, preds)
  model_aupd = average_precision_score(labels,preds)
  model_brier_score = mean_squared_error(labels, preds)
  climo_brier_score = mean_squared_error(labels, np.ones(labels.size) * labels.sum() / labels.size)
  model_brier_skill_score = 1 - model_brier_score / climo_brier_score
  print(f"AUC: {model_auc:0.5f}")
  print(f"AUPD: {model_aupd:0.5f}")
  print(f"Brier Score: {model_brier_score:0.5f}")
  print(f"Brier Score (Climatology): {climo_brier_score:0.5f}")
  print(f"Brier Skill Score: {model_brier_skill_score:0.5f}")

  of = open(f'{outdir}/verification_scores.txt','w')
  of.write("Test Size : "+str(labels.size)+'\n')
  of.write("AUC: " + "{:.5f}".format(model_auc) + '\n')
  of.write("AUPD: " + "{:.5f}".format(model_aupd) + '\n')
  of.write("Brier Score: " + "{:.5f}".format(model_brier_score) + '\n')
  of.write("Brier Score (Climatology): " + "{:.5f}".format(climo_brier_score) + '\n')
  of.write("Brier Skill Score: " + "{:.5f}".format(model_brier_skill_score) + '\n')

  FONT_SIZE = 8
  plt.rc('font', size=FONT_SIZE)
  plt.rc('axes', titlesize=FONT_SIZE)
  plt.rc('axes', labelsize=FONT_SIZE)
  plt.rc('xtick', labelsize=FONT_SIZE)
  plt.rc('ytick', labelsize=FONT_SIZE)
  plt.rc('legend', fontsize=FONT_SIZE)
  plt.rc('figure', titlesize=FONT_SIZE)

  # Performance diagram
  scores_dict = performance_diagrams.plot_performance_diagram(\
                  observed_labels=labels, forecast_probabilities=preds)
  csi = scores_dict['csi']; pss = scores_dict['pss']; bin_acc = scores_dict['bin_acc']; bin_bias = scores_dict['bin_bias']
  perf_diagram_file_name = f'{outdir}/performance_diagram.png'

  plt.savefig(perf_diagram_file_name, dpi=300, bbox_inches='tight')
  plt.close()

  # Write out scores
  best_ind = np.argmax(csi)
  prob_of_max_csi = best_ind / 1000. # this only works in binarization_thresholds is linspace(0,1.01,0.001)
  of.write('Max CSI: ' + "{:.5f}".format(np.max(csi)) + '; at prob: ' + "{:.3f}".format(prob_of_max_csi) +'\n')
  of.write('bin. freq. bias (at max CSI): ' + "{:.3f}".format(bin_bias[best_ind]) + '\n')
  of.write('bin. acc. (at max CSI): ' + "{:.3f}".format(bin_acc[best_ind]) + '\n')
  of.write('Max PSS: ' + "{:.3f}".format(np.max(pss)) + '; at prob: ' + "{:.3f}".format(np.argmax(pss)/1000.) + '\n')
  if(feature_names is not None): of.write(f"features: {feature_names} \n")
  of.close()


  # Attributes diagram
  attributes_diagrams.plot_attributes_diagram(observed_labels=labels,forecast_probabilities=preds, num_bins=20)
  attr_diagram_file_name = f'{outdir}/attributes_diagram.png'
  plt.savefig(attr_diagram_file_name, dpi=300,bbox_inches="tight")
  plt.close()

  pickle.dump(scores_dict,open(f"{outdir}/scores.pkl","wb"))

  return scores_dict
