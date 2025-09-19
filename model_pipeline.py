# this whole thing is a bioactivity prediction & analysis pipeline.
# it works as a script or you can import its functions.
#
# main ideas:
# 1. splits features into 'cheap' vs 'expensive' groups. you can tweak what goes where.
# 2. trains a 'stage 1' model to predict expensive features from cheap ones. related to transfer learning.
# 3. runs a leave-one-out cross-validation (loocv) to test a bunch of different models
#    (e.g., using only sequence, only structure, all combined, or the 'transfer' ones).
# 4. compares l1 (lasso) and l2 (ridge) logistic regression, which is good for small datasets.
# 5. can also estimate bias & variance using bootstrap if you want.
#
# if you would like to use it in your own script - 
# ---------------------------
# >>> import pandas as pd
# >>> from pipeline import PipelineConfig, run_pipeline
# >>> df = pd.read_csv('filtered_residues_with_MD.csv')
# >>> results_df, details = run_pipeline(df, config=PipelineConfig(verbose=True))
# >>> print(results_df)
#
# if you get a new expensive feature, like 'MM_GBSA_Energy', just make sure its name
# has 'gbsa' in it (and 'gbsa' is in the `docking_patterns` list in the config),
# or just add the full column name to `expensive_explicit` in the config.
#
# new cheap features are handled automatically. if they're 0/1, they'll probably
# be grouped as 'structure_only', otherwise they just become generic cheap features.
#
# you can still run it from the command line too:
#     python pipeline.py --csv filtered_residues_with_MD.csv --save_report report.tsv

from __future__ import annotations
import argparse
import sys
import warnings
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, precision_recall_fscore_support,
    roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.utils import resample
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning) 

def load_mmgbsa_features(directory: str, cfg: PipelineConfig) -> Optional[pd.DataFrame]:
    """grabs mm-gbsa csvs from a folder, averages, and returns a feature df.

    it looks for files like: prime_mmgbsa_A123-out.csv (case doesn't matter).
    it pulls out the position number (123).
    for each file, it:
        - reads the csv.
        - drops empty columns.
        - keeps only numeric columns.
        - calculates the mean for each numeric column.
        - prefixes the column names with 'mmgbsa_' and stores the result.
    if it finds multiple files for the same position, the last one wins (and it'll warn you).
    spits out a df with columns: position, mmgbsa_<original_col_lower>.
    """
    if not os.path.isdir(directory): # check for directory
        _vprint(cfg, f"[MMGBSA] Directory not found: {directory}")
        return None
    pattern = re.compile(r"prime[_-]mmgbsa_([A-Za-z]+)(\d+)-out\.csv", re.IGNORECASE) # regex to find the right files
    records = []
    files = []
    for root, _, fnames in os.walk(directory): # walk through the directory to find all matching files
        for f in fnames:
            if pattern.match(f):
                files.append(os.path.join(root, f))
    if not files: # no files found, nothing to do
        _vprint(cfg, f"[MMGBSA] No matching files in {directory}")
        return None
    for fp in files: # now process each file we found
        m = pattern.search(os.path.basename(fp))
        if not m:
            continue
        pos_str = m.group(2)
        try:
            position = int(pos_str) # grab the position number
        except ValueError:
            continue
        try:
            df_file = pd.read_csv(fp) # try to read the csv
        except Exception as e:
            _vprint(cfg, f"[MMGBSA] Failed reading {fp}: {e}")
            continue
        
        df_file = df_file.loc[:, ~df_file.columns.str.contains('^Unnamed')] # drop unnamed columns
        
        cols_numeric = [] # figure out which columns are numeric
        for c in df_file.columns:
            if c.lower() == 'title': # keep 'title' out of it
                continue
            
            series = pd.to_numeric(df_file[c], errors='coerce') # try to convert to numbers
            if series.notna().sum() == 0: # if it's all NaNs, skip
                continue
            cols_numeric.append(c)
        if not cols_numeric: # if no numeric columns, move on
            continue
        means = df_file[cols_numeric].apply(pd.to_numeric, errors='coerce').mean(axis=0, skipna=True) # calc the mean of each col
        rec = {'position': position}
        for c, v in means.items(): # add the means to our record
            norm_c = 'mmgbsa_' + c.strip().lower()
            rec[norm_c] = v
        records.append(rec)
    if not records: # if we didn't get any data, warn and return None
        _vprint(cfg, "[MMGBSA] No usable numeric content extracted")
        return None
    feat_df = pd.DataFrame(records).sort_values('position').drop_duplicates('position', keep='last') # create the final df
    return feat_df


@dataclass
class PipelineConfig:
    """the config of the pipeline

    just change the patterns or add explicit column names here to adapt to new data.
    patterns are just substrings, case doesn't matter.
    """
    label_col: str = 'Bioactivity' # what we're trying to predict
    # patterns to identify column groups
    md_patterns: List[str] = field(default_factory=lambda: ['rmsd', 'rmsf', 'sasa', 'rg', 'entropy','pc1', 'pc2', 'mahalanobis'])
    docking_patterns: List[str] = field(default_factory=lambda: ['mmgbsa', 'vdw'])
    stability_patterns: List[str] = field(default_factory=lambda: ['stability', 'instability', 'delta_pi', 'ddg'])
    sequence_id_cols: List[str] = field(default_factory=lambda: ['position', 'residue'])
    # if patterns aren't enough, just put column names here
    cheap_explicit: List[str] = field(default_factory=list)
    expensive_explicit: List[str] = field(default_factory=list)
    # columns to just ignore completely
    exclude_columns: List[str] = field(default_factory=list)
    # settings for bootstrap
    bootstrap_B: int = 100 # how many bootstrap samples
    # turn this on to calculate bias/variance (takes longer)
    compute_bias_variance: bool = True
    # verbosity
    verbose: bool = False # print a bunch of stuff
    # pca settings
    auto_pca_feature_threshold: int = 15 # if a model has more than this many features, run pca
    random_state: int = 42
    # whether to calculate feature importances
    compute_feature_importance: bool = True



def _vprint(cfg: PipelineConfig, *msg): 
    if cfg.verbose:
        print(*msg)


def infer_groups(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, List[str]]:
    """sorts columns into buckets like 'cheap', 'expensive', etc. using patterns and explicit lists.

    rules:
      1. ignore the label column and any excluded columns.
      2. if a column is in an explicit list, it goes there. no questions asked.
      3. otherwise, try to match it against the pattern lists (md, docking, stability).
      4. anything left over is considered 'cheap'.
      5. 'position' and 'residue' are always cheap.
    this is more flexible than just assuming columns are in a certain order.
    """
    label = config.label_col
    assert label in df.columns, f"Label column '{label}' missing"

    cols = [c for c in df.columns if c not in config.exclude_columns and c != label]
    lower_map = {c: c.lower() for c in cols} 

    md_cols, docking_cols, stability_cols = [], [], []
    sequence_cols = [c for c in config.sequence_id_cols if c in df.columns]
    cheap_generic = []
    expensive_all = []

    def matches(patterns: List[str], col_lower: str) -> bool: 
        return any(p in col_lower for p in patterns)

    # first pass to sort the columns
    for c in cols:
        cl = lower_map[c]
        if c in config.expensive_explicit: # check explicit expensive list first
            expensive_all.append(c)
            continue
        if c in config.cheap_explicit: # then explicit cheap list
            cheap_generic.append(c)
            continue
        if c in sequence_cols: # sequence identifiers are handled separately
            continue
        if matches(config.md_patterns, cl): # md features are expensive
            md_cols.append(c)
            expensive_all.append(c)
            continue
        if matches(config.docking_patterns, cl): # docking features are expensive
            docking_cols.append(c)
            expensive_all.append(c)
            continue
        if matches(config.stability_patterns, cl): # stability features are usually cheap
            stability_cols.append(c)
            cheap_generic.append(c)
            continue
        
        if any(t in cl for t in ['mahalanobis', 'pc1', 'pc2']) and c not in expensive_all: # heuristic: some simulation outputs are always expensive
            docking_cols.append(c)
            expensive_all.append(c)
            continue
        # if we don't know what it is, assume it's cheap
        cheap_generic.append(c)

    
    def unique(seq): # a little function to remove duplicates while keeping order
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    sequence_only = sequence_cols
    structure_only = [c for c in cheap_generic if c not in stability_cols + sequence_cols]
    cheap_all = unique(sequence_cols + structure_only + stability_cols)
    expensive_all = unique(expensive_all)
    md_cols = unique(md_cols)
    docking_cols = unique(docking_cols)
    stability_cols = unique(stability_cols)

    groups = { # put everything into a dict
        'sequence_only': sequence_only,
        'structure_only': structure_only,
        'stability_only': stability_cols,
        'md_only': md_cols,
        'docking_only': docking_cols,
        'expensive_all': expensive_all,
        'cheap_all': cheap_all,
    }
    _vprint(config, "[infer_groups] Groups inferred:")
    for k, v in groups.items():
        _vprint(config, f"  - {k}: {len(v)} cols")
    return groups


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    """makes a preprocessor: one-hot for 'residue', scale everything else."""
    numeric_cols = [c for c in feature_cols if c != 'residue']
    transformers = []
    if 'residue' in feature_cols: # if we have a residue column, one-hot encode it
        transformers.append(('residue_ohe', OneHotEncoder(handle_unknown='ignore'), ['residue']))
    if numeric_cols: # for all other numeric columns, impute missing values and scale them
        transformers.append(('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), numeric_cols))
    return ColumnTransformer(transformers)


def tune_logreg(X, y, penalty: str, cv_splits: int = 5) -> LogisticRegression:
    """quick and dirty grid search for the best C in logistic regression."""
    Cs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] # the C values we'll try
    solver = 'liblinear' if penalty in ('l1', 'l2') else 'lbfgs'
    best_score = np.inf
    best_model = None
    
    unique_classes = np.unique(y) 
    if len(unique_classes) < 2:
        return LogisticRegression(penalty=penalty, C=1.0, solver=solver, max_iter=500)
    cv_k = min(cv_splits, len(y))
    if cv_k < 3:
        cv_k = 2
    splitter = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
    for C in Cs:
        losses = []
        for tr_idx, va_idx in splitter.split(X, y):
            model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=500)
            model.fit(X[tr_idx], y[tr_idx])
            
            if len(model.classes_) == 2: # get probability for the positive class
                probs = model.predict_proba(X[va_idx])[:, 1]
            else: # if a fold is weird and only has one class
                probs = np.full(len(va_idx), y[tr_idx].mean())
            
            probs = np.clip(probs, 1e-6, 1 - 1e-6) # clip probs to avoid log(0) errors in log_loss
            losses.append(log_loss(y[va_idx], probs))
        mean_loss = np.mean(losses)
        if mean_loss < best_score: # if this C is better, save it
            best_score = mean_loss
            best_model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=500)
    return best_model


@dataclass
class LOOResult:
    """just a simple container to hold the results from a leave-one-out run."""
    model_name: str
    penalty: str
    accuracy: float
    roc_auc: Optional[float] # area under the roc curve
    log_loss_value: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    f2: Optional[float] # f2 score, favors recall
    n: int
    bias: Optional[float] = None
    variance: Optional[float] = None
    stage1_r2_mean: Optional[float] = None # how well our stage1 model did
    f2_opt: Optional[float] = None # best possible f2 score after tweaking the threshold
    f2_opt_threshold: Optional[float] = None
    f2_opt_precision: Optional[float] = None
    f2_opt_recall: Optional[float] = None


def bias_variance_bootstrap(model_builder, X_train, y_train, x_test, y_test, B=100, random_state=42):
    """estimates bias & variance for one test point using bootstrap. returns mean_prob, bias, variance."""
    rng = np.random.RandomState(random_state)
    preds = []
    for b in range(B):
        
        X_bs, y_bs = resample(X_train, y_train, random_state=rng.randint(0, 1_000_000)) # grab a bootstrap sample, with replacement
        if len(np.unique(y_bs)) < 2:
            continue  # skip if we don't have both classes in our sample
        model = model_builder()
        model.fit(X_bs, y_bs)
        if len(model.classes_) == 2:
            p = model.predict_proba(x_test.reshape(1, -1))[0, 1]
        else:
            p = float(y_bs.mean())
        preds.append(p)
    if not preds: # if all our samples were bad
        return 0.5, None, None
    preds = np.array(preds)
    mean_p = preds.mean()
    bias = (mean_p - y_test) ** 2
    variance = preds.var(ddof=1)
    return mean_p, bias, variance


def stage1_fit_predict(cheap_df: pd.DataFrame, expensive_df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, float, Optional[Pipeline]]:
    """trains a model to predict expensive features from cheap ones and returns the predictions."""
    
    data = pd.concat([cheap_df, expensive_df], axis=1) # combine cheap and expensive data
    valid = ~expensive_df.isna().any(axis=1) # only use rows where we have *all* the expensive data
    data_valid = data[valid]
    if data_valid.empty or data_valid.shape[0] < 5:
        # not enough data to train, so we'll just have to use mean imputation later
        preds = pd.DataFrame(index=cheap_df.index, columns=expensive_df.columns, dtype=float)
        return preds, float('nan'), None

    X = data_valid[cheap_df.columns]
    y = data_valid[expensive_df.columns]

    num_cols = X.columns.tolist()
    # prep the cheap features, one-hot encoding 'residue' if it's there
    residue_present = 'residue' in X.columns
    transformers = []
    if residue_present:
        transformers.append(('res_ohe', OneHotEncoder(handle_unknown='ignore'), ['residue']))
        num_cols_no_res = [c for c in num_cols if c != 'residue']
    else:
        num_cols_no_res = num_cols
    if num_cols_no_res:
        transformers.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ]), num_cols_no_res))
    pre = ColumnTransformer(transformers)

    base = Ridge(alpha=1.0) # use a simple ridge regressor
    reg = MultiOutputRegressor(base)
    pipe = Pipeline([
        ('pre', pre),
        ('reg', reg)
    ])
    _vprint(cfg, f"[Stage1] Fitting surrogate on {X.shape[0]} rows, {X.shape[1]} cheap -> {y.shape[1]} expensive outputs")
    pipe.fit(X, y)

    # quick in-sample r^2 check. not super robust but gives us an idea.
    try:
        r2_scores = []
        for i, col in enumerate(y.columns):
            y_pred_col = pipe.predict(X)[:, i]
            ss_res = ((y[col] - y_pred_col) ** 2).sum()
            ss_tot = ((y[col] - y[col].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            r2_scores.append(r2)
        mean_r2 = float(np.nanmean(r2_scores))
    except Exception:
        mean_r2 = float('nan')

    all_preds_arr = pipe.predict(cheap_df) # make predictions for all rows
    preds = pd.DataFrame(all_preds_arr, index=cheap_df.index, columns=expensive_df.columns)
    _vprint(cfg, f"[Stage1] Mean in-sample R2 across expensive outputs: {mean_r2:.3f}")
    return preds, mean_r2, pipe


def run_loocv(df: pd.DataFrame, groups: Dict[str, List[str]], cfg: PipelineConfig) -> Tuple[List[LOOResult], Dict[Tuple[str,str], pd.DataFrame], Dict[Tuple[str,str], pd.DataFrame], Pipeline]:
    label_col = cfg.label_col
    labeled_mask = ~df[label_col].isna() # find rows that have labels
    labeled_df = df[labeled_mask].copy()
    y = labeled_df[label_col].astype(int).values
    results: List[LOOResult] = []
    loo = LeaveOneOut()

    # first, predict expensive features for all rows using our stage1 model
    cheap_all_cols = groups['cheap_all']
    expensive_cols = groups['expensive_all']
    cheap_all_df = df[cheap_all_cols]
    expensive_all_df = df[expensive_cols]
    stage1_preds, stage1_r2, stage1_model = stage1_fit_predict(cheap_all_df, expensive_all_df, cfg)

    # just get the predictions for the labeled rows
    stage1_preds_labeled = stage1_preds.loc[labeled_df.index]

    # these are all the different models we're gonna test
    variants: Dict[str, List[str]] = {
        'sequence_only': groups['sequence_only'],
        'structure_only': groups['structure_only'],
        'stability_only': groups['stability_only'],
        'md_only': groups['md_only'],
        'docking_only': groups['docking_only'],
        'combined_all': groups['cheap_all'] + groups['expensive_all'],
        'transfer_augmented': groups['cheap_all']  # cheap features + predicted expensive ones
    }

    penalties = ['l2', 'l1']  # l2 (ridge) vs l1 (lasso)
    importance_records: Dict[Tuple[str,str], pd.DataFrame] = {} # for saving feature importances
    prediction_records: Dict[Tuple[str,str], List[Tuple[int,float,int]]] = {} # for saving raw predictions

    for variant_name, cols in variants.items(): # loop through each model variant
        if not cols and variant_name != 'transfer_augmented': # skip if there are no columns for this variant
            continue
        if variant_name == 'transfer_augmented':
            base_X_df = labeled_df[groups['cheap_all']].copy()
            X_full = pd.concat([base_X_df.reset_index(drop=True), # combine cheap feats with predicted expensive ones
                                 stage1_preds_labeled.reset_index(drop=True).add_prefix('TL_')], axis=1)
        else:
            X_full = labeled_df[cols].copy()

        categorical = [c for c in X_full.columns if c == 'residue']
        numeric_cols = [c for c in X_full.columns if c not in categorical]
        pre = ColumnTransformer([ # set up the preprocessor for this variant's data
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('sc', StandardScaler())
            ]), numeric_cols)
        ])
        # if there are too many features, and we're not calculating importances, just use pca
        apply_pca = (X_full.shape[1] > cfg.auto_pca_feature_threshold and len(y) > 5 and not cfg.compute_feature_importance)
        pca_step = ('pca', PCA(n_components=min(len(y)-1, max(2, int(np.sqrt(X_full.shape[1])))))) if apply_pca else None

        for penalty in penalties: # now try both l1 and l2 penalties
            y_probs: List[float] = []
            y_preds: List[int] = []
            biases: List[float] = []
            variances: List[float] = []
            fold_importances: List[pd.DataFrame] = []
            for fold_i, (train_idx, test_idx) in enumerate(loo.split(X_full)): # the main leave-one-out loop
                X_train_df = X_full.iloc[train_idx]
                X_test_df = X_full.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                pre.fit(X_train_df, y_train) # preprocess the data
                X_train = pre.transform(X_train_df)
                X_test = pre.transform(X_test_df)
                if pca_step: # apply pca if we decided to
                    pca = PCA(n_components=pca_step[1].n_components)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                tuned = tune_logreg(X_train, y_train, penalty=penalty) # find the best C for this fold
                tuned.fit(X_train, y_train)
                prob = tuned.predict_proba(X_test)[0, 1] if len(tuned.classes_) == 2 else float(y_train.mean())
                y_probs.append(prob)
                y_preds.append(int(prob >= 0.5))
                if cfg.compute_bias_variance and variant_name in ('transfer_augmented','combined_all') and penalty=='l2':
                    def builder(): # helper function for bootstrapping
                        return LogisticRegression(penalty=penalty, C=tuned.C, solver=tuned.solver, max_iter=500)
                    _, b, v = bias_variance_bootstrap(builder, X_train, y_train, X_test[0], y_test, B=cfg.bootstrap_B)
                    if b is not None:
                        biases.append(b); variances.append(v)
                if cfg.compute_feature_importance and not apply_pca:
                    # this part gets the feature names and their coefficients to check importance
                    feat_names: List[str] = []
                    cat_len = 0
                    if 'cat' in pre.named_transformers_:
                        cat_trans = pre.named_transformers_['cat']
                        if isinstance(cat_trans, OneHotEncoder) and categorical:
                            
                            cats_attr = None # get the categories from the one-hot encoder
                            if hasattr(cat_trans, 'categories_'):
                                cats_attr = cat_trans.categories_
                            elif hasattr(cat_trans, 'categories'):
                                cats_attr = cat_trans.categories
                            if cats_attr is not None and len(cats_attr) > 0:
                                cats = list(cats_attr[0])
                                feat_names.extend([f"residue={c}" for c in cats])
                                cat_len = len(cats)
                    if 'num' in pre.named_transformers_:
                        num_pipe = pre.named_transformers_['num']
                        num_cols_used = list(getattr(num_pipe, 'feature_names_in_', [])) or numeric_cols
                        feat_names.extend(num_cols_used)
                    
                    coefs = tuned.coef_[0] # don't try to get importances if something went wrong
                    records = []
                    for idx_f, fname in enumerate(feat_names):
                        if idx_f >= len(coefs):
                            break
                        coef_std = coefs[idx_f]
                        coef_unstd = np.nan # try to un-scale the coefficient to see its original impact
                        if idx_f >= cat_len and 'num' in pre.named_transformers_:
                            scaler = pre.named_transformers_['num'].named_steps.get('sc')
                            if scaler is not None and hasattr(scaler,'scale_'):
                                num_index = idx_f - cat_len
                                if num_index < len(scaler.scale_):
                                    scale_val = scaler.scale_[num_index]
                                    if scale_val not in (0, None):
                                        coef_unstd = coef_std / scale_val
                        records.append({'feature': fname,'coef_std':coef_std,'coef_unstd':coef_unstd,'fold':fold_i})
                    if records:
                        fold_importances.append(pd.DataFrame(records))
            _vprint(cfg, f"[LOOCV] Variant={variant_name} Penalty={penalty} folds={len(y_probs)}")
            y_probs_arr = np.array(y_probs); y_preds_arr = np.array(y_preds)
            # save each prediction so we can analyze them later
            preds_df = pd.DataFrame({
                'index': labeled_df.index.tolist(),  # loocv order matches original row order
                'y_true': labeled_df[cfg.label_col].astype(int).values,
                'y_prob': y_probs_arr,
                'y_pred': y_preds_arr
            })
            prediction_records[(variant_name, penalty)] = preds_df
            acc = accuracy_score(y, y_preds_arr) # calculate all the metrics
            try:
                roc = roc_auc_score(y, y_probs_arr) if len(np.unique(y))==2 else None
            except ValueError:
                roc = None
            ll = log_loss(y, np.clip(y_probs_arr,1e-6,1-1e-6))
            pr, rc, f1, _ = precision_recall_fscore_support(y, y_preds_arr, average='binary', zero_division=0)
            
            if (4*pr + rc) > 0: # f2 score, cares more about recall
                f2 = (5 * pr * rc) / (4 * pr + rc)
            else:
                f2 = 0.0
            
            results.append(LOOResult(variant_name, penalty, acc, roc, ll, pr, rc, f1, f2, len(y),
                                      bias=np.mean(biases) if biases else None,
                                      variance=np.mean(variances) if variances else None,
                                      stage1_r2_mean=stage1_r2 if variant_name=='transfer_augmented' else None))
            if cfg.compute_feature_importance and fold_importances:
                all_fold_imp = pd.concat(fold_importances, ignore_index=True) # aggregate importances from all folds
                agg = all_fold_imp.groupby('feature').agg(
                    mean_coef_std=('coef_std','mean'),
                    std_coef_std=('coef_std','std'),
                    mean_coef_unstd=('coef_unstd','mean'),
                    nonzero_fraction=('coef_std', lambda x: (np.array(x)!=0).mean())
                ).sort_values('mean_coef_std', key=lambda s: s.abs(), ascending=False).reset_index()
                
                abs_sum = agg['mean_coef_std'].abs().sum() # calculate relative importance
                if abs_sum > 0:
                    agg['relative_importance'] = agg['mean_coef_std'].abs() / abs_sum
                else:
                    agg['relative_importance'] = 0.0
                importance_records[(variant_name, penalty)] = agg

    # now that we have all the raw loocv predictions, let's find the best threshold for f2 for each model
    beta_sq = 4.0  # for beta=2
    for r in results:
        key = (r.model_name, r.penalty)
        preds_df = prediction_records.get(key)
        if preds_df is None or preds_df['y_true'].nunique() < 2:
            continue
        y_true = preds_df['y_true'].values.astype(int)
        y_prob = preds_df['y_prob'].values
        
        grid = np.linspace(0.0, 1.0, 201) # check a bunch of thresholds
        candidates = np.unique(np.concatenate([grid, y_prob]))
        best_f2 = -1.0
        best_thr = 0.5
        best_prec = 0.0
        best_rec = 0.0
        for t in candidates: # loop through thresholds to find the best one
            y_hat = (y_prob >= t).astype(int)
            tp = np.sum((y_hat==1) & (y_true==1))
            fp = np.sum((y_hat==1) & (y_true==0))
            fn = np.sum((y_hat==0) & (y_true==1))
            if tp==0 and fp==0:
                prec = 0.0
            else:
                prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
            if prec==0 and rec==0:
                f2_val = 0.0
            else:
                denom = (beta_sq*prec + rec)
                f2_val = ( (1+beta_sq) * prec * rec / denom ) if denom>0 else 0.0
            # we want the highest f2, but if it's a tie, we prefer higher recall
            if f2_val > best_f2 or (abs(f2_val-best_f2)<1e-9 and rec>best_rec):
                best_f2 = f2_val; best_thr = t; best_prec = prec; best_rec = rec
        r.f2_opt = best_f2 if best_f2>=0 else None
        r.f2_opt_threshold = best_thr
        r.f2_opt_precision = best_prec
        r.f2_opt_recall = best_rec
    return results, importance_records, prediction_records, stage1_model


def results_to_dataframe(results: List[LOOResult]) -> pd.DataFrame:
    """converts the list of result objects to a nice pandas df."""
    rows = []
    for r in results:
        rows.append(dict(
            Variant=r.model_name,
            Penalty=r.penalty,
            Acc=r.accuracy,
            ROC_AUC=r.roc_auc,
            LogLoss=r.log_loss_value,
            Prec=r.precision,
            Recall=r.recall,
            F1=r.f1,
            F2=r.f2,
            F2_Opt=r.f2_opt,
            F2_Opt_Threshold=r.f2_opt_threshold,
            F2_Opt_Prec=r.f2_opt_precision,
            F2_Opt_Recall=r.f2_opt_recall,
            Bias=r.bias,
            Variance=r.variance,
            Stage1_R2=r.stage1_r2_mean
        ))
    return pd.DataFrame(rows)


def run_pipeline(data: Union[str, pd.DataFrame], config: Optional[PipelineConfig] = None,
                 return_details: bool = True,
                 mmgbsa_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """this is the main function to run the whole pipeline.

    args:
        data: path to a csv or a pandas dataframe.
        config: a PipelineConfig object. if you don't provide one, it uses the default.
        return_details: if true, it returns a dictionary with a bunch of extra stuff
                        like the feature groups, raw predictions, etc.
    returns:
        results_df: a dataframe with all the performance metrics.
        details: the dictionary with all the extra goodies.
    """
    cfg = config or PipelineConfig()
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    # if an mmgbsa directory is provided, load and merge those features
    if mmgbsa_dir is not None:
        mmgbsa_features = load_mmgbsa_features(mmgbsa_dir, cfg)
        if mmgbsa_features is not None:
            if 'position' in df.columns:
                before_cols = set(df.columns)
                df = df.merge(mmgbsa_features, on='position', how='left')
                new_cols = [c for c in df.columns if c not in before_cols]
                _vprint(cfg, f"[run_pipeline] Added {len(new_cols)} MM-GBSA feature columns from '{mmgbsa_dir}'")
            else:
                _vprint(cfg, "[run_pipeline] Skipping MM-GBSA merge: 'position' column missing in base data")
    df.columns = [c.strip() for c in df.columns] # clean up column names
    _vprint(cfg, f"[run_pipeline] Data shape: {df.shape}")
    groups = infer_groups(df, cfg)
    results, importances, prediction_records, stage1_model = run_loocv(df, groups, cfg)
    results_df = results_to_dataframe(results)
    if cfg.verbose:
        print("\nLOOCV Results (raw numbers):")
        print(results_df)
    details: Dict[str, Any] = {}
    if return_details:
        # re-run stage1 prediction on the whole dataset for the details dictionary
        cheap_all_df = df[groups['cheap_all']]
        stage1_preds, stage1_r2, _ = stage1_fit_predict(cheap_all_df, df[groups['expensive_all']], cfg)
        details = {
            'groups': groups,
            'raw_results': results,
            'stage1_preds': stage1_preds,
            'stage1_r2': stage1_r2,
            'feature_importances': importances,
            'stage1_model': stage1_model,
            'predictions_per_model': prediction_records,
            'augmented_df': df  # include the merged mm-gbsa columns if any
        }
    return results_df, details


def train_full_variant(df: pd.DataFrame, details: Dict[str, Any], cfg: PipelineConfig,
                       variant: str = 'transfer_augmented', penalty: str = 'l2') -> Dict[str, Any]:
    """trains a final model on all labeled data. good for making new predictions."""
    groups = details['groups']
    label_col = cfg.label_col
    labeled_df = df[~df[label_col].isna()].copy() # only train on labeled data
    y = labeled_df[label_col].astype(int).values
    if variant == 'transfer_augmented':
        stage1_model = details.get('stage1_model')
        if stage1_model is None: # rebuild the stage1 model if we don't have it
            cheap_all_df = df[groups['cheap_all']]
            _, _, stage1_model = stage1_fit_predict(cheap_all_df, df[groups['expensive_all']], cfg)
        preds_exp = stage1_model.predict(labeled_df[groups['cheap_all']])
        preds_exp_df = pd.DataFrame(preds_exp, columns=details['stage1_preds'].columns, index=labeled_df.index)
        X_df = pd.concat([labeled_df[groups['cheap_all']], preds_exp_df.add_prefix('TL_')], axis=1)
    elif variant == 'combined_all':
        X_df = labeled_df[groups['cheap_all'] + groups['expensive_all']]
    else:
        X_df = labeled_df[groups.get(variant, [])]
    categorical = [c for c in X_df.columns if c == 'residue'] # same preprocessing as in loocv
    numeric_cols = [c for c in X_df.columns if c not in categorical]
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ]), numeric_cols)
    ])
    pre.fit(X_df, y)
    X = pre.transform(X_df)
    tuned = tune_logreg(X, y, penalty=penalty) # tune and fit the final model
    tuned.fit(X, y)
    return { # package everything up into an "artifact" dictionary
        'variant': variant,
        'penalty': penalty,
        'model': tuned,
        'preprocessor': pre,
        'feature_columns': list(X_df.columns),
        'groups': groups,
        'stage1_model': details.get('stage1_model')
    }


def predict_new_sites(new_df: pd.DataFrame, artifact: Dict[str, Any], cfg: Optional[PipelineConfig] = None) -> pd.DataFrame:
    """predicts bioactivity for new data, especially rows that are missing expensive features.

    if the model is 'transfer_augmented', it uses the stage1 model to create the missing features first.
    """
    cfg = cfg or PipelineConfig()
    groups = artifact['groups']
    variant = artifact['variant']
    stage1_model = artifact.get('stage1_model')
    if variant == 'transfer_augmented':
        if stage1_model is None:
            raise ValueError("Stage1 model missing in artifact for transfer_augmented predictions")
        cheap_df = new_df[groups['cheap_all']].copy()
        preds = stage1_model.predict(cheap_df) # generate the expensive features
        # make sure the predicted columns match what the model was trained on
        tl_cols = [c for c in artifact['feature_columns'] if c.startswith('TL_')]
        preds_df = pd.DataFrame(preds, columns=[c.replace('TL_','') for c in tl_cols])
        preds_df = preds_df.add_prefix('TL_')
        X_input = pd.concat([cheap_df, preds_df], axis=1).reindex(columns=artifact['feature_columns'], fill_value=np.nan)
    else:
        X_input = new_df.reindex(columns=artifact['feature_columns'])
    pre = artifact['preprocessor']
    Xt = pre.transform(X_input) # preprocess the new data
    model = artifact['model']
    probs = model.predict_proba(Xt)[:,1] if len(model.classes_)==2 else np.full(len(new_df), np.nan) # get prediction probabilities
    if len(probs) != len(new_df):  # defensive fallback, just in case
        probs = probs[:len(new_df)]
    out = new_df.copy()
    out['Predicted_Bioactivity_Prob'] = probs
    return out


def predict_missing_md_sites(df: pd.DataFrame, details: Dict[str, Any], cfg: PipelineConfig,
                             variant: str = 'transfer_augmented', penalty: str = 'l2') -> pd.DataFrame:
    """finds rows that are missing md features and predicts their bioactivity."""
    groups = details['groups']
    md_cols = groups.get('md_only', [])
    if not md_cols:
        raise ValueError("No MD columns identified to determine missing MD sites.")
    # a row is missing md if all of its md columns are nan
    present_md_cols = [c for c in md_cols if c in df.columns]
    if not present_md_cols:
        missing_mask = np.ones(len(df), dtype=bool) # if no md columns exist at all, all rows are "missing"
    else:
        missing_mask = df[present_md_cols].isna().all(axis=1)
    missing_df = df[missing_mask].copy()
    if missing_df.empty:
        return pd.DataFrame()
    
    artifact = train_full_variant(df, details, cfg, variant=variant, penalty=penalty) # train the model
    
    preds = predict_new_sites(missing_df, artifact, cfg) # make predictions
    return preds


def predict_all_variants(df: pd.DataFrame, details: Dict[str, Any], cfg: PipelineConfig,
                         variants: Optional[List[str]] = None,
                         penalties: Tuple[str, ...] = ('l2','l1')) -> pd.DataFrame:
    """generates prediction probabilities for *every* row for *every* model variant.

    it trains a fresh model for each (variant, penalty) combo on all the labeled data,
    then applies it to all rows in the dataframe.
    """
    groups = details['groups']
    if variants is None:
        variants = ['sequence_only','structure_only','stability_only','md_only','docking_only','combined_all','transfer_augmented']
    base_cols = [c for c in cfg.sequence_id_cols if c in df.columns] # get id columns
    if not base_cols:
        base_cols = [df.index.name] if df.index.name else []
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)
    for variant in variants:
        if variant != 'transfer_augmented' and not groups.get(variant, []) and variant not in ('combined_all','transfer_augmented'):
            continue
        for pen in penalties:
            try:
                artifact = train_full_variant(df, details, cfg, variant=variant, penalty=pen)
                # use the same prediction logic for consistency
                preds_df = predict_new_sites(df, artifact, cfg)
                col_name = f"Prob_{variant}_{pen}"
                out[col_name] = preds_df['Predicted_Bioactivity_Prob'].values
            except Exception as e:
                # if something breaks, just fill with NaNs instead of crashing
                out[f"Prob_{variant}_{pen}"] = np.nan
                _vprint(cfg, f"[predict_all_variants] Failed {variant} {pen}: {e}")
    return out


def _ensure_dir(path: str): # little helper to make sure a directory exists
    if not path:
        return  # current directory is fine
    os.makedirs(path, exist_ok=True)


def save_report_table_png(df: pd.DataFrame, output_path: str, title: str = 'LOOCV Summary', font: str = 'Times New Roman'):
    """makes a pretty, publication-style png table from a dataframe.

    assumes the df cells are already formatted as strings. it tries to auto-size itself.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    
    rcParams['font.family'] = font # set the font (it'll fail silently if you don't have it)
    # figure out how wide each column needs to be based on the text inside
    col_char_widths = []
    for c in df.columns:
        max_len = max([len(str(c))] + [len(str(v)) for v in df[c].values])
        col_char_widths.append(max_len)
    
    char_w = 0.13  # this was tuned by trial and error
    table_width = sum(w*char_w for w in col_char_widths) + 0.6
    table_height = 0.45 * (len(df) + 1) + 0.8  # for rows + header + title
    fig, ax = plt.subplots(figsize=(max(6, table_width), max(2.5, table_height)))
    ax.axis('off')
    # build the table
    cell_text = df.values.tolist()
    col_labels = df.columns.tolist()
    the_table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    # set column widths proportionally
    total_chars = sum(col_char_widths)
    for i, (cw, col) in enumerate(zip(col_char_widths, df.columns)):
        rel = cw / total_chars if total_chars>0 else 1/len(df.columns)
        the_table.auto_set_column_width(i)
        for (row, col_idx), cell in the_table.get_celld().items():
            
            if col_idx == i: # match the column index
                cell.set_width(rel)
    # style the header
    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2f2f2f')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            
            if row % 2 == 0: # alternate row shading
                cell.set_facecolor('#f2f2f7')
            else:
                cell.set_facecolor('white')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)



def generate_visualizations(df: pd.DataFrame, results_df: pd.DataFrame, details: Dict[str, Any],
                            cfg: PipelineConfig, output_dir: str = 'plots', top_n_features: int = 15,
                            exclude_variants: Optional[set] = None) -> Dict[str,str]:
    """creates a bunch of plots to summarize performance, feature importance, etc.

    returns a dictionary mapping plot names to their file paths.
    """
    _ensure_dir(output_dir)
    plot_paths: Dict[str,str] = {}
    exclude_variants = exclude_variants or set()
    viz_results_df = results_df[~results_df['Variant'].isin(exclude_variants)].copy()
    # performance bar plot (accuracy, f1, f2)
    perf_melt = viz_results_df.melt(id_vars=['Variant','Penalty'], value_vars=['Acc','F1','F2','F2_Opt'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(10,5))
    sns.barplot(data=perf_melt, x='Variant', y='Score', hue='Metric', palette='viridis')
    plt.xticks(rotation=40, ha='right')
    plt.title('Model Performance (LOOCV)')
    plt.tight_layout()
    p1 = os.path.join(output_dir,'performance_bar.png')
    plt.savefig(p1, dpi=160); plt.close(); plot_paths['performance_bar']=p1

    # a chart focused on recall and f2
    if {'Recall','F2'}.issubset(viz_results_df.columns):
        value_vars = ['Recall','F2','F2_Opt'] if 'F2_Opt' in viz_results_df.columns else ['Recall','F2']
        recall_melt = viz_results_df.melt(id_vars=['Variant','Penalty'], value_vars=value_vars, var_name='Metric', value_name='Score')
        plt.figure(figsize=(10,5))
        sns.barplot(data=recall_melt, x='Variant', y='Score', hue='Metric', palette='crest')
        plt.xticks(rotation=40, ha='right')
        plt.ylim(0,1)
        plt.title('Recall-Prioritized Metrics (Recall & F2)')
        plt.tight_layout()
        p1b = os.path.join(output_dir,'recall_f2_bar.png')
        plt.savefig(p1b, dpi=160); plt.close(); plot_paths['recall_f2_bar']=p1b

    # curves showing f2 score at different thresholds
    pred_map = details.get('predictions_per_model', {}) if 'predictions_per_model' in details else {}
    if pred_map:
        for (variant, penalty), preds_df in list(pred_map.items())[:6]:  # limit to the first 6 to avoid too many plots
            if variant in exclude_variants or preds_df['y_true'].nunique()<2: continue
            y_true = preds_df['y_true'].values.astype(int)
            y_prob = preds_df['y_prob'].values
            thr_grid = np.linspace(0,1,101)
            f2_vals = []
            for t in thr_grid:
                y_hat = (y_prob>=t).astype(int)
                tp = ((y_hat==1)&(y_true==1)).sum(); fp = ((y_hat==1)&(y_true==0)).sum(); fn = ((y_hat==0)&(y_true==1)).sum()
                prec = tp/(tp+fp) if (tp+fp)>0 else 0
                rec = tp/(tp+fn) if (tp+fn)>0 else 0
                if prec==0 and rec==0:
                    f2v=0
                else:
                    denom = (4*prec + rec)
                    f2v = (5*prec*rec/denom) if denom>0 else 0
                f2_vals.append(f2v)
            plt.figure(figsize=(4,3))
            plt.plot(thr_grid, f2_vals, label='F2')
            # mark the optimal point we found earlier
            row_match = results_df[(results_df['Variant']==variant)&(results_df['Penalty']==penalty)]
            if not row_match.empty and not pd.isna(row_match.iloc[0].get('F2_Opt_Threshold')):
                t_opt = row_match.iloc[0]['F2_Opt_Threshold']; f2_opt = row_match.iloc[0]['F2_Opt']
                plt.axvline(t_opt, color='red', linestyle='--', linewidth=1)
                plt.scatter([t_opt],[f2_opt], color='red', s=25, zorder=5)
            plt.xlabel('Threshold'); plt.ylabel('F2'); plt.ylim(0,1)
            plt.title(f'F2 vs Threshold\n{variant} {penalty}')
            plt.tight_layout()
            fp_fx = os.path.join(output_dir, f'f2_threshold_{variant}_{penalty}.png')
            plt.savefig(fp_fx, dpi=140); plt.close(); plot_paths[f'f2_threshold_{variant}_{penalty}']=fp_fx
        # always generate these curves for the most important models (l2)
        for key_variant in ['combined_all','transfer_augmented']:
            pair = (key_variant,'l2')
            preds_df = pred_map.get(pair)
            if preds_df is None or preds_df['y_true'].nunique()<2:
                continue
            y_true = preds_df['y_true'].values.astype(int)
            y_prob = preds_df['y_prob'].values
            thr_grid = np.linspace(0,1,201)
            f2_vals = []
            for t in thr_grid:
                y_hat = (y_prob>=t).astype(int)
                tp = ((y_hat==1)&(y_true==1)).sum(); fp = ((y_hat==1)&(y_true==0)).sum(); fn = ((y_hat==0)&(y_true==1)).sum()
                prec = tp/(tp+fp) if (tp+fp)>0 else 0
                rec = tp/(tp+fn) if (tp+fn)>0 else 0
                if prec==0 and rec==0:
                    f2v=0
                else:
                    denom = (4*prec + rec)
                    f2v = (5*prec*rec/denom) if denom>0 else 0
                f2_vals.append(f2v)
            
            row_match = results_df[(results_df['Variant']==key_variant)&(results_df['Penalty']=='l2')]
            plt.figure(figsize=(4.5,3.2))
            plt.plot(thr_grid, f2_vals, label='F2')
            if not row_match.empty and not pd.isna(row_match.iloc[0].get('F2_Opt_Threshold')):
                t_opt = row_match.iloc[0]['F2_Opt_Threshold']; f2_opt = row_match.iloc[0]['F2_Opt']
                plt.axvline(t_opt, color='red', linestyle='--', linewidth=1, label=f'Opt {t_opt:.2f}')
                plt.scatter([t_opt],[f2_opt], color='red', s=28, zorder=5)
            plt.xlabel('Threshold'); plt.ylabel('F2'); plt.ylim(0,1)
            plt.title(f'F2 Threshold Curve\n{key_variant} (l2)')
            plt.legend(fontsize=7)
            plt.tight_layout()
            fp_key = os.path.join(output_dir, f'f2_threshold_{key_variant}_l2.png')
            plt.savefig(fp_key, dpi=150); plt.close(); plot_paths[f'f2_threshold_{key_variant}_l2']=fp_key

        # and for l1 too, if we have it
        for key_variant in ['combined_all','transfer_augmented']:
            pair = (key_variant,'l1')
            preds_df = pred_map.get(pair)
            if preds_df is None or preds_df['y_true'].nunique()<2:
                continue
            y_true = preds_df['y_true'].values.astype(int)
            y_prob = preds_df['y_prob'].values
            thr_grid = np.linspace(0,1,201)
            f2_vals = []
            for t in thr_grid:
                y_hat = (y_prob>=t).astype(int)
                tp = ((y_hat==1)&(y_true==1)).sum(); fp = ((y_hat==1)&(y_true==0)).sum(); fn = ((y_hat==0)&(y_true==1)).sum()
                prec = tp/(tp+fp) if (tp+fp)>0 else 0
                rec = tp/(tp+fn) if (tp+fn)>0 else 0
                if prec==0 and rec==0:
                    f2v=0
                else:
                    denom = (4*prec + rec)
                    f2v = (5*prec*rec/denom) if denom>0 else 0
                f2_vals.append(f2v)
            
            row_match = results_df[(results_df['Variant']==key_variant)&(results_df['Penalty']=='l1')]
            plt.figure(figsize=(4.5,3.2))
            plt.plot(thr_grid, f2_vals, label='F2')
            if not row_match.empty and not pd.isna(row_match.iloc[0].get('F2_Opt_Threshold')):
                t_opt = row_match.iloc[0]['F2_Opt_Threshold']; f2_opt = row_match.iloc[0]['F2_Opt']
                plt.axvline(t_opt, color='red', linestyle='--', linewidth=1, label=f'Opt {t_opt:.2f}')
                plt.scatter([t_opt],[f2_opt], color='red', s=28, zorder=5)
            plt.xlabel('Threshold'); plt.ylabel('F2'); plt.ylim(0,1)
            plt.title(f'F2 Threshold Curve\n{key_variant} (l1)')
            plt.legend(fontsize=7)
            plt.tight_layout()
            fp_key_l1 = os.path.join(output_dir, f'f2_threshold_{key_variant}_l1.png')
            plt.savefig(fp_key_l1, dpi=150); plt.close(); plot_paths[f'f2_threshold_{key_variant}_l1']=fp_key_l1

    # bias vs variance scatter plot
    bv = viz_results_df.dropna(subset=['Bias','Variance'])
    if not bv.empty:
        plt.figure(figsize=(6,5))
        sns.scatterplot(data=bv, x='Bias', y='Variance', hue='Variant', style='Penalty', s=90)
        for _,r in bv.iterrows():
            plt.text(r.Bias, r.Variance, r.Variant[:4], fontsize=8) # label the points
        plt.title('Bias-Variance Tradeoff')
        plt.tight_layout()
        p2 = os.path.join(output_dir,'bias_variance.png')
        plt.savefig(p2, dpi=160); plt.close(); plot_paths['bias_variance']=p2

    # feature importance plots
    importances = details.get('feature_importances', {})
    for (variant, penalty), imp_df in importances.items():
        if imp_df.empty or variant in exclude_variants: continue
        top_imp = imp_df.head(top_n_features)
        plt.figure(figsize=(7, max(3, 0.35*len(top_imp))))
        
        sns.barplot(data=top_imp, y='feature', x='mean_coef_std', hue='feature', dodge=False, palette='magma', legend=False)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.title(f'Feature Importance (std coeff) {variant} {penalty}')
        plt.tight_layout()
        fp = os.path.join(output_dir, f'feat_importance_{variant}_{penalty}.png')
        plt.savefig(fp, dpi=160); plt.close(); plot_paths[f'feat_importance_{variant}_{penalty}']=fp

    # distribution plots for the top features overall
    all_top_feats = set()
    for (variant, _), imp in importances.items():
        if variant in exclude_variants: continue
        all_top_feats.update(imp.head(int(top_n_features/3)).feature.tolist())
    
    all_top_feats = [f for f in all_top_feats if not f.startswith('residue=') and f in df.columns] # filter out one-hot encoded features
    
    all_top_feats = all_top_feats[:min(10, len(all_top_feats))] # limit to a reasonable number
    if all_top_feats and cfg.label_col in df.columns:
        lab_col = cfg.label_col
        sub = df[[lab_col]+all_top_feats].copy()
        melted = sub.melt(id_vars=[lab_col], var_name='Feature', value_name='Value')
        plt.figure(figsize=(12, 0.8*len(all_top_feats)+3))
        sns.boxplot(data=melted, x='Value', y='Feature', hue=lab_col, orient='h')
        plt.title('Top Feature Distributions by Bioactivity')
        plt.tight_layout()
        fp = os.path.join(output_dir,'top_feature_distributions.png')
        plt.savefig(fp, dpi=160); plt.close(); plot_paths['top_feature_distributions']=fp

    # correlation heatmap for cheap features
    cheap_cols = details['groups']['cheap_all']
    cheap_numeric = [c for c in cheap_cols if c in df.columns and df[c].dtype != object]
    if len(cheap_numeric) >= 3:
        corr = df[cheap_numeric].corr().replace([np.inf,-np.inf], np.nan).fillna(0)
        plt.figure(figsize=(min(12, 0.6*len(corr)), min(10, 0.6*len(corr))))
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title('Cheap Feature Correlation Heatmap')
        plt.tight_layout()
        fp = os.path.join(output_dir,'cheap_feature_correlation.png')
        plt.savefig(fp, dpi=160); plt.close(); plot_paths['cheap_feature_correlation']=fp

    # scatter plots to see how well the stage1 model did
    stage1_preds = details.get('stage1_preds')
    if stage1_preds is not None and not stage1_preds.empty:
        expensive_cols = details['groups']['expensive_all']
        # pick up to 6 columns to plot
        actual_df = df[expensive_cols]
        valid_cols = [c for c in expensive_cols if c in actual_df.columns and actual_df[c].notna().sum()>5]
        for c in valid_cols[:6]:
            plt.figure(figsize=(4,4))
            plt.scatter(actual_df[c], stage1_preds[c], alpha=0.6)
            
            try: # compute r^2 for the plot title
                mask = actual_df[c].notna()
                ss_res = ((actual_df.loc[mask,c]-stage1_preds.loc[mask,c])**2).sum()
                ss_tot = ((actual_df.loc[mask,c]-actual_df.loc[mask,c].mean())**2).sum()
                r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
            except Exception:
                r2 = np.nan
            plt.xlabel('Actual'); plt.ylabel('Predicted')
            plt.title(f'Stage1 {c} R2={r2:.2f}')
            plt.tight_layout()
            fp = os.path.join(output_dir,f'stage1_{c}_scatter.png')
            plt.savefig(fp, dpi=150); plt.close(); plot_paths[f'stage1_{c}']=fp

    # save a big summary table of all feature importances
    if importances:
        rows = []
        for (variant, penalty), imp_df in importances.items():
            for _, r in imp_df.iterrows():
                rows.append({
                    'Variant': variant,
                    'Penalty': penalty,
                    'Feature': r['feature'],
                    'MeanStdCoef': r['mean_coef_std'],
                    'StdStdCoef': r['std_coef_std'],
                    'MeanUnstdCoef': r['mean_coef_unstd'],
                    'NonZeroFraction': r['nonzero_fraction']
                })
        fi_summary = pd.DataFrame(rows)
        fp = os.path.join(output_dir,'feature_importance_summary.tsv')
        fi_summary.to_csv(fp, sep='\t', index=False)
        plot_paths['feature_importance_summary']=fp
        # relative importance bar plots for the key models
        for penalty_target in ['l2','l1']:
            key_pairs = [p for p in importances.keys() if p[0] in ('combined_all','transfer_augmented') and p[1]==penalty_target]
            for variant, penalty in key_pairs:
                imp_df = importances[(variant, penalty)].copy()
                if 'relative_importance' in imp_df.columns:
                    top_imp = imp_df.head(top_n_features)
                    plt.figure(figsize=(8, max(3, 0.4*len(top_imp))))
                    sns.barplot(data=top_imp, x='relative_importance', y='feature', hue='feature', dodge=False, palette='cubehelix', legend=False)
                    plt.title(f'Relative Importance {variant} {penalty}')
                    plt.xlabel('Relative Importance (|coef| normalized)')
                    plt.tight_layout()
                    fp2 = os.path.join(output_dir, f'relative_importance_{variant}_{penalty}.png')
                    plt.savefig(fp2, dpi=170); plt.close(); plot_paths[f'relative_importance_{variant}_{penalty}']=fp2

    # roc and pr curves using the aggregated loocv predictions
    pred_map = details.get('predictions_per_model', {})
    if pred_map:
        # roc curves
        plt.figure(figsize=(7,6))
        any_curve = False
        for (variant, penalty), preds_df in pred_map.items():
            if variant in exclude_variants or preds_df['y_true'].nunique() < 2: continue
            fpr, tpr, _ = roc_curve(preds_df['y_true'], preds_df['y_prob'])
            auc_val = roc_auc_score(preds_df['y_true'], preds_df['y_prob'])
            plt.plot(fpr, tpr, label=f"{variant[:12]} {penalty} AUC={auc_val:.2f}")
            any_curve = True
        if any_curve:
            plt.plot([0,1],[0,1],'--', color='grey') # the random chance line
            plt.xlabel('FPR'); plt.ylabel('TPR')
            plt.title('ROC Curves (LOOCV)')
            plt.legend(fontsize=8)
            plt.tight_layout()
            fp = os.path.join(output_dir,'roc_curves.png')
            plt.savefig(fp, dpi=170); plt.close(); plot_paths['roc_curves']=fp
        # pr curves
        plt.figure(figsize=(7,6))
        any_curve = False
        for (variant, penalty), preds_df in pred_map.items():
            if variant in exclude_variants or preds_df['y_true'].nunique() < 2: continue
            precision, recall, _ = precision_recall_curve(preds_df['y_true'], preds_df['y_prob'])
            ap = average_precision_score(preds_df['y_true'], preds_df['y_prob'])
            plt.plot(recall, precision, label=f"{variant[:12]} {penalty} AP={ap:.2f}")
            any_curve = True
        if any_curve:
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title('Precision-Recall Curves (LOOCV)')
            plt.legend(fontsize=8)
            plt.tight_layout()
            fp = os.path.join(output_dir,'pr_curves.png')
            plt.savefig(fp, dpi=170); plt.close(); plot_paths['pr_curves']=fp
        # calibration plot for the main models
        for key_variant in ['combined_all','transfer_augmented']:
            pair = next(((v,p) for (v,p) in pred_map.keys() if v==key_variant and p=='l2'), None)
            if not pair: continue
            preds_df = pred_map[pair]
            if preds_df['y_true'].nunique()<2: continue
            probs = np.clip(preds_df['y_prob'],1e-6,1-1e-6)
            bins = np.linspace(0,1,8)
            inds = np.digitize(probs, bins)-1
            df_cal = pd.DataFrame({'prob':probs,'y':preds_df['y_true'],'bin':inds})
            cal = df_cal.groupby('bin').agg(mean_prob=('prob','mean'), actual=('y','mean'), count=('y','size'))
            cal = cal[cal['count']>=2] # only use bins with a few samples
            if cal.empty: continue
            plt.figure(figsize=(5,5))
            plt.plot([0,1],[0,1],'--', color='grey') # perfect calibration line
            plt.plot(cal['mean_prob'], cal['actual'], marker='o')
            plt.xlabel('Predicted Probability'); plt.ylabel('Observed Frequency')
            plt.title(f'Calibration: {key_variant} (l2)')
            plt.tight_layout()
            fp = os.path.join(output_dir,f'calibration_{key_variant}.png')
            plt.savefig(fp, dpi=160); plt.close(); plot_paths[f'calibration_{key_variant}']=fp
    # confusion matrices for the top 4 models by accuracy
    top_models = viz_results_df.sort_values('Acc', ascending=False).head(4)[['Variant','Penalty']].apply(tuple, axis=1).tolist()
    for pair in top_models:
        if pair not in pred_map: continue
        preds_df = pred_map[pair]
        if preds_df['y_true'].nunique()<2: continue
        cm = confusion_matrix(preds_df['y_true'], preds_df['y_pred'])
        plt.figure(figsize=(3.5,3.2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title(f'CM {pair[0][:10]} {pair[1]}')
        plt.tight_layout()
        fp = os.path.join(output_dir,f'cm_{pair[0]}_{pair[1]}.png')
        plt.savefig(fp, dpi=170); plt.close(); plot_paths[f'cm_{pair[0]}_{pair[1]}']=fp

    # plot predicted probabilities against residue position, if 'position' column exists
    if 'position' in df.columns:
            try:
                # generate predictions for all variants (just once)
                all_variant_preds = predict_all_variants(df, details, cfg)
                # just grab the probability columns
                prob_cols = [c for c in all_variant_preds.columns if c.startswith('Prob_')]
                if prob_cols:
                    long_records = []
                    pos_series = df['position'].values
                    for col in prob_cols:
                        variant_pen = col.replace('Prob_','')  # e.g., sequence_only_l2
                        parts = variant_pen.rsplit('_', 1)
                        if len(parts)==2:
                            variant_name, pen = parts
                        else:
                            variant_name, pen = variant_pen, 'na'
                        vals = all_variant_preds[col].values
                        for p,v in zip(pos_series, vals):
                            if variant_name in exclude_variants: continue
                            long_records.append({'position':p,'prob':v,'variant':variant_name,'penalty':pen})
                    long_df = pd.DataFrame(long_records)
                    # combined bar plot for just l2 models
                    l2_df = long_df[long_df['penalty']=='l2']
                    if not l2_df.empty:
                        plt.figure(figsize=(max(12, len(l2_df['position'].unique())*0.08),5))
                        sns.barplot(data=l2_df, x='position', y='prob', hue='variant', dodge=True, palette='tab10', edgecolor='none')
                        plt.ylabel('Predicted Bioactivity Prob (L2)')
                        plt.title('Predicted Bioactivity vs Position (Bars, L2 variants)')
                        plt.xticks(rotation=90, fontsize=6)
                        plt.tight_layout()
                        fp = os.path.join(output_dir,'position_probabilities_l2.png')
                        plt.savefig(fp, dpi=170); plt.close(); plot_paths['position_probabilities_l2']=fp
                    # facet grid with plots for each variant
                    subset_variants = long_df['variant'].unique().tolist()
                    if len(subset_variants)>0:
                        fg_df = long_df.copy()
                        g = sns.FacetGrid(fg_df, col='variant', hue='penalty', col_wrap=4, sharey=True, height=2.2, aspect=1.4)
                        for ax,(variant_name, subdf) in zip(g.axes.flatten(), fg_df.groupby('variant')):
                            sns.barplot(data=subdf, x='position', y='prob', hue='penalty', dodge=True, palette='Set2', ax=ax, edgecolor='none')
                            ax.set_title(variant_name, fontsize=9)
                            ax.tick_params(axis='x', labelrotation=90, labelsize=5)
                            if ax != g.axes.flatten()[0] and ax.legend_:
                                ax.legend_.remove() # remove individual legends
                        
                        first_ax = g.axes.flatten()[0] # make one global legend
                        if first_ax.legend_:
                            handles, labels = first_ax.get_legend_handles_labels()
                            g.fig.legend(handles, labels, loc='upper right', fontsize=8)
                        for ax in g.axes.flatten():
                            ax.set_xlabel('Pos'); ax.set_ylabel('Prob')
                        plt.tight_layout()
                        fp2 = os.path.join(output_dir,'position_probabilities_facets.png')
                        plt.savefig(fp2, dpi=160); plt.close(); plot_paths['position_probabilities_facets']=fp2
            except Exception as e:
                _vprint(cfg, f"[generate_visualizations] position plot failed: {e}")
    return plot_paths


def main():
    parser = argparse.ArgumentParser(description="Bioactivity Transfer Learning Pipeline (CLI)")
    parser.add_argument('--csv', required=True, help='Path to CSV data file.')
    parser.add_argument('--save_report', default='report.tsv', help='Output summary TSV file.')
    parser.add_argument('--no-bias-variance', action='store_true', help='Disable bootstrap bias/variance estimation.')
    parser.add_argument('--verbose', action='store_true', help='Verbose progress output.')
    parser.add_argument('--bootstrap-B', type=int, default=100, help='Bootstrap iterations for bias/variance.')
    parser.add_argument('--predict-missing-md', default=None, help='If set, write predictions for rows lacking MD features to this TSV file.')
    parser.add_argument('--predict-all-variants', default=None, help='If set, write per-variant probability predictions for ALL rows to this TSV file.')
    parser.add_argument('--mmgbsa-dir', default=None, help='Directory containing prime_mmgbsa_[RES][POS]-out.csv (recursively searched). Averaged columns merged as docking features.')
    args = parser.parse_args()

    cfg = PipelineConfig(verbose=args.verbose, compute_bias_variance=not args.no_bias_variance, bootstrap_B=args.bootstrap_B)
    print("Install deps (if needed): pip install numpy pandas scikit-learn scipy joblib tabulate")
    results_df, details = run_pipeline(args.csv, config=cfg, return_details=True, mmgbsa_dir=args.mmgbsa_dir)
    
    # don't show these in the final table, they're usually not the best performers and clutter things up
    exclude_variants_display = {'sequence_only','structure_only'}
    filtered_results_df = results_df[~results_df['Variant'].isin(exclude_variants_display)].copy()
    if filtered_results_df.empty:
        filtered_results_df = results_df.copy()  # fallback to avoid an empty table if all variants were excluded
    filtered_results_df.to_csv(args.save_report, sep='\t', index=False)
    print(f"Saved filtered report (excluding {exclude_variants_display}) to {args.save_report}")
    # make the table look nice for the console output
    display_df = filtered_results_df.copy()
    numeric_cols = [c for c in display_df.columns if display_df[c].dtype != object]
    for c in numeric_cols:
        display_df[c] = display_df[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else 'NA')
    # save a nice looking png of the results table
    try:
        png_out = os.path.splitext(args.save_report)[0] + '.png'
        print(f"[DEBUG] Creating table PNG at {png_out} with shape {display_df.shape}")
        save_report_table_png(display_df, png_out, title='LOOCV Summary (Filtered)')
        print(f"Saved table PNG to {png_out}")
    except Exception as e:
        print(f"Failed to create table PNG: {e}")
    print("\nLOOCV Results:")
    print(tabulate(display_df, headers='keys', tablefmt='github'))
    print("\nCritical Notes:")
    print("- Compare transfer_augmented vs combined_all to see if the real expensive features are worth it.")
    print("- L1 (lasso) is only useful for feature selection if its performance is close to L2 (ridge).")
    print("- The sample size is really small, so take all results with a grain of salt. More data would be better.")
    print("- A low Stage1_R2 means the surrogate model is weak. Maybe need better cheap features.")
    try:
        # use the dataframe that might have mm-gbsa features merged in
        viz_df = details.get('augmented_df') if isinstance(details, dict) else None
        if viz_df is None:
            viz_df = pd.read_csv(args.csv)
        plot_paths = generate_visualizations(viz_df, results_df, details, cfg, output_dir='plots', exclude_variants=exclude_variants_display)
        print(f"Generated {len(plot_paths)} plots/files under ./plots")
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    # optionally, predict for sites missing md data
    if args.predict_missing_md:
        try:
            df_full = pd.read_csv(args.csv)
            missing_preds = predict_missing_md_sites(df_full, details, cfg, variant='transfer_augmented', penalty='l2')
            if not missing_preds.empty:
                missing_preds.to_csv(args.predict_missing_md, sep='\t', index=False)
                print(f"Saved predictions for missing-MD rows to {args.predict_missing_md} (n={len(missing_preds)})")
            else:
                print("No rows identified as missing all MD features.")
        except Exception as e:
            print(f"Failed to predict missing MD sites: {e}")
    if args.predict_all_variants:
        try:
            df_full = pd.read_csv(args.csv)
            all_preds = predict_all_variants(df_full, details, cfg)
            # add the true label back in if it exists, for comparison
            if cfg.label_col in df_full.columns:
                all_preds[cfg.label_col] = df_full[cfg.label_col].values
            all_preds.to_csv(args.predict_all_variants, sep='\t', index=False)
            print(f"Saved full per-variant predictions to {args.predict_all_variants}")
        except Exception as e:
            print(f"Failed to produce all-variant predictions: {e}")


if __name__ == '__main__':
    main()