import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.calibration import calibration_curve

# SET SEABORN STYLE
sns.set_style("whitegrid")
sns.set_color_codes()

# GENEREL EVALUATION
def evaluate_classification_results(y_true, y_prob):
    target_names = ['Negativ', 'Positiv']
    # ROC Curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.clf()
    plt.close('all')
    plt.style.use('seaborn-paper')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(fpr,tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Total')
    ax1.legend(loc="lower right")

    ax2.plot(fpr,tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
    ax2.set_xlim([0.0, 0.1])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC - Operating Area')
    ax2.legend(loc="lower right")

    average_precision = metrics.average_precision_score(y_true=y_true, y_score=y_prob)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,y_prob)
    ax3.step(recall, precision, color='b', alpha=0.2, where='post')
    ax3.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlim([0.0, 1.0])
    ax3.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    return {"Average precision": average_precision, "AUC": roc_auc}
def plot_hist(y_prob, bins=10):
    sns.distplot(pd.Series(y_prob, name="Estimated probability"), bins=bins)
def plot_calibration_curve(y_true, y_probs, bins):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for legend, y_prob in y_probs.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob['y_prob'], n_bins=bins)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=y_prob['legend'])
        ax2.hist(y_prob['y_prob'], range=(0, 1), bins=10, label=y_prob['legend'], histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.show()
def plot_loss_curve(loss_curve, batch_size):
    loss_df = pd.DataFrame({'Loss': loss_curve, 'Steps': np.arange(len(loss_curve)) // batch_size})
    sns.lineplot(data=loss_df, x='Steps', y="Loss")
def save_loss_curve(loss_curve, batch_size):
    loss_df = pd.DataFrame({'Loss': loss_curve, 'Steps': np.arange(len(loss_curve)) // batch_size})
    plt.clf()
    sns.lineplot(data=loss_df, x='Steps', y="Loss").figure.savefig('loss.png')
# PR / AUC CURVE
# Callable functions: pr_curve, plot_precision_recall_vs_threshold, precision_recall_threshold, multiple_pr_curves,
# multiple_auc_curves
def pr_curve(y_true, y_prob):
    fig, ax1 = plt.subplots()
    average_precision = metrics.average_precision_score(y_true=y_true, y_score=y_prob)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,y_prob)
    ax1.step(recall, precision, color='b', alpha=0.1, where='post')
    ax1.fill_between(recall, precision, step='post', alpha=0.1, color='b')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlim([0.0, 1.0])
    ax1.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]
def plot_precision_recall_vs_threshold(y_true, y_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
def precision_recall_threshold(y_true, y_prob, t=0.5):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_prob, t)
    print(pd.DataFrame(confusion_matrix(y_true, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))

    # plot the curve
    plt.step(recall, precision, color='b', alpha=0.1, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.1, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP')

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(recall[close_default_clf], precision[close_default_clf], '^', c='k',
             markersize=15)
def multiple_pr_curves(y_true, y_probs):
    '''
    Her er et eksempel på hvordan input forberedes
    y_probs =   {
                'y_prob_1' : {'legend':'Keras DNN', 'y_prob':y_prob_1},
                'y_prob_2' : {'legend':'RandomForest', 'y_prob':y_prob_2[:,1]},
                'y_prob_3': {'legend': 'Logistic Regression', 'y_prob': y_prob_3[:,1]},
                'y_prob_4': {'legend': 'Gradient boosting', 'y_prob': y_prob_4[:,1]},
                'y_prob_5': {'legend': '10000 Units', 'y_prob': y_prob_5[:,1]}
            }
    '''
    for legend, y_prob in y_probs.items():
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob['y_prob'], pos_label=1)
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob['y_prob'])
        average_precision = metrics.average_precision_score(y_true=y_true, y_score=y_prob['y_prob'])
        plt.plot(recall, precision, label='{0}, AP: {1:.3f} AUROC: {2:.3f}'.format(y_prob['legend'], average_precision, metrics.auc(fpr, tpr)))
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP')
    print(y_prob['legend'])
def multiple_auc_curves(y_true, y_probs):
    '''
     Her er et eksempel på hvordan input forberedes
    y_probs =   {
                'y_prob_1' : {'legend':'Keras DNN', 'y_prob':y_prob_1},
                'y_prob_2' : {'legend':'RandomForest', 'y_prob':y_prob_2[:,1]},
                'y_prob_3': {'legend': 'Logistic Regression', 'y_prob': y_prob_3[:,1]},
                'y_prob_4': {'legend': 'Gradient boosting', 'y_prob': y_prob_4[:,1]},
                'y_prob_5': {'legend': '10000 Units', 'y_prob': y_prob_5[:,1]}
            }
    '''
    for legend, y_prob in y_probs.items():
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob['y_prob'], pos_label=1)
        average_precision = metrics.average_precision_score(y_true=y_true, y_score=y_prob['y_prob'])
        plt.plot(fpr, tpr, lw=1.5, label='{0}, AP: {1:.3f} AUROC: {2:.3f}'.format(y_prob['legend'], average_precision, metrics.auc(fpr, tpr)))
    plt.legend()
    plt.xlabel('False positive rate / Sensitivity')
    plt.ylabel('False negative rate / 1-Specificity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Receiver operating characteristic (ROC) curve')
    print(y_prob['legend'])

# DCA
# Callable functions: decision_curve_analysis, plot_dca
def initialize_result_dataframes(event_rate, thresh_lo, thresh_hi, thresh_step, harm, combined):
    """Initializes the net_benefit and interventions_avoided dataFrames for the
    given threshold boundaries and event rate
    Parameters
    ----------
    event_rate : float
    thresh_lo : float
    thresh_hi : float
    thresh_step : float
    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        properly initialized net_benefit, interventions_avoided dataframes
    """
    # initialize threshold series for each dataFrame
    net_benefit = pd.Series(frange(thresh_lo, thresh_hi + thresh_step, thresh_step),
                            name='threshold')
    interventions_avoided = pd.DataFrame(net_benefit)

    # construct 'all' and 'none' columns for net_benefit
    net_benefit_all = event_rate - (1 - event_rate) * (net_benefit / (1 - net_benefit))
    net_benefit_all.name = 'all'
    net_benefit_none_combined = (1 - event_rate) - (event_rate * (1 - net_benefit) / net_benefit)

    if combined == True:
        net_benefit = pd.concat([net_benefit, net_benefit_all], axis=1)
        net_benefit['none'] = net_benefit_none_combined
        net_benefit['prevalence'] = 1

    if combined == False:
        net_benefit = pd.concat([net_benefit, net_benefit_all], axis=1)
        net_benefit['none'] = 0
        net_benefit['prevalence'] = event_rate
    return net_benefit, interventions_avoided
def calc_tf_positives(data, outcome, predictor, net_benefit_threshold, j):
    """Calculate the number of true/false positives for the given parameters
    Parameters
    ----------
    data : pd.DataFrame
        the data set to analyze
    outcome : str
        the column of the data frame to use as the outcome
    predictor : str
        the column to use as the predictor for this calculation
    net_benefit_threshold : pd.Series
        the threshold column of the net_benefit data frame
    j : int
        the index in the net_benefit data frame to use
    Returns
    -------
    tuple(float, float)
        the number of true positives, false positives
    """
    true_positives = false_positives = 0
    # create a filter mask
    filter_mask = data[predictor] >= net_benefit_threshold[j]
    filter_mask_sum = filter_mask.sum()
    if filter_mask_sum == 0:
        pass
    else:
        # get all outcomes where the filter_mask is 'True'
        filtered_outcomes = map(lambda x, y: x if y == True else np.nan,
                                data[outcome], filter_mask)
        filtered_outcomes = [outcome for outcome in filtered_outcomes
                             if outcome is not np.nan]  # drop all NaN values
        true_positives = mean(filtered_outcomes) * filter_mask_sum
        false_positives = (1 - mean(filtered_outcomes)) * filter_mask_sum

    return true_positives, false_positives
def calc_tf_negative(data, outcome, predictor, net_benefit_threshold, j):
    """Calculate the number of true/false positives for the given parameters
    Parameters
    ----------
    data : pd.DataFrame
        the data set to analyze
    outcome : str
        the column of the data frame to use as the outcome
    predictor : str
        the column to use as the predictor for this calculation
    net_benefit_threshold : pd.Series
        the threshold column of the net_benefit data frame
    j : int
        the index in the net_benefit data frame to use
    Returns
    -------
    tuple(float, float)
        the number of true positives, false positives
    """
    true_negatives = false_negatives = 0
    # create a filter mask
    filter_mask = data[predictor] < net_benefit_threshold[j]
    filter_mask_sum = filter_mask.sum()
    if filter_mask_sum == 0:
        pass
    else:
        # get all outcomes where the filter_mask is 'True'

        filtered_outcomes = map(lambda x, y: x if y == True else np.nan,
                                data[outcome], filter_mask)
        filtered_outcomes = [outcome for outcome in filtered_outcomes
                             if outcome is not np.nan]  # drop all NaN values
        false_negatives = mean(filtered_outcomes) * filter_mask_sum
        true_negatives = (1 - mean(filtered_outcomes)) * filter_mask_sum

    return true_negatives, false_negatives
def calculate_net_benefit(index, net_benefit_threshold, harm, combined, true_positives, false_positives, true_negatives, false_negatives, num_observations):
    """Calculates the net benefit for an index within the construction of net_benefit
    loop
    This function calculates the net_benefit for a particular predictor at the given index, however
    the predictor doesn't need to be supplied to this function and should already be determined
    from the true/false positive calculation
    NOTE: true/false positives should be generated by using the calc_tf_positives
    function for the predictor of interest
    Parameters
    ----------
    net_benefit_threshold : pd.Series
        the 'threshold' column of the net_benefit dataframe for the analysis
    harm : float
        the harm value for the predictor
    true_positives : float
        the number of true positives for the given predictor
    false_positives : float
        the number of false positives for the given predictor
    num_observations : int
        the number of observations in the data set
    index : int
        the index in the Series to compute for
    Returns
    -------
    float
        value for the net benefit at `index` for the predictor
    """
    # normalize the true/false positives by the number of observations
    tp_norm = true_positives / num_observations
    fp_norm = false_positives / num_observations
    tn_norm = true_negatives / num_observations
    fn_norm = false_negatives / num_observations

    # calculate the multiplier for the false positives
    multiplier_treated = net_benefit_threshold[index] / (1 - net_benefit_threshold[index])
    multiplier_untreated = (1 - net_benefit_threshold[index]) / net_benefit_threshold[index]

    #print('tp_norm: {}, fp_norm: {}, tn_norm: {}, fn_norm: {}, sum: : {}, multiplier_treated: {}, multiplier_untreated: {}'.format(tp_norm, fp_norm, tn_norm, fn_norm, sum([tp_norm, fp_norm, tn_norm, fn_norm]), multiplier_treated, multiplier_untreated))
    #print('Combined: {}'.format(tp_norm + tn_norm - (fp_norm * multiplier_treated) - (fn_norm * multiplier_untreated)))
    #print('Tærskelværdi: {}, Positive led: {}, fp led: {}, fn led: {}'.format(net_benefit_threshold[index], tp_norm + tn_norm, fp_norm * multiplier_treated, fn_norm * multiplier_untreated))

    if combined == True:
        return tp_norm + tn_norm - (fp_norm * multiplier_treated) - (fn_norm * multiplier_untreated)
    if combined == False:
        return tp_norm - (fp_norm * multiplier_treated)
def frange(start, stop, step):
    """Generator that can create ranges of floats
    Credit: http://stackoverflow.com/questions/7267226/range-for-floats
    Parameters
    ----------
    start : float
       the minimum value of the range
    stop : float
        the maximum value of the range
    step : float
        the step between values in the range
    Yields
    ------
    float
        the next number in the range `start` to `stop`-`step`
    """
    while start < stop:
        yield start
        start += step
def mean(iterable):
    """Calculates the mean of the given iterable
    Parameters
    ----------
    iterable: int, float
        an iterable of ints or floats
    Returns
    -------
    float
        the arithmetic mean of the iterable
    """
    return sum(iterable) / len(iterable)
def dca(data, outcome, predictors, thresh_lo=0.01, thresh_hi=0.99, thresh_step=0.01, probabilities=None, harms=None, combined=False, intervention_per=100, smooth_results=False, lowess_frac=0.10):
    """Performs decision curve analysis on the input data set

    Parameters
    ----------
    data : pd.DataFrame
        the data set to analyze
    outcome : str
        the column of the data frame to use as the outcome
        this must be coded as a boolean (T/F) or (0/1)
    predictors : str OR list(str)
        the column(s) that will be used to predict the outcome
    thresh_lo : float
        lower bound for threshold probabilities (defaults to 0.01)
    thresh_hi : float
        upper bound for threshold probabilities (defaults to 0.99)
    thresh_step : float
        step size for the set of threshold probabilities [x_start:x_stop]
    probability : bool or list(bool)
        whether the outcome is coded as a probability
        probability must have the same length as the predictors list
    harm : float or list(float)
        the harm associated with each predictor
        harm must have the same length as the predictors list
    intervention_per : int
        interventions per `intervention_per` patients
    smooth_results : bool
        use lowess smoothing to smooth the result data series
    lowess_frac : float
        the fraction of the data used when estimating each endogenous value

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        A tuple of length 2 with net_benefit, interventions_avoided
        net_benefit : TODO
        interventions_avoided : TODO
    """
    # calculate useful constants for the net benefit calculation
    num_observations = len(data[outcome])  # number of observations in data set
    event_rate = mean(data[outcome])  # the rate at which the outcome happens

    # create DataFrames for holding results
    net_benefit, interventions_avoided = \
        initialize_result_dataframes(event_rate, thresh_lo, thresh_hi, thresh_step, harm=harms, combined=combined)

    for i, predictor in enumerate(predictors):  # for each predictor
        net_benefit[predictor] = np.nan  # initialize new column of net_benefits

        for j in range(0, len(net_benefit['threshold'])):  # for each threshold value
            # calculate true/false positives
            true_positives, false_positives = \
                calc_tf_positives(data, outcome, predictor,
                                  net_benefit['threshold'], j)
            true_negatives, false_negatives = \
                calc_tf_negative(data, outcome, predictor,
                                  net_benefit['threshold'], j)
            # calculate net benefit
            net_benefit_value = \
                calculate_net_benefit(j,net_benefit['threshold'], harms, combined,
                                      true_positives, false_positives, true_negatives, false_negatives,
                                      num_observations)
            net_benefit.set_value(j, predictor, net_benefit_value)

    return net_benefit
def decision_curve_analysis(y_true, y_prob, convert_label_to_zero=False,thresh_lo=0.01, thresh_step=0.01, harms=0, combined=False):
    # Input skal være to numpy arrays som vender nedad, altså med shape: (:,)
    if convert_label_to_zero is not False:
        y_true = [0 if y==convert_label_to_zero else y for y in y_true]
    data = [('outcome', y_true), ('model', y_prob),]
    data = pd.DataFrame.from_items(data)
    return dca(data=data, outcome='outcome', predictors=['model'], harms=harms,
               thresh_step=thresh_step, combined=combined,thresh_lo=thresh_lo)
def plot_dca(dca_results=False, y_min=-1, y_max=1):
    '''
    dca_results_1 = decision_curve_analysis(y_true=np.asarray(y_Test.values), y_prob=np.asanyarray(y_prob_1[:,0]), thresh_step=0.01, harms=1, combined=False)
    dca_results_2 = decision_curve_analysis(y_true=np.asarray(y_Test.values), y_prob=np.asanyarray(y_prob_2[:,0]), thresh_step=0.01, harms=1, combined=False)
    dca_results_3 = decision_curve_analysis(y_true=np.asarray(y_Test.values), y_prob=np.asanyarray(y_prob_3[:,0]), thresh_step=0.01, harms=1, combined=False)
    dca_results_4 = decision_curve_analysis(y_true=np.asarray(y_Test.values), y_prob=np.asanyarray(y_prob_4[:,0]), thresh_step=0.01, harms=1, combined=False)
    dca_results_5 = decision_curve_analysis(y_true=np.asarray(y_Test.values), y_prob=np.asanyarray(y_prob_5[:,0]), thresh_step=0.01, harms=1, combined=False)

    dca_results =   {
                'dca_results_1' : {'legend':'Keras DNN', 'results':dca_results_1},
                'dca_results_2' : {'legend':'RandomForest', 'results':dca_results_2},
                'dca_results_3': {'legend': 'Logistic Regression', 'results': dca_results_3},
                'dca_results_4': {'legend': 'Gradient boosting', 'results': dca_results_4},
                'dca_results_5': {'legend': 'SVM', 'results': dca_results_5}
            }
    '''
    sns.set_style("whitegrid")
    sns.set_color_codes()

    for name, dca_result in dca_results.items():
        plt.plot(dca_result['results']['threshold'],
                 dca_result['results']['model'],
                 label=dca_result['legend'], linewidth=2.0)
    plt.plot(dca_results['dca_results_1']['results']['threshold'], dca_results['dca_results_1']['results']['all'], 'k:', label='Treat all', linewidth=1.0)
    plt.plot(dca_results['dca_results_1']['results']['threshold'], dca_results['dca_results_1']['results']['none'], 'k-.', label='Treat none', linewidth=1.0)
    plt.plot(dca_results['dca_results_1']['results']['threshold'], dca_results['dca_results_1']['results']['prevalence'], 'k--', label='Prevalence / Perfect model', linewidth=1.0)
    plt.xlim([0.01, 1])
    plt.ylim([y_min, y_max])
    plt.xlabel('Threshold propability (%)')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve')
    plt.legend(loc="upper right")
    plt.show()

# LIFT / GAIN CHARTS
# Callable functions: plot_lift_gains
def calc_lift(y_true, y_prob, bins=10):
    """
    Takes input arrays and trained SkLearn Classifier and returns a Pandas
    DataFrame with the average lift generated by the model in each bin

    Parameters
    -------------------

    y:    A 1-d Numpy array or Pandas Series with shape = [n_samples]
          IMPORTANT: Code is only configured for binary target variable
          of 1 for success and 0 for failure

    clf:  A trained SkLearn classifier object
    bins: Number of equal sized buckets to divide observations across
          Default value is 10
    """

    # Predicted Probability that y = 1
    # Predicted Value of Y
    cols = ['ACTUAL', 'PROB_POSITIVE']
    data = [y_true, y_prob]
    df = pd.DataFrame(dict(zip(cols, data)))

    # Observations where y=1
    total_positive_n = df['ACTUAL'].sum()
    # Total Observations
    total_n = df.index.size
    natural_positive_prob = total_positive_n / float(total_n)

    # Check to see if predicted zeros exceed single bin
    zero_count = df.loc[df['PROB_POSITIVE'] == 0, 'PROB_POSITIVE'].index.size
    zero_pct = zero_count / float(total_n)

    # Check to see if predicted ones exceed single bin
    one_count = df.loc[df['PROB_POSITIVE'] == 1, 'PROB_POSITIVE'].index.size
    one_pct = one_count / float(total_n)

    # If zeros exceed single bin, add random noise
    if zero_pct > 1 / float(bins):
        print('Too many zeros!')
        # Find min non-negative predicted probability
        prob_min = df.loc[df['PROB_POSITIVE'] != 0, 'PROB_POSITIVE'].min()
        df.loc[df['PROB_POSITIVE'] == 0, 'PROB_POSITIVE'] = np.random.uniform(0, prob_min, zero_count)
    # If ones exceed single bin, add random noise
    if one_pct > 1 / float(bins):
        print("Too many ones!")
        prob_max = df.loc[df['PROB_POSITIVE'] != 1, 'PROB_POSITIVE'].max()
        df.loc[df['PROB_POSITIVE'] == 1, 'PROB_POSITIVE'] = np.random.uniform(prob_max, 1, one_count)

    # Create Bins where First Bin has Observations with the
    # Lowest Predicted Probability that y = 1
    df['BIN_POSITIVE'] = pd.qcut(df['PROB_POSITIVE'], bins, labels=False)

    pos_group_df = df.groupby('BIN_POSITIVE')
    # Percentage of Observations in each Bin where y = 1
    count_positive = pos_group_df['ACTUAL'].sum()
    count_bin = pos_group_df['ACTUAL'].count()
    lift_positive = count_positive / count_bin
    lift_index_positive = (lift_positive / natural_positive_prob) * 100

    # Consolidate Results into Output Dataframe
    lift_df = pd.DataFrame({
        'COUNT_POSITIVE': count_positive,
        'COUNT_BIN': count_bin,
        'LIFT_POSITIVE': lift_positive,
        'LIFT_POSITIVE_INDEX': lift_index_positive,
        'BASELINE_POSITIVE': natural_positive_prob})

    # Calculate Data for Gains Chart
    lift_df = lift_df.append(pd.DataFrame(np.zeros((1, 5)), index=[10], columns=lift_df.columns))
    lift_df.sort_index(ascending=False, inplace=True)
    lift_df['CUM_POSITIVE'] = lift_df['COUNT_POSITIVE'].cumsum() / lift_df['COUNT_POSITIVE'].sum()
    lift_df['CUM_TOTAL'] = lift_df['COUNT_BIN'].cumsum() / lift_df['COUNT_BIN'].sum()
    return lift_df
def plot_lift_gains(y_true, y_prob, bins):
    '''

    Call example:
    gs, fig, ax1, ax2 = plot_lift_gains(y_true=y_Test, y_prob=y_prob_2[:,1], bins=10)

    :param y_true:
    :param y_prob:
    :param bins:
    '''
    data = calc_lift(y_true, y_prob, bins=bins)

    sns.set_style("whitegrid")
    sns.set_color_codes()
    fig = plt.figure(figsize=(12,6))

    #Configure Grid
    gs = gridspec.GridSpec(2,1)

    #Add Subplots
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    lift_plot_data = data[data['CUM_TOTAL']>0]
    ax1.plot(lift_plot_data['CUM_TOTAL'], lift_plot_data['LIFT_POSITIVE_INDEX'],color='k', label='Classifier Performance')
    ax1.axhline(100,linestyle='dashed',color='r',label="Baseline Performance")
    ax1.set_ylim(top=1000)
    ax2.plot(data['CUM_TOTAL'],data['CUM_POSITIVE'],color='k',label='Classifier Performance')
    ax2.plot(data['CUM_TOTAL'],data['CUM_TOTAL'],color='r',linestyle='dashed',label='Baseline Performance')
    ax1.fill_between(lift_plot_data['CUM_TOTAL'],lift_plot_data['LIFT_POSITIVE_INDEX'],100,where=lift_plot_data['LIFT_POSITIVE_INDEX']>=100,interpolate=True,color='b',alpha=0.25)
    ax2.fill_between(data['CUM_TOTAL'],data['CUM_POSITIVE'],data['CUM_TOTAL'],where= data['CUM_POSITIVE']> data['CUM_TOTAL'],interpolate=True,color='b',alpha=0.25)
    ax1.legend(loc='center right',frameon=True)
    ax2.legend(loc='center right',frameon=True)
    ax1.set_xticks(np.linspace(0,1,6))
    ax2.set_xticks(np.linspace(0, 1, 6))
    y1_vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0f}x'.format(val/100) for val in y1_vals])
    x2_vals = ax2.get_xticks()
    ax2.set_xticklabels(['{:3.0f}%'.format(val*100) for val in x2_vals])
    y2_vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:3.0f}%'.format(val*100) for val in y2_vals])
    ax1.xaxis.set_ticklabels([])
    ax1.set_xlabel("Population %")
    ax2.set_xlabel("Population %")
    ax1.set_ylabel("Upgrade Lift Index")
    ax1.yaxis.set_label_coords(-0.05,0.5)
    ax2.set_ylabel("Target %")
    ax1.set_title("Lift Chart")
    ax2.set_title("Gains Chart")
    ax1.set_xlim([0.0, 1])
    ax2.set_xlim([0.0, 1])
    gs.tight_layout(fig)
    return (gs,fig,ax1,ax2)
