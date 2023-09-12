import pandas as pd

from scipy.stats import ttest_ind, shapiro, ttest_rel, wilcoxon

from decorators import log_info
from decorators import log_info
from draw_results import plot_curves_and_matrix
# draw_system_t_roc_curve, draw_system_roc_curve
from logger import logger


def shapiro_tests(model_1_results, model_2_results, column_name = 'accuracy', alpha = 0.05):
    stat1, p1 = shapiro(model_1_results[column_name])
    stat2, p2 = shapiro(model_2_results[column_name])
    if p1 > alpha or p2 > alpha:
        return False
    return True

@log_info
def paired_statistical_tests(model_1_results, model_2_results, column_name = 'accuracy', alpha = 0.05):
    if shapiro_tests(model_1_results, model_2_results, column_name, alpha):
        logger.info('Both models look Gaussian (fail to reject H0)')
        t_stat, p_value = ttest_rel(model_1_results[column_name], model_2_results[column_name])
        logger.info(f"T-statistic: {t_stat} with p-value: {p_value}")
        return t_stat, p_value, 'T-test'

    else:
        logger.info('At least one of models does not look Gaussian (reject H0)')
        t_stat, p_value = wilcoxon(x=model_1_results[column_name], y=model_2_results[column_name])
        logger.info(f"Wilcoxon statistic: {t_stat} with p-value: {p_value}")
        return t_stat, p_value, 'Wilcoxon'


