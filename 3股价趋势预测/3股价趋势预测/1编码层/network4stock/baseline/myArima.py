#修改自 https://github.com/Jenniferz28/Time-Series-ARIMA-XGBOOST-RNN/blob/master/myArima.py
import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


class Arima_Class:
    def __init__(self, arima_para, seasonal_para, show_fig):
        # 是否画图
        self.show_fig = show_fig
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para['p']
        d = arima_para['d']
        q = arima_para['q']
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        if seasonal_para == [0]:
            self.seasonal_pdq = [None]
        elif type(seasonal_para) == list:
            self.seasonal_pdq = []
            for i in seasonal_para:
                self.seasonal_pdq += [(x[0], x[1], x[2], i)
                                      for x in list(itertools.product(p, d, q))]

        else:
            self.seasonal_pdq = [(x[0], x[1], x[2], seasonal_para)
                                 for x in list(itertools.product(p, d, q))]

    def fit(self, ts):
        warnings.filterwarnings("ignore")
        results_list = []
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(ts,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()

                    print('ARIMA{}x{}seasonal - AIC:{}'.format(param,
                                                               param_seasonal, results.aic))
                    results_list.append([param, param_seasonal, results.aic])
                except Exception as e:
                    print(e)
                    continue
        results_list = np.array(results_list)
        lowest_AIC = np.argmin(results_list[:, 2])
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('ARIMA{}x{}seasonal with lowest_AIC:{}'.format(
            results_list[lowest_AIC, 0], results_list[lowest_AIC, 1], results_list[lowest_AIC, 2]))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        self.best_args = (results_list[lowest_AIC, 0], results_list[lowest_AIC, 1], results_list[lowest_AIC, 2])
        mod = sm.tsa.statespace.SARIMAX(ts,
                                        order=results_list[lowest_AIC, 0],
                                        seasonal_order=results_list[lowest_AIC, 1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.final_result = mod.fit()
        print('Final model summary:')
        print(self.final_result.summary().tables[1])
        if self.show_fig:
            print('Final model diagnostics:')
            self.final_result.plot_diagnostics(figsize=(15, 12))
            plt.tight_layout()
            plt.savefig('model_diagnostics.png', dpi=300)
            plt.show()

    def pred(self, ts, plot_start, pred_start, dynamic, ts_label):

        pred_dynamic = self.final_result.get_prediction(
            start=pd.to_datetime(pred_start), dynamic=dynamic, full_results=True)
        pred_dynamic_ci = pred_dynamic.conf_int()
        if self.show_fig:
            ax = ts[plot_start:].plot(label='observed', figsize=(15, 10))

            if dynamic == False:
                pred_dynamic.predicted_mean.plot(
                    label='One-step ahead Forecast', ax=ax)
            else:
                pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

            ax.fill_between(pred_dynamic_ci.index,
                            pred_dynamic_ci.iloc[:, 0],
                            pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
            ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(plot_start), ts.index[-1],
                             alpha=.1, zorder=-1)
            ax.set_xlabel('Time')
            ax.set_ylabel(ts_label)
            plt.legend()
            plt.tight_layout()
            if dynamic == False:
                plt.savefig(ts_label + '_one_step_pred.png', dpi=300)
            else:
                plt.savefig(ts_label + '_dynamic_pred.png', dpi=300)
            plt.show()
        return pred_dynamic.predicted_mean

    def forcast(self, ts, n_steps, ts_label):
        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)

        # Get confidence intervals of forecasts
        pred_ci = pred_uc.conf_int()
        if self.show_fig:
            ax = ts.plot(label='observed', figsize=(15, 10))
            pred_uc.predicted_mean.plot(ax=ax, label='Forecast in Future')
            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0],
                            pred_ci.iloc[:, 1], color='k', alpha=.25)
            ax.set_xlabel('Time')
            ax.set_ylabel(ts_label)
            plt.tight_layout()
            plt.savefig(ts_label + '_forcast.png', dpi=300)
            plt.legend()
            plt.show()
        return pred_uc.predicted_mean
