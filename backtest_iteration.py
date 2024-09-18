import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
from hmmlearn import hmm
from Performance import performance_review
import warnings
warnings.filterwarnings("ignore")


class HMM_Trading_Model:

    def __init__(self, N, model):

        self.data = None
        self.N = N
        self.model = model

    def train(self, data):

        self.data = data
        self.model.fit(data)

    def hist_hidden_state(self):

        return self.model.predict(self.data)

    def latest_hidden_state(self):

        return self.hist_hidden_state()[-1]

    def latest_hidden_state_post_prob(self):

        return self.model.predict_proba(self.data)[-1, self.latest_hidden_state()]

    def transition_matrix(self):

        return self.model.transmat_

    def cond_observation_mean(self):

        return self.model.means_

    def strat_one_signal_generation_info(self):

        occurence_latest_hidden_state = sum(self.hist_hidden_state() == self.latest_hidden_state())
        total_hist_next_day_observation_given_state = sum(
            self.data[:, 0].reshape(1, -1)[0][1:] * (self.hist_hidden_state() == self.latest_hidden_state())[:-1])
        win_rate = sum(
            (np.sign(self.data[:, 0].reshape(1, -1)[0][1:]) == np.sign(total_hist_next_day_observation_given_state)) * (
                                                                                                                                   self.hist_hidden_state() == self.latest_hidden_state())[
                                                                                                                       :-1]) / sum(
            (self.hist_hidden_state() == self.latest_hidden_state())[:-1])

        return occurence_latest_hidden_state, total_hist_next_day_observation_given_state, win_rate

    def strat_two_signal_generation_info(self):

        cond_next_daily_ret_mean = \
        (self.transition_matrix()[self.latest_hidden_state()][:] @ self.cond_observation_mean())[0]

        return cond_next_daily_ret_mean

    def empirical_kelly_betting_size(self):

        hist_hidden_state = self.hist_hidden_state()
        latest_hidden_state = self.latest_hidden_state()
        next_day_observation_given_state = self.data[
            np.where(hist_hidden_state == latest_hidden_state)[0][:-1] + np.array(1), 0]
        mean = np.mean(next_day_observation_given_state)
        var = np.var(next_day_observation_given_state)

        return abs(mean / var)

    def model_kelly_betting_size(self):
        latest_hidden_state = self.latest_hidden_state()
        mu_1 = self.model.means_[latest_hidden_state][0]
        sigma_11 = self.model.covars_[latest_hidden_state][0, 0]
        mu_2 = self.model.means_[latest_hidden_state][1:]
        sigma_12 = self.model.covars_[latest_hidden_state][0, 1:]
        sigma_21 = self.model.covars_[latest_hidden_state][1:, 0]
        sigma_22 = self.model.covars_[latest_hidden_state][1:, 1:]

        mean = mu_1 + sigma_12 @ np.linalg.inv(sigma_22) @ (self.data[-1][1:] - mu_2)
        covar = sigma_11 - sigma_12 @ np.linalg.inv(sigma_22) @ sigma_21

        return abs(mean / covar)

    def strat_one_signal(self, long_win_rate_thres=0, short_win_rate_thres=0):

        W = len(self.data)
        N = self.N

        occurence_latest_hidden_state, total_hist_next_day_observation_given_state, win_rate = self.strat_one_signal_generation_info()

        # step 1

        if occurence_latest_hidden_state <= W / (3 * N):

            return 0

        else:

            # step 2

            if np.sign(total_hist_next_day_observation_given_state) > 0 and win_rate > long_win_rate_thres:

                return 1

            elif np.sign(total_hist_next_day_observation_given_state) < 0 and win_rate > short_win_rate_thres:

                return -1

            else:
                return 0

    def strat_two_signal(self, long_thres=0, short_thres=0):

        # Tower Property
        cond_next_daily_ret_mean = self.strat_two_signal_generation_info()

        if cond_next_daily_ret_mean > long_thres:
            return 1
        elif cond_next_daily_ret_mean < short_thres:
            return -1
        else:
            return 0

# Signal Generation

def model_recursive_training(data_df, N, model, W):
    data_len = len(data_df)

    occurence_latest_hidden_state_ls = []
    total_hist_next_day_observation_given_state_ls = []
    win_rate_ls = []
    cond_next_daily_ret_mean_ls = []
    em_kelly_bs_ls = []
    model_kelly_bs_ls = []

    for i in range(W, data_len):
        print(i)

        # obtain data input

        curr_data_input = data_df.iloc[(i - W):i]
        o_t = np.array(curr_data_input).reshape(-1, len(data_df.columns))

        # model training

        model.train(o_t)

        # signal info generation

        occurence_latest_hidden_state, total_hist_next_day_observation_given_state, win_rate = model.strat_one_signal_generation_info()
        cond_next_daily_ret_mean = model.strat_two_signal_generation_info()
        empirical_kelly_betting_size = model.empirical_kelly_betting_size()
        model_kelly_betting_size = model.model_kelly_betting_size()


        occurence_latest_hidden_state_ls.append(occurence_latest_hidden_state)
        total_hist_next_day_observation_given_state_ls.append(total_hist_next_day_observation_given_state)
        win_rate_ls.append(win_rate)
        cond_next_daily_ret_mean_ls.append(cond_next_daily_ret_mean)
        em_kelly_bs_ls.append(empirical_kelly_betting_size)
        model_kelly_bs_ls.append(model_kelly_betting_size)

    signal_info_df = pd.DataFrame(index=data_df.index[W:data_len])
    signal_info_df['occurence_latest_hidden_state'] = occurence_latest_hidden_state_ls
    signal_info_df['total_hist_next_day_observation_given_state'] = total_hist_next_day_observation_given_state_ls
    signal_info_df['win_rate'] = win_rate_ls
    signal_info_df['cond_next_daily_ret_mean'] = cond_next_daily_ret_mean_ls
    signal_info_df['empirical_kelly_betting_size'] = em_kelly_bs_ls
    signal_info_df['model_kelly_betting_size'] = model_kelly_bs_ls

    return signal_info_df

def signal_generation(signal_info_df, signal_type, long_thres, short_thres, W=None, N=None):
    signal_ls = []

    for i in range(len(signal_info_df)):

        if signal_type == 1:

            occurence_latest_hidden_state = signal_info_df.occurence_latest_hidden_state.iloc[i]
            total_hist_next_day_observation_given_state = \
            signal_info_df.total_hist_next_day_observation_given_state.iloc[i]
            win_rate = signal_info_df.win_rate.iloc[i]

            # step 1

            if occurence_latest_hidden_state <= W / (3 * N):

                signal = 0

            else:

                # step 2

                if np.sign(total_hist_next_day_observation_given_state) > 0 and win_rate > long_thres:

                    signal = 1

                elif np.sign(total_hist_next_day_observation_given_state) < 0 and win_rate > short_thres:

                    signal = -1

                else:
                    signal = 0


        elif signal_type == 2:

            cond_next_daily_ret_mean = signal_info_df.cond_next_daily_ret_mean.iloc[i]

            if cond_next_daily_ret_mean > long_thres:

                signal = 1

            elif cond_next_daily_ret_mean < short_thres:

                signal = -1

            else:

                signal = 0

        else:
            print("Invalid signal type")
            return None

        signal_ls.append(signal)

    signal_df = pd.DataFrame(index=signal_info_df.index)
    signal_df['signal'] = signal_ls

    return signal_df

class backtest:

    def __init__(self, signal, instrument, freq='d', signal_info_df=None):
        self.signal = signal
        self.signal_info_df = signal_info_df
        self.instrument = instrument
        self.train_daily_ret_df = None
        self.test_daily_ret_df = None
        self.freq = freq

    def train(self, train_start_date, train_end_date, tc, signal_shift=1, betting_scheme=None):

        train_signal_df = self.signal[self.signal.index >= train_start_date]
        train_signal_df = train_signal_df[train_signal_df.index <= train_end_date]

        train_instrument_df = self.instrument[self.instrument.index >= train_start_date]
        train_instrument_df = train_instrument_df[train_instrument_df.index <= train_end_date]

        train_concat_data_df = pd.merge(train_instrument_df, train_signal_df, how='inner', left_index=True,
                                        right_index=True)

        if betting_scheme == "win_rate":
            train_concat_data_df = pd.merge(train_concat_data_df, self.signal_info_df.win_rate, how='inner',
                                            left_index=True, right_index=True)
            train_concat_data_df.columns = ["Open", "signal", "Size"]
        elif betting_scheme == "empirical_kelly":
            train_concat_data_df = pd.merge(train_concat_data_df, self.signal_info_df.empirical_kelly_betting_size,
                                            how='inner', left_index=True, right_index=True)
            train_concat_data_df.columns = ["Open", "signal", "Size"]
        elif betting_scheme == "model_kelly":
            train_concat_data_df = pd.merge(train_concat_data_df, self.signal_info_df.model_kelly_betting_size,
                                            how='inner', left_index=True, right_index=True)
            train_concat_data_df.columns = ["Open", "signal", "Size"]

        if betting_scheme is not None:
            train_daily_ret_df = (train_concat_data_df.Open.pct_change()) * train_concat_data_df.signal.shift(
                signal_shift) * [min(1, size) for size in train_concat_data_df.Size.shift(signal_shift)]
        else:
            train_daily_ret_df = (train_concat_data_df.Open.pct_change()) * train_concat_data_df.signal.shift(
                signal_shift)

        train_daily_ret_df.dropna(inplace=True)
        train_daily_ret_df = train_daily_ret_df - np.abs(
            [train_concat_data_df.signal[0]] + list(train_concat_data_df.signal.diff().dropna()[:-1])) * tc
        benchmark_daily_ret_df = train_concat_data_df.Open.pct_change()
        benchmark_daily_ret_df.dropna(inplace=True)
        self.train_daily_ret_df = train_daily_ret_df

        performance = performance_review(train_daily_ret_df, benchmark_daily_ret_df,
                                         train_concat_data_df.signal.shift(1), self.freq)

        return train_daily_ret_df, benchmark_daily_ret_df, performance.annualized_ret(), performance.annualized_alpha(), performance.sharpe_ratio(), performance.max_drawdown(), performance.daily_win_rate()

    def test(self, test_start_date, tc, signal_shift=1, betting_scheme=None):

        test_signal_df = self.signal[self.signal.index >= test_start_date]
        test_instrument_df = self.instrument[self.instrument.index >= test_start_date]

        test_concat_data_df = pd.merge(test_instrument_df, test_signal_df, how='inner', left_index=True,
                                       right_index=True)

        if betting_scheme == "win_rate":
            test_concat_data_df = pd.merge(test_concat_data_df, self.signal_info_df.win_rate, how='inner',
                                           left_index=True, right_index=True)
            test_concat_data_df.columns = ["Open", "signal", "Size"]
        elif betting_scheme == "empirical_kelly":
            test_concat_data_df = pd.merge(test_concat_data_df, self.signal_info_df.empirical_kelly_betting_size,
                                           how='inner', left_index=True, right_index=True)
            test_concat_data_df.columns = ["Open", "signal", "Size"]
        elif betting_scheme == "model_kelly":
            test_concat_data_df = pd.merge(test_concat_data_df, self.signal_info_df.model_kelly_betting_size,
                                           how='inner', left_index=True, right_index=True)
            test_concat_data_df.columns = ["Open", "signal", "Size"]

        if betting_scheme is not None:
            test_daily_ret_df = (test_concat_data_df.Open.pct_change()) * test_concat_data_df.signal.shift(
                signal_shift) * [min(1, size) for size in test_concat_data_df.Size.shift(signal_shift)]
        else:
            test_daily_ret_df = (test_concat_data_df.Open.pct_change()) * test_concat_data_df.signal.shift(signal_shift)

        test_daily_ret_df.dropna(inplace=True)
        test_daily_ret_df = test_daily_ret_df - np.abs(
            [test_concat_data_df.signal[0]] + list(test_concat_data_df.signal.diff().dropna()[:-1])) * tc
        benchmark_daily_ret_df = test_concat_data_df.Open.pct_change()
        benchmark_daily_ret_df.dropna(inplace=True)
        self.test_daily_ret_df = test_daily_ret_df

        performance = performance_review(test_daily_ret_df, benchmark_daily_ret_df, test_concat_data_df.signal.shift(1),
                                         self.freq)

        return test_daily_ret_df, benchmark_daily_ret_df, performance.annualized_ret(), performance.annualized_alpha(), performance.sharpe_ratio(), performance.max_drawdown(), performance.daily_win_rate()

def backtest_long_short_thres(signal_info_df, instrument_data_df, N,W, signal_type, tc, train_start_date, train_end_date, freq = 'd', signal_shift = 1):
    long_thres_ls = [i/10 for i in range(10)]
    short_thres_ls = [i/10 for i in range(10)]

    annualized_ret_df = pd.DataFrame()
    annualized_alpha_df = pd.DataFrame()
    annualized_sharpe_df = pd.DataFrame()
    annualized_mdd_df = pd.DataFrame()
    win_rate_df = pd.DataFrame()

    for long_thres in long_thres_ls:

        annualized_ret_ls = []
        annualized_alpha_ls = []
        annualized_sharpe_ls = []
        annualized_mdd_ls = []
        win_rate_ls = []

        for short_thres in short_thres_ls:

            signal_df = signal_generation(signal_info_df, signal_type, long_thres, short_thres, W , N )
            backtest_obj = backtest(signal_df, instrument_data_df, freq)

            _, _, annualized_ret, annualized_alpha, sharpe_ratio, max_drawdown,  daily_win_rate = backtest_obj.train(train_start_date, train_end_date, tc, signal_shift)
            annualized_ret_ls.append(annualized_ret)
            annualized_alpha_ls.append(annualized_alpha)
            annualized_sharpe_ls.append(sharpe_ratio)
            annualized_mdd_ls.append(max_drawdown)
            win_rate_ls.append(daily_win_rate)

        annualized_ret_df[long_thres] = annualized_ret_ls
        annualized_alpha_df[long_thres] = annualized_alpha_ls
        annualized_sharpe_df[long_thres] = annualized_sharpe_ls
        annualized_mdd_df[long_thres] = annualized_mdd_ls
        win_rate_df[long_thres] = win_rate_ls

    annualized_ret_df.index = short_thres_ls
    annualized_alpha_df.index = short_thres_ls
    annualized_sharpe_df.index = short_thres_ls
    annualized_mdd_df.index = short_thres_ls
    win_rate_df.index = short_thres_ls

    print('annualized ret')
    print(annualized_ret_df)
    print('annualized alpha')
    print(annualized_alpha_df)
    print('annualized sharpe')
    print(annualized_sharpe_df)
    print('annualized mdd')
    print(annualized_mdd_df)
    print('win rate')
    print(win_rate_df)

if __name__ == "__main__":

    ticker_ls = ["SPX", "NDX", "DJI"]
    w_ls = [[120], [252],[252]]
    N_ls = [3,2,2]
    freq_ls = ['w', 'd', 'd']

    iteration_status_df = pd.DataFrame(columns = ['Status'])

    for i, ticker in enumerate(ticker_ls):

        freq = freq_ls[i]


        if freq == 'd':
            data_df = pd.read_excel("US_HMM_data_input_1d.xlsx", index_col=0)
        elif freq == 'w':
            data_df = pd.read_excel("US_HMM_data_input_1w.xlsx", index_col=0)

        input_df = pd.DataFrame(index=data_df.index)
        input_df['stock_return'] = np.log(data_df[ticker + '_Close']).diff()
        input_df['DXY_change'] = data_df['DXY_Close'].pct_change()
        input_df['UST_10y_yield'] = data_df['UST_10y_yield_Close']
        input_df['yield_slope_change'] = ((data_df['UST_10y_yield_Close'] - data_df['UST_13w_yield_Close']).diff())
        # input_df['PMI_change'] = data_df['PMI'].diff()
        input_df.dropna(inplace=True)

        if ticker != "SPX":
            instrument_data_df = pd.DataFrame(data_df[ticker + "_Next_Open"])
        else:
            instrument_data_df = pd.DataFrame(data_df[ticker + "_Fut_Next_Open"])

        instrument_data_df.columns = ["Open"]
        index_ls = [index.strftime("%Y-%m-%d") for index in instrument_data_df.index]
        instrument_data_df.index = index_ls

        Ws = w_ls[i]
        N = N_ls[i]

        for w in Ws:

            three_state_ghmm_model = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=1000, algorithm='map',
                                                     random_state=99)
            three_state_ghmm_trading_model = HMM_Trading_Model(N, three_state_ghmm_model)

            # model recursive_training
            signal_info_df = model_recursive_training(input_df, N, three_state_ghmm_trading_model, w)

            try:

                signal_info_df.to_csv("GHMM_hs_3_" + ticker +"_DXY_UST_w_" + str(w) + "_" + str(freq) + "_freq_v2_map.csv")
                status = "T"

            except:

                status = "F"

            iteration_status_df.loc["GHMM_hs_" + str(N) + "_" + ticker +"_DXY_UST_w_" + str(w) + "_" + str(freq) + "_freq_v2"] = status

    iteration_status_df.to_csv("GHMM_iteration_status.csv")


