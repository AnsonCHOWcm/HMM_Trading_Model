import numpy as np
import math

class performance_review:

    def __init__(self, daily_ret, benchmark_daily_ret=[], signal=[], freq = 'd'):
        self.daily_ret = daily_ret
        self.benchmark_daily_ret = benchmark_daily_ret
        self.signal = signal
        self.freq = freq

    def num_bt_days(self):
        return len(self.daily_ret)

    def daily_win_rate(self):
        win_rate = len([i for i in self.daily_ret if i > 0]) / len(self.signal != 0)
        return win_rate

    def cum_daily_ret(self):
        cum_daily_return = np.cumprod(np.array(self.daily_ret) + 1)
        return cum_daily_return

    def avg_trade_ret(self):
        prev_trade_lv = 1
        curr_trade_lv = 1
        trading = False
        cum_lv = np.cumprod(1 + self.daily_ret)
        trade_PnL = []

        for i in range(len(cum_lv)):
            curr_lv = cum_lv[i]
            if not trading and curr_lv != prev_trade_lv:
                trading = True
            elif trading and curr_lv == curr_trade_lv:
                trading = False
                trade_PnL.append(curr_lv / prev_trade_lv - 1)
                prev_trade_lv = curr_lv
            curr_trade_lv = curr_lv
        avg_trade_ret = np.mean(trade_PnL)
        return avg_trade_ret

    def annualized_alpha(self):

        if self.freq == 'd':
            d = 252
        elif self.freq == 'm':
            d = 12
        elif self.freq == 'w':
            d = 52

        cum_daily_return = self.cum_daily_ret()
        cum_annualized_ret = np.log(cum_daily_return[-1] / cum_daily_return[0]) / len(cum_daily_return) * (d)
        cum_annualized_benchmark_ret = np.log(np.cumprod(np.array(self.benchmark_daily_ret) + 1)[-1] / np.cumprod(np.array(self.benchmark_daily_ret) + 1)[0]) / len(self.benchmark_daily_ret) * (252 if self.freq == 'd' else 12)
        cum_annualized_alpha = cum_annualized_ret - cum_annualized_benchmark_ret
        return cum_annualized_alpha

    def annualized_ret(self):

        if self.freq == 'd':
            d = 252
        elif self.freq == 'm':
            d = 12
        elif self.freq == 'w':
            d = 52

        cum_daily_return = self.cum_daily_ret()
        cum_annualized_ret = np.log(cum_daily_return[-1] / cum_daily_return[0]) / len(cum_daily_return) * (d)
        return cum_annualized_ret

    def annualized_vol(self):

        if self.freq == 'd':
            d = 252
        elif self.freq == 'm':
            d = 12
        elif self.freq == 'w':
            d = 52

        cum_daily_return = self.cum_daily_ret()
        cum_annualized_std = (np.log(cum_daily_return[1:] / cum_daily_return[:-1])).std() * math.sqrt((d))
        return cum_annualized_std

    def sharpe_ratio(self):
        cum_annualized_ret = self.annualized_ret()
        cum_annualized_std = self.annualized_vol()
        sharpe_ratio = cum_annualized_ret / cum_annualized_std if cum_annualized_std != 0 else 0
        return sharpe_ratio

    def max_drawdown(self):
        cum_daily_return = self.cum_daily_ret()
        max_drawdown = np.max((np.maximum.accumulate(cum_daily_return) - cum_daily_return)/np.maximum.accumulate(cum_daily_return))
        return max_drawdown
