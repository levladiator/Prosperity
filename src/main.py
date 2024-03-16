from typing import List, Dict
from datamodel import OrderDepth, TradingState, Order
import copy
import collections

empty_dict = {'AMETHYSTS': 0, 'STARFRUIT': 0}
INF = int(1e9)


def def_value():
    return copy.deepcopy(empty_dict)


class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    volume_traded = copy.deepcopy(empty_dict)

    starfruit_cache = []
    starfruit_dim = 8
    steps = 0

    std = 25

    def calc_next_price_starfruit(self):
        # starfruit cache stores price from 1 day ago, current day resp
        # by price, here we mean mid-price

        coef = [0.04510426, 0.0507533,  0.06015958, 0.09988052, 0.11262721, 0.1870036, 0.16404486, 0.28060765]
        intercept = -0.9779184549179263
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mx_vol = -1

        for ask, vol in order_dict.items():
            if buy == 0:
                vol *= -1
            tot_vol += vol
            if tot_vol > mx_vol:
                mx_vol = vol
                best_val = ask

        return tot_vol, best_val

    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        o_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        o_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(o_sell)
        buy_vol, best_buy_pr = self.values_extract(o_buy, 1)

        c_pos = self.position[product]

        mx_with_buy = -1

        for ask, vol in o_sell.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) 
                                    and (ask == acc_bid))) and c_pos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
                c_pos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 1)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (c_pos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            c_pos += num

        if (c_pos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            c_pos += num

        if c_pos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
            orders.append(Order(product, bid_pr, num))
            c_pos += num

        c_pos = self.position[product]

        for bid, vol in o_buy.items():
            if ((bid > acc_ask) or ((self.position[product] > 0) 
                                    and (bid == acc_ask))) and c_pos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
                # order_for is a negative number denoting how much we will sell
                c_pos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (c_pos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            c_pos += num

        if (c_pos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
            orders.append(Order(product, max(undercut_sell + 1, acc_ask + 1), num))
            c_pos += num

        if c_pos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS'] - c_pos)
            orders.append(Order(product, sell_pr, num))
            c_pos += num

        return orders

    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        o_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        o_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(o_sell)
        buy_vol, best_buy_pr = self.values_extract(o_buy, 1)

        c_pos = self.position[product]

        for ask, vol in o_sell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and c_pos < LIMIT:
                order_for = min(-vol, LIMIT - c_pos)
                c_pos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if c_pos < LIMIT:
            num = LIMIT - c_pos
            orders.append(Order(product, bid_pr, num))
            c_pos += num

        c_pos = self.position[product]

        for bid, vol in o_buy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and c_pos > -LIMIT:
                order_for = max(-vol, -LIMIT - c_pos)
                # order_for is a negative number denoting how much we will sell
                c_pos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if c_pos > -LIMIT:
            num = -LIMIT - c_pos
            orders.append(Order(product, sell_pr, num))
            c_pos += num

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS': [], 'STARFRUIT': []}

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')

        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)

        _, bs_starfruit = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.starfruit_cache.append((bs_starfruit + bb_starfruit) / 2)

        INF = 1e9

        starfruit_lb = -INF
        starfruit_ub = INF

        if len(self.starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit() - 1
            starfruit_ub = self.calc_next_price_starfruit() + 1

        amethysts_lb = 10000
        amethysts_ub = 10000

        # CHANGE FROM HERE

        acc_bid = {'AMETHYSTS': amethysts_lb, 'STARFRUIT': starfruit_lb}  # we want to buy at slightly below
        acc_ask = {'AMETHYSTS': amethysts_ub, 'STARFRUIT': starfruit_ub}  # we want to sell at slightly above

        self.steps += 1

        for product in ['AMETHYSTS', 'STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
            result[product] += orders

        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp - 100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.volume_traded[product] += abs(trade.quantity)

        # print(f'Will trade {result}')
        print("End transmission")

        traderData = "SAMPLE"
        conversions = 1

        return result, conversions, traderData
