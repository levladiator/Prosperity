import numpy as np
from typing import List, Dict, Tuple
from datamodel import OrderDepth, TradingState, Order


class Trader:
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}

    def compute_orders_amethyst(self, order_depths, position, acc_bid, acc_ask):
        product = "AMETHYSTS"
        orders = []

        buy_orders = order_depths.buy_orders
        sell_orders = order_depths.sell_orders

        pos_limit = self.POSITION_LIMIT[product]
        curr_pos = position

        # Compute buy orders based on order book
        for ask, vol in sell_orders.items():
            if curr_pos >= pos_limit:
                break

            if ask < acc_bid or (ask == acc_bid and position < 0):
                order_volume = min(-vol, pos_limit - curr_pos)
                order_price = ask
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        # Compute remaining buy orders based on position
        best_bid = next(iter(buy_orders))

        if curr_pos < pos_limit:
            order_volume = min(2 * pos_limit, pos_limit - curr_pos)

            if position < 0:
                order_price = min(best_bid + 2, acc_bid - 1)
            elif position > 15:
                order_price = min(best_bid, acc_bid - 1)
            else:
                order_price = min(best_bid + 1, acc_bid - 1)

            orders.append(Order(product, order_price, order_volume))
            curr_pos += order_volume

        curr_pos = position

        # Compute sell orders based on order book
        for bid, vol in buy_orders.items():
            if curr_pos > -pos_limit:
                break

            if bid > acc_ask or (bid == acc_ask and position > 0):
                order_volume = max(-vol, -pos_limit - curr_pos)
                order_price = bid
                curr_pos += order_volume
                orders.append(Order(product, order_price, order_volume))

        # Compute remaining sell orders based on product position
        best_ask = next(iter(sell_orders))

        if curr_pos > -pos_limit:
            order_volume = max(-2 * pos_limit, -pos_limit - curr_pos)

            if position > 0:
                order_price = max(best_ask - 2, acc_ask + 1)
            elif position < 15:
                order_price = max(best_ask, acc_ask + 1)
            else:
                order_price = max(best_ask - 1, acc_ask + 1)

            orders.append(Order(product, order_price, order_volume))
            curr_pos += order_volume

        return orders

    def run(self, state: TradingState):
        final_orders = {"AMETHYSTS": [], "STARFRUIT": []}

        final_orders["AMETHYSTS"] += (
            self.compute_orders_amethyst(state.order_depths["AMETHYSTS"],
                                         state.position["AMETHYSTS"] if "AMETHYSTS" in state.position else 0,
                                         10000,
                                         10000))

        traderData = "SAMPLE"
        conversions = 1
        return final_orders, conversions, traderData

    def exampleStrategy(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            # Initialize the list of Orders to be sent as an empty list
            orders: List[Order] = []
            # Define a fair value for the PRODUCT. Might be different for each tradable item
            # Note that this value of 10 is just a dummy value, you should likely change it!
            acceptable_price = 10
            # All print statements output will be delivered inside test results
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            # Order depth list come already sorted.
            # We can simply pick first item to check first item to get best bid or offer
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    # In case the lowest ask is lower than our fair value,
                    # This presents an opportunity for us to buy cheaply
                    # The code below therefore sends a BUY order at the price level of the ask,
                    # with the same quantity
                    # We expect this order to trade with the sell order
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    # Similar situation with sell orders
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            result[product] = orders

            # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
