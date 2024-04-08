import random

from datamodel import Listing, OrderDepth, Trade, TradingState
from main import Trader
import unittest
import numpy as np
from tqdm import tqdm


class MyTestCase(unittest.TestCase):
    timestamp = 1000

    listings = {
        "AMETHYSTS": Listing(
            symbol="AMETHYSTS",
            product="AMETHYSTS",
            denomination="SEASHELLS"
        ),
        "PRODUCT2": Listing(
            symbol="STARFRUIT",
            product="STARFRUIT",
            denomination="SEASHELLS"
        ),
    }

    order_depths = {
        "AMETHYSTS": OrderDepth(
            buy_orders={10: 7, 9: 5},
            sell_orders={11: -4, 12: -8}
        ),
        "STARFRUIT": OrderDepth(
            buy_orders={142: 3, 141: 5},
            sell_orders={144: -5, 145: -8}
        ),
    }

    own_trades = {
        "AMETHYSTS": [],
        "STARFRUIT": []
    }

    market_trades = {
        "AMETHYSTS": [
            Trade(
                symbol="AMETHYSTS",
                price=11,
                quantity=4,
                buyer="",
                seller="",
                timestamp=900
            )
        ],
        "STARFRUIT": []
    }

    position = {
        "AMETHYSTS": 3,
        "STARFRUIT": -5
    }

    observations = {}
    traderData = ""

    state = TradingState(
        traderData,
        timestamp,
        listings,
        order_depths,
        own_trades,
        market_trades,
        position,
        observations
    )

    trader = Trader()

    def test_main(self):
        self.trader.run(self.state)
        # assert True == True

    def test_manual_trading(self):
        probs = np.linspace(0, 1, 101)
        probs = probs / np.sum(probs)
        possible_prices = np.arange(900, 1001)
        avg = []
        for _ in tqdm(range(20)):
            stats = []
            for lowest_bid in range(900, 1001):
                for highest_bid in range (lowest_bid, 1001):
                    reserve_prices = random.choices(possible_prices, probs, k=10000)
                    profit = 0
                    for price in reserve_prices:
                        if lowest_bid >= price:
                            profit += 1000 - lowest_bid
                        elif highest_bid >= price:
                            profit += 1000 - highest_bid

                    stats.append((profit, lowest_bid, highest_bid))

            avg.append(sorted(stats, key=lambda x: x[0])[-1])

        print(avg)
        avg = np.mean(avg, axis=0)
        print("Average best profit: ", avg[0])
        print("Average best lower bound: ", avg[1])
        print("Average best upper bound: ", avg[2])

if __name__ == '__main__':
    unittest.main()



