from datamodel import Listing, OrderDepth, Trade, TradingState
from main import Trader
import unittest


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

    def test_something(self):
        self.trader.run(self.state)
        # assert True == True


if __name__ == '__main__':
    unittest.main()



