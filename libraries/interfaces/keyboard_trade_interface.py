from datetime import datetime
import keyboard
import yaml
import time
import sys
sys.path.append('/Users/user/Desktop/diff')
from libraries.oanda import fetch_account_details
from libraries.oanda import create_order
from libraries.oanda import close_position


# NOTES
"""

There are plenty more details, but we are now finally off an running.
Seriouslty I've spent probbly 2 days on this thing.  wow.


    WILL need fault handling for insufficient margin
    need good printout to show relevant tradae details    
    need to make sure it's synchonized - especially at startup - with oanda reality

    
In General:
    Keypress is ugly - ignore for now
    could probably collect more information for each transaction
    since transaction might be useful if i nees more information

    
Notes on Quantity Calculation 

    Need to make sure that I buy a lower enough number of units so that I won't have a margin closeout
    before the stop loss hist ( though an extra backup margin closeout shortly after that point is good )
    
    Need to examine how many pips I should reasonably need to win.
    I am not able to buy unlimited quantitis of stock,
    so I need to win a minimal amount on each one.
    Practice will show me this.

"""


class TradeInterface:

    def __init__(self, pair):

        # Some Key loop Variables - not stored in configurations
        self.order_timestamp = datetime.now()
        self.active_order = False
        self.active_direction = ''
        self.order = {}
        self.kill = {}
        self.time_delay = 200000
        self.pair = pair

        # Import Configurations and 'global_variables'
        print('\n\nKeyboard Trader Activated\n\n')
        self.configs_file = '/Users/user/Desktop/diff/configs.yaml'
        with open(self.configs_file) as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)

        # Fetch Account Data from Oanda On Startup
        account_details = fetch_account_details(self.configs['oanda_api'], self.configs['oanda_account'])
        self.configs['account_margin'] = float(account_details['account']['marginRate'])
        self.configs['account_balance'] = float(account_details['account']['balance'])
        self.configs['open_positions'] = int(account_details['account']['openTradeCount'])
        self.configs['last_transaction_id'] = int(account_details['account']['lastTransactionID'])
        self.configs['margin_used'] = float(account_details['account']['marginUsed'])
        self.configs['margin_available'] = float(account_details['account']['marginAvailable'])
        self.write_configs()

        # Close All Current Trades:

        # Start keyboard Loop
        keyboard.hook(self.my_keyboard_hook)
        keyboard.wait()


    def write_configs(self):
        with open(self.configs_file, 'w') as f:
            yaml.dump(self.configs, f)


    def print_msg(self, msg):
        msg = msg + ': {}'
        msg = msg.format(datetime.now())
        print(msg)


    def kill_order(self):
        proceed = (datetime.now() - self.order_timestamp).microseconds > self.time_delay
        if proceed:
            if self.active_order:
                order_timestamp = datetime.now()
                self.print_msg('\nKILL Keypress')
                # self.kill = close_position(self.active_direction, self.configs['pair'])
                self.kill = close_position(self.active_direction, self.pair)
                self.active_order = False
                # order_create = 'longOrderCreateTransaction'
                # order_fill = 'longOrderFillTransaction'
                # if self.active_direction == 'sell':
                #     order_create = 'shortOrderCreateTransaction'
                #     order_fill = 'shortOrderFillTransaction'
                # print('Kill Create Timestamp: {}'.format(self.kill[order_create]['time']))
                # print('Kill Fill Timestamp:   {}'.format(self.kill[order_fill]['time']))
                # print('Kill Price: {}'.format(self.kill[order_fill]['price']))
                # print('Order Commission     : {}'.format(self.kill[order_fill]['commission']))
                # self.configs['account_balance'] = float(self.kill[order_fill]['accountBalance'])
                # self.configs['last_transaction_id'] = self.kill['lastTransactionID']
                self.write_configs()
                print(self.configs)
            else:
                print('No Active Orders to close')
            self.order_timestamp = datetime.now()


    def purchase_order(self, direction):
        proceed = (datetime.now() - self.order_timestamp).microseconds > self.time_delay
        if proceed:
            if not self.active_order:
                # if direction == 'buy':
                #     price = self.configs['bid']
                # else:
                #     price = self.configs['ask']

                # print(self.configs['trade_risk'], self.configs['account_balance'])
                # print(type(self.configs['trade_risk']), type(self.configs['account_balance']))
                self.order = create_order(self.pair,
                                          direction )
                # self.order = create_order(self.configs['pair'],
                #                           direction,
                #                           self.configs['pips'],
                #                           self.configs['commission'],
                #                           self.configs['trade_risk'] * self.configs['account_balance'],
                #                           price,
                #                           self.configs['oanda_api'],
                #                           self.configs['oanda_account'])
                # print()
                self.order_timestamp = datetime.now()
                self.active_direction = direction
                self.print_msg('\nBuy Keypress')
                self.active_order = True
                print('Order Created: {}'.format(self.order['orderCreateTransaction']['time']))
                self.write_configs()
                print(self.configs)
                print(self.order)
            else:
                print('Active Order: can not place new')
            self.order_timestamp = datetime.now()


    def adjust_pips(self, direction):
        proceed = (datetime.now() - self.order_timestamp).microseconds > self.time_delay
        if proceed:
            if direction == 'up':
                self.configs['pips'] += self.configs['pips_step']
            else:
                if self.configs['pips'] > self.configs['pips_step']:
                    self.configs['pips'] -= self.configs['pips_step']
                else:
                    print('Can not lower pips value further')
            self.print_msg('Pips loss Raised to {}'.format(self.configs['pips']))
            self.write_configs()
            self.order_timestamp = datetime.now()


    def my_keyboard_hook(self, keyboard_event):
        """ Callback Loop"""
        # Key Read
        key = keyboard_event.scan_code
        # Kill
        if key == 49:
            self.kill_order()
        # Buy
        if key == 36:
            self.purchase_order('buy')
        # Sell
        if key == 57:
            self.purchase_order('sell')
        # Adjust Pips Up
        if key == 126:
            self.adjust_pips('up')
        # Adjust pips down
        if key == 125:
            self.adjust_pips('down')
        # Otherwise, print just to make sure it's running:
        if key not in [49, 36, 57, 125, 126]:
            print(' Keyboard trader active')


if __name__ == '__main__':
    t = TradeInterface('EUR_DKK')

"""
account_balance: 821.6841
account_margin: 0.02
active_direction: ''
active_order: false
ask: 1.11569
bid: 1.11562
commission: 5.0e-05
last_transaction: 0
last_transaction_id: '17399'
margin_available: 837.5841
margin_used: 0.0
oanda_account: 101-001-7518331-001
oanda_api: f3e81960f4aa75e7e3da1f670df51fae-73d8ac6fb611eb976731c859e2517a9b
open_positions: 0
pair: EUR_USD
pair_margin: 0.02
pips: 5.0
pips_step: 0.25
realized: 0
trade_risk: 0.01
"""


