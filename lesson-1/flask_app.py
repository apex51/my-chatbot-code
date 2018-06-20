from flask import Flask
from flask import jsonify
from flask import request
import requests


app = Flask(__name__)

coin_dict = {
    "比特币": "btcbtc",
    "以太坊": "ethbtc",
    "莱特币": "ltcbtc",
    "EOS": "eosbtc"
}

fiat_dict = {
    "美元": "usd",
    "人民币": "cny"
}

# Pull crypto prices from web
wci_url = ("https://www.worldcoinindex.com/apiservice/ticker?"
           "key=6pwVQDfWjG0FwsE2TWORWtIGSajcES"
           "&label={}&fiat=usd")


@app.route('/')
def hello_world():
    return jsonify(fulfillmentText='Hello from Flask!')


@app.route('/get_price', methods=['GET', 'POST'])
def get_price():
    json = request.get_json()
    coin_name = json["queryResult"]["parameters"]["coin_name"]
    default_response = json["queryResult"]["fulfillmentText"]

    coin_code = coin_dict[coin_name]
    r = requests.get(wci_url.format(coin_code))
    price = r.json()["Markets"][0]["Price"]
    fiat_unit = "美元"
    success_response = "{}的价格是{:.2f}{}".format(coin_name, price, fiat_unit)

    dic = {'fulfillmentText': success_response}
    return jsonify(dic)
