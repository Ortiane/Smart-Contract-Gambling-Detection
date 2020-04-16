import pandas as pd
import requests
from tabulate import tabulate

from config import *

END_BLOCK = "999999999"
DICE_2_WIN = "0xD1CEeeeee83F8bCF3BEDad437202b6154E9F5405"

def hex_string_to_int_string(x):
    if x == "0x":
        return str(0)
    return str(int(x, 16))


def get_contract_info(
    contract_address,
    count=2000,
    column_names=["value", "internal_value", "token_value"],
):
    # Get transactions
    txn_req = f"http://api.etherscan.io/api?module=account&action=txlist&address={contract_address}&startblock=0&endblock={END_BLOCK}&page=1&offset=100&sort=desc&apikey={ETHERSCAN_API_KEY}"
    df = pd.DataFrame.from_dict(requests.get(txn_req).json()["result"])
    # Get internal transactions
    internal_req = f"http://api.etherscan.io/api?module=account&action=txlistinternal&address={contract_address}&startblock=0&endblock={END_BLOCK}&page=1&offset=100&sort=desc&apikey={ETHERSCAN_API_KEY}"
    internal_df = pd.DataFrame.from_dict(requests.get(internal_req).json()["result"])    
    internal_df.rename(
        columns={
            "from": "internal_from",
            "to": "internal_to",
            "value": "internal_value",
        },
        inplace=True,
    )
    # Get logs for token transfers
    logs_req = f"https://api.etherscan.io/api?module=logs&action=getLogs&fromBlock=0&toBlock=latest&address={contract_address}&page=1&offset=100&sort=desc&apikey={ETHERSCAN_API_KEY}"
    logs_df = pd.DataFrame.from_dict(requests.get(logs_req).json()["result"])
    if not logs_df.empty:
        logs_df = logs_df.apply(
            {
                "data": hex_string_to_int_string,
                "transactionHash": hex_string_to_int_string,
            }
        )
        logs_df.rename(
            columns={"data": "token_value", "transactionHash": "hash"}, inplace=True
        )

    if not internal_df.empty:
        df = df.merge(
            internal_df[["internal_from", "internal_to", "internal_value", "hash"]],
            how="left",
            on="hash",
        )
    else:
        df["internal_from"] = 0
        df["internal_to"] = 0
        df["internal_value"] = 0

    if not logs_df.empty:
        df = df.merge(logs_df, how="left", on="hash")
    else:
        df["token_value"] = 0
    df.sort_values(["blockNumber"], ascending=False, inplace=True)
    df.reset_index(inplace=True)
    # fill nans
    df.fillna(0, inplace=True)

    df = df[column_names].iloc[:count, :]
    return df


# ["label": 0,1, "values": list of integers (positive = sending money, negative = receiving)]
if __name__ == "__main__":
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        -1,
    ):
    # row 1: [[v_0,i_0,t_0],[v_1,i_1,t_1],... [v_t,i_t,t_t]]
        print(
            tabulate(
                get_contract_info(
                    "0x8995AD7dEaBd17c31b62AC89EE5f7D850a4BeDb0",
                    count=100,
                    column_names=[
                        "blockNumber",
                        "from",
                        "to",
                        "internal_from",
                        "internal_to",
                        "value",
                        "internal_value",
                        "token_value",
                    ],
                ),
                headers=[
                    "time",
                    "blockNumber",
                    "from",
                    "to",
                    "internal_from",
                    "internal_to",
                    "value",
                    "internal_value",
                    "token_value",
                ],
            )
        )
