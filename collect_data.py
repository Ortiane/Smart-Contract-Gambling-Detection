import numpy as np
import pandas as pd
import requests
from tabulate import tabulate
import csv
from config import *

END_BLOCK = "999999999"
DICE_2_WIN = "0xD1CEeeeee83F8bCF3BEDad437202b6154E9F5405"


def hex_string_to_int_string(x):
    if x == "0x":
        return str(0)
    return str(int(x[:16], 16))


# All requests to Etherscan API gets by default 10000 entries
# To get fewer entires, use &page=<page number> and &offset=<max records to return>
def get_contract_info(
    contract_address,
    count=1000,
    column_names=["value", "internal_value", "token_value"],
):
    # Get transactions
    txn_req = f"http://api.etherscan.io/api?module=account&action=txlist&address={contract_address}&startblock=0&endblock={END_BLOCK}&sort=desc&apikey={ETHERSCAN_API_KEY}"
    df = pd.DataFrame.from_dict(requests.get(txn_req).json()["result"]).iloc[:count, :]
    # Get internal transactions
    internal_req = f"http://api.etherscan.io/api?module=account&action=txlistinternal&address={contract_address}&startblock=0&endblock={END_BLOCK}&sort=desc&apikey={ETHERSCAN_API_KEY}"
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
    logs_req = f"https://api.etherscan.io/api?module=logs&action=getLogs&fromBlock={df.blockNumber.min()}&toBlock=latest&address={contract_address}&sort=desc&apikey={ETHERSCAN_API_KEY}"
    logs_df = pd.DataFrame.from_dict(requests.get(logs_req).json()["result"])
    if not logs_df.empty:
        logs_df = logs_df.apply(
            {"data": hex_string_to_int_string, "transactionHash": lambda x: x,}
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

    df = df[column_names]
    return df


def print_contract_info(contract_address, count=1000):
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        -1,
    ):
        print(
            tabulate(
                get_contract_info(
                    contract_address,
                    count=count,
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

def process_single_contract(contract_address, contract_label, count):
    info = get_contract_info(
        contract_address,
        count=count,
        column_names=["value", "internal_value", "token_value",],
    )
    desc = [contract_label, contract_address] + list(info.to_numpy().flatten())
    asc = [contract_label, contract_address] + list(info.to_numpy().flatten())[::-1]
    return desc, asc


# row 1: [[v_0,i_0,t_0],[v_1,i_1,t_1],... [v_t,i_t,t_t]]
def collect(max_seq_len):
    with open("contracts.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        contract_list = []
        ct = 0
        for line in reader:
            if ct == 0:
                ct = 1
                continue
            contract_list.append(line[:2])
    with open("data.csv", "w+") as contract_info_fh:
        index = 0
        for contract in contract_list:
            contract_id = contract[0]
            contract_label = contract[1]
            contract_id = contract_id.strip()
            contract_label = int(contract_label)
            # data augmentation
            # descending order
            desc, asc = process_single_contract(
                contract_id, contract_label, max_seq_len * 10
            )
            desc = [str(item) for item in desc]
            for i in range(len(desc) // 300):
                contract_info_fh.write(
                    ",".join(
                        desc[:2]
                        + desc[max_seq_len * i * 3 + 2 : max_seq_len * (i + 1) * 3 + 2]
                    )
                    + "\n"
                )
            # ascending order
            # contract_info_fh.write(",".join(asc) + '\n')
            contract_info_fh.flush()
            print("Finished contract " + str(index) + " " + contract_id)
            index += 1


# ["label": 0,1, "values": list of integers (positive = sending money, negative = receiving)]
if __name__ == "__main__":
    collect(100)
