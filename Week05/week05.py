import numpy as np
import pandas as pd
from pyECLAT import ECLAT
from itertools import combinations

class DataGenerator:
    def __init__(self):
        self.data = [
            ['Wine', 'Chips', 'Bread', 'Butter', 'Milk', 'Apple'],
            ['Wine', np.nan, 'Bread', 'Butter', 'Milk', np.nan],
            [np.nan, np.nan, 'Bread', 'Butter', 'Milk', np.nan],
            [np.nan, 'Chips', np.nan, np.nan, np.nan, 'Apple'],
            ['Wine', 'Chips', 'Bread', 'Butter', 'Milk', 'Apple'],
            ['Wine', 'Chips', np.nan, np.nan, 'Milk', np.nan],
            ['Wine', 'Chips', 'Bread', 'Butter', np.nan, 'Apple'],
            ['Wine', 'Chips', np.nan, np.nan, 'Milk', np.nan],
            ['Wine', np.nan, 'Bread', np.nan, np.nan, 'Apple'],
            ['Wine', np.nan, 'Bread', 'Butter', 'Milk', np.nan],
            [np.nan, 'Chips', 'Bread', 'Butter', np.nan, 'Apple'],
            ['Wine', np.nan, np.nan, 'Butter', 'Milk', 'Apple'],
            ['Wine', 'Chips', 'Bread', 'Butter', 'Milk', np.nan],
            ['Wine', np.nan, 'Bread', np.nan, 'Milk', 'Apple'],
            ['Wine', np.nan, 'Bread', 'Butter', 'Milk', 'Apple'],
            ['Wine', 'Chips', 'Bread', 'Butter', 'Milk', 'Apple'],
            [np.nan, 'Chips', 'Bread', 'Butter', 'Milk', 'Apple'],
            [np.nan, 'Chips', np.nan, 'Butter', 'Milk', 'Apple'],
            ['Wine', 'Chips', 'Bread', 'Butter', 'Milk', 'Apple'],
            ['Wine', np.nan, 'Bread', 'Butter', 'Milk', 'Apple'],
            ['Wine', 'Chips', 'Bread', np.nan, 'Milk', 'Apple'],
            [np.nan, 'Chips', np.nan, np.nan, np.nan, np.nan]
        ]
        

    def load_data(self):
        data_frame = pd.DataFrame(self.data)
        return data_frame

class VerticalAprioriModel:
    def __init__(self, data, min_support_ratio):
        self.records = data
        self.n_transactions = len(data)
        self.min_support_count = min_support_ratio * self.n_transactions
        self.vertical_data = self._build_vertical_data()

    def _build_vertical_data(self):
        v_data = {}
        for idx, transaction in enumerate(self.records):
            for item in transaction:
                if item is not None and str(item) != 'nan':
                    if item not in v_data:
                        v_data[item] = set()
                    v_data[item].add(idx)
        return v_data

    def run_eclat(self):
        frequent_itemsets = {}
        
        items = sorted([item for item, tids in self.vertical_data.items() 
                       if len(tids) >= self.min_support_count])
        
        self._recurse(items, set(), self.vertical_data, frequent_itemsets)
        return frequent_itemsets

    def _recurse(self, items_to_test, current_prefix, v_data, result):
        for i, item in enumerate(items_to_test):
            new_itemset = tuple(sorted(list(current_prefix) + [item]))
            tid_list = v_data[item]
            
            result[new_itemset] = len(tid_list) / self.n_transactions
            
            suffix_items = []
            new_v_data = {}
            for j in range(i + 1, len(items_to_test)):
                next_item = items_to_test[j]
                intersection = tid_list.intersection(v_data[next_item])
                
                if len(intersection) >= self.min_support_count:
                    suffix_items.append(next_item)
                    new_v_data[next_item] = intersection
            
            if suffix_items:
                self._recurse(suffix_items, set(new_itemset), new_v_data, result)

def generate_rules_eclat(frequent_itemsets, min_conf=0.8):
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2: continue
        
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                
                support_a = frequent_itemsets.get(antecedent)
                if support_a:
                    conf = support / support_a
                    if conf >= min_conf:
                        rules.append({
                            "Rule": f"{antecedent} -> {consequent}",
                            "Support": support,
                            "Confidence": conf
                        })
    return pd.DataFrame(rules)


def Part1(df):
    print("# Part 1: Using the library pyECLAT")
    eclat_instance = ECLAT(data=df, verbose=False)
    
    get_speed, all_supports = eclat_instance.fit(
        min_support=0.6,
        min_combination=1,
        max_combination=5,
        separator=' & '
    )
    
    res_df = pd.DataFrame(all_supports.items(), columns=['Itemset', 'Support'])
    print(res_df.sort_values(by='Support', ascending=False))
    print("\n")

def Part2(df):
    print("# Part 2: Vertical Apriori from scratch")
    records = []
    for row in df.values:
        records.append([item for item in row if str(item) != 'nan'])
        
    model = VerticalAprioriModel(records, min_support_ratio=0.6)
    frequent_itemsets = model.run_eclat()
    
    print("The popular sets founded:")
    itemset_df = pd.DataFrame([
        {'Itemset': list(k), 'Support': v} for k, v in frequent_itemsets.items()
    ])
    print(itemset_df.sort_values(by='Support', ascending=False))

    print("\nCombined rules (Confidence >= 80%):")
    rules = generate_rules_eclat(frequent_itemsets, min_conf=0.8)
    print(rules)

if __name__ == "__main__":
    gen = DataGenerator()
    df = gen.load_data()
    
    Part1(df)
    Part2(df)