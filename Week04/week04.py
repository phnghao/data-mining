import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
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


def dataFrame2List(data_frame):
    records = []
    for i in range(0, data_frame.shape[0]):
        records.append([str(data_frame.values[i, j]) for j in range(0, data_frame.shape[1]) if str(data_frame.values[i, j]) != 'nan'])
    return records
    
def  records2Transaction(records):
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df

def Apriori(df):
    frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
    print(frequent_itemsets)
    rules = association_rules(
        frequent_itemsets,
        metric="support",
        support_only=True,
        min_threshold=0.1
    )
    print("\nAssociation Rules")
    rules = rules[["antecedents", "consequents", "support"]]
    print(rules)

# Apriori from scratch
class AprioriModel:
    def __init__(self, data, min_support):
        self.data = [set(row) for row in data]
        self.min_support = min_support
        self.total_transactions = len(data)

    def get_support(self, itemset):
        count = 0
        for transaction in self.data:
            if itemset.issubset(transaction):
                count += 1
        return count / self.total_transactions

    def get_frequent_1_itemsets(self):
        items = set()
        for transaction in self.data:
            for item in transaction:
                items.add(frozenset([item]))
        
        frequent_itemsets = {}
        for item in items:
            sup = self.get_support(item)
            if sup >= self.min_support:
                frequent_itemsets[item] = sup
        return frequent_itemsets

    def apriori_gen(self, frequent_itemsets, k):
        candidates = set()
        list_frequent = list(frequent_itemsets.keys())
        n = len(list_frequent)
        
        for i in range(n):
            for j in range(i + 1, n):
                l1 = list(list_frequent[i])
                l2 = list(list_frequent[j])
                l1.sort()
                l2.sort()
                
                if l1[:k-2] == l2[:k-2]:
                    candidate = list_frequent[i] | list_frequent[j]
                    if self.has_infrequent_subset(candidate, frequent_itemsets, k):
                        continue
                    candidates.add(candidate)
        return candidates

    def has_infrequent_subset(self, candidate, frequent_itemsets, k):
        for subset in combinations(candidate, k-1):
            if frozenset(subset) not in frequent_itemsets:
                return True
        return False

    def run(self):
        all_frequent_itemsets = {}
        
        l_k = self.get_frequent_1_itemsets()
        all_frequent_itemsets.update(l_k)
        
        k = 2
        while len(l_k) > 0:
            c_k = self.apriori_gen(l_k, k)
            l_k_new = {}
            for candidate in c_k:
                sup = self.get_support(candidate)
                if sup >= self.min_support:
                    l_k_new[candidate] = sup
            
            if not l_k_new:
                break
                
            all_frequent_itemsets.update(l_k_new)
            l_k = l_k_new
            k += 1
            
        return all_frequent_itemsets

def format_result(frequent_dict):
    res = []
    for itemset, sup in frequent_dict.items():
        res.append({'support': sup, 'itemsets': tuple(sorted(list(itemset)))})
    return pd.DataFrame(res).sort_values(by='support', ascending=False)

from itertools import combinations

def generate_rules(frequent_itemsets, min_conf=0.6):
    rules = []

    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue

        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent

                support_itemset = frequent_itemsets[itemset]
                support_antecedent = frequent_itemsets.get(antecedent, 0)

                if support_antecedent > 0:
                    confidence = support_itemset / support_antecedent

                    if confidence >= min_conf:
                        rules.append({
                            "antecedents": tuple(antecedent),
                            "consequents": tuple(consequent),
                            "support": support_itemset
                        })

    return pd.DataFrame(rules)

def Part1(df):
    print("Part 1")
    print()
    print("Reimplementing in the lab")
    records = dataFrame2List(df)
    data_frame = records2Transaction(records)
    print("Apriori model")
    Apriori(data_frame)
    print()

def Part2(df):
    print("Part 2")
    print()
    print("Implement Apriori model")
    records = dataFrame2List(df)
    model = AprioriModel(records, min_support=0.6)
    result = model.run()
    print("results")
    print(format_result(result))

    rules = generate_rules(result, min_conf=0.6)
    print("\nAssociation Rules")
    print(rules)
    print()

if __name__ == "__main__":
    data_generator = DataGenerator()
    data_frame = data_generator.load_data()
    print(data_frame)
    print()
    Part1(data_frame)
    Part2(data_frame)