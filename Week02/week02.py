import pandas as pd
import numpy as np
import unittest

def norm(vector, p = 1):
    vector = np.asarray(vector)

    if p == "inf":
        return np.max(np.abs(vector))
    
    if p < 1:
        raise ValueError("p must be greater than or equal 1")

    else:
        return np.power(np.sum(np.power(np.abs(vector), p)), 1/p)

class TestNorm(unittest.TestCase):
    def test_negative_p(self):
        with self.assertRaises(ValueError):
            norm([1, 2], p = 0)

    def test_l1(self):
        x = range(1, 7)
        l1 = norm(x, p = 1)
        self.assertAlmostEqual(l1, np.linalg.norm(x, ord = 1))

    def test_l2(self):
        x  = [3, 4]
        l2 = norm(x, p = 2)
        self.assertAlmostEqual(l2, np.linalg.norm(x, ord = 2))

    def test_linf(self):
        x = range(0, 15)
        linf = norm(x, p = "inf")
        self.assertEqual(linf, 14)

    def test_l10(self):
        x = range(0, 4)
        l10 = norm(x, p = 10)
        self.assertAlmostEqual(l10, np.linalg.norm(x, ord = 10))

class DataLoader:
    def __init__(self):
        self.ionosphere_path = "./data/ionosphere.data"
        self.kddcup_path = "./data/kddcup.data.corrected"
    
    def ionosphere_load(self):
        data = pd.read_csv(self.ionosphere_path, header = None)
        return data
    
    def kddcup_load(self):
        cols = [1, 2, 3, 41]
        data = pd.read_csv(self.kddcup_path, header = None, usecols = cols)
        categorical_data = data.select_dtypes(include=['object'])
        return categorical_data.drop_duplicates().reset_index(drop=True)
        

def calculate_norm_batches(tensor, p = 1, batch_size = 50):
    tensor = tensor[:batch_size, :]
    diff = tensor[:, None, :] - tensor[None, :, :]
    if p == "inf":
        return np.max(np.abs(diff), axis = 2)
    else:
        return np.power(np.sum(np.power(np.abs(diff), p), axis = 2) , 1/p)
    
class TestNormBatches(unittest.TestCase):
    def setUp(self):
        self.data = np.array([
            [3, 1],
            [1, 2],
            [2, 0],
            [2, 3]
        ])

    def test_l1(self):
        result = calculate_norm_batches(self.data, p = 1, batch_size= 4)
        self.assertEqual(result[0, 1], 3)
        self.assertEqual(result[0, 2], 2)
        self.assertEqual(result[1, 0], 3)
        self.assertEqual(result[0, 0], 0)

    def test_l2_(self):
        result = calculate_norm_batches(self.data, p = 2, batch_size = 4)
        self.assertAlmostEqual(result[0, 1], np.sqrt(5))
        self.assertAlmostEqual(result[2, 3], 3.0)

    def test_linf(self):
        result = calculate_norm_batches(self.data, p = "inf", batch_size = 4)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[0, 2], 1)

    def test_output_shape(self):
        batches = 3
        result = calculate_norm_batches(self.data, p = 2, batch_size = batches)
        self.assertEqual(result.shape, (3, 3))

def overlap(df, batch_size = None):
    if batch_size is None:
        batch_size = len(df)

    subset = df.iloc[: batch_size, :3].values
    matches = (subset[:, None, :] == subset[None, :, :])
    return np.mean(matches, axis = 2)
    
class TestOverlap(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            ['tcp', 'http', 'SF'],
            ['tcp', 'smtp', 'SF'],
            ['udp', 'http', 'REJ'],
            ['udp', 'dns', 'S0']
        ])

    def test_identity(self):
        result = overlap(self.data)
        self.assertEqual(result[0, 0], 1.0)
        self.assertEqual(result[2, 2], 1.0)

    def test_partial_match(self):
        result = overlap(self.data)
        self.assertAlmostEqual(result[0, 1], 2/3)

    def test_low_match(self):
        result = overlap(self.data)
        self.assertAlmostEqual(result[0, 2], 1/3)

    def test_no_match(self):
        result = overlap(self.data)
        self.assertEqual(result[0, 3], 0.0)

    def test_symmetry(self):
        result = overlap(self.data)
        self.assertEqual(result[1, 2], result[2, 1])

    def test_output_shape(self):
        result = overlap(self.data, batch_size=3)
        self.assertEqual(result.shape, (3, 3))

def inverse_frequence(df, batch_size = 100):
    subset = df.iloc[: batch_size, :3]

    log_freq_maps = []
    for col in subset.columns:
        freqs = df[col].value_counts()
        log_freq_maps.append(np.log10(freqs).to_dict())

    log_f_matrix = np.array([
        [log_freq_maps[c][val] for c, val in enumerate(row)]
        for row in subset.values
    ])

    log_prod = log_f_matrix[:, None, :] * log_f_matrix[None, :, :]

    s_mismatch = 1 / (1 + log_prod)
    matches = (subset.values[:, None, :] == subset.values[None, :, :])

    sim_components = np.where(matches, 1.0, s_mismatch)
    return np.mean(sim_components, axis = 2)

class TestInverseFrequency(unittest.TestCase):
    def setUp(self):
        data = []
        for _ in range(100): data.append(['A', 'X', 'Y'])
        for _ in range(10):  data.append(['B', 'X', 'Y'])
        
        self.df = pd.DataFrame(data)

    def test_identity(self):
        result = inverse_frequence(self.df, batch_size=5)
        self.assertAlmostEqual(result[0, 0], 1.0)

    def test_symmetry(self):
        result = inverse_frequence(self.df, batch_size=110)
        self.assertAlmostEqual(result[0, 100], result[100, 0])

    def test_math_logic(self):
        result = inverse_frequence(self.df, batch_size=110)
        
        expected_sim = ((1/(1 + 2*1)) + 1.0 + 1.0) / 3
        self.assertAlmostEqual(result[0, 100], expected_sim, places=4)

    def test_rare_values(self):
        rare_df = pd.concat([self.df, pd.DataFrame([['C', 'X', 'Y']])], ignore_index=True)
        result = inverse_frequence(rare_df, batch_size=111)
        
        self.assertAlmostEqual(result[0, 110], 1.0)


def nearest_neighbors(sim_matrix):
    return np.argsort(-sim_matrix, axis = 1)

def Part1():
    print("="*50)
    print("Part 1:")
    print("="*50)
    data_loader = DataLoader()

    print("# 1. Load the Ionosphere data")
    ionosphere_data = data_loader.ionosphere_load()
    print(ionosphere_data)
    print()

    print("# 2. Drop the end column")
    ionosphere_data.pop(ionosphere_data.columns[-1])
    print(ionosphere_data)
    print()

    print("# 3. Initialize the points point1, point2 corresponding to lines 0, 1, and 2 of the array, and calculate the L1, L2, L_inf norms")
    ionosphere_array = ionosphere_data.values
    print(ionosphere_array)
    print()

    point1 = ionosphere_array[0, :]
    point2 = ionosphere_array[1, :]
    point3 = ionosphere_array[2, :]
    
    # p = 1
    dist1_p1_p2 = norm(point1 - point2, p = 1)
    dist1_p1_p3 = norm(point1 - point3, p = 1)

    # p = 2
    dist2_p1_p2 = norm(point1 - point2, p = 2)
    dist2_p1_p3 = norm(point1 - point3, p = 2)

    # p = inf
    dist_inf_p1_p2 = norm(point1 - point2, p = "inf")
    dist_inf_p1_p3 = norm(point1 - point3, p = "inf")

    print("L1 distance between point1 and point2:", dist1_p1_p2)
    print("L1 distance between point1 and point3:", dist1_p1_p3)

    print("L2 distance between point1 and point2:", dist2_p1_p2)
    print("L2 distance between point1 and point3:", dist2_p1_p3)

    print("L-infinity distance between point1 and point2:", dist_inf_p1_p2)
    print("L-infinity distance between point1 and point3:", dist_inf_p1_p3)

    print()

    print("# 4. Using function to calculate the L1, L2, L_inf norm for 50 fisrt lines")
    print("The L1 norm distance-matrix of the first 50 lines in the Ionosphere Data")
    L1_matrix_dist = calculate_norm_batches(ionosphere_array, p = 1, batch_size=50)
    print(pd.DataFrame(L1_matrix_dist))
    print()
    print("The L2 norm distance-matrix of the first 50 lines in the Ionosphere Data")
    L2_matrix_dist = calculate_norm_batches(ionosphere_array, p = 2, batch_size=50)
    print(pd.DataFrame(L2_matrix_dist))
    print()
    print("The L_inf norm distance-matrix of the first 50 lines in the Ionosphere Data")
    L_inf_matrix_dist = calculate_norm_batches(ionosphere_array, p = "inf", batch_size=50)
    print(pd.DataFrame(L_inf_matrix_dist))
    print()

def Part2():
    print("="*50)
    print("Part 2:")
    print("="*50)
    data_loader = DataLoader()

    kddcup_data = data_loader.kddcup_load()
    print("# 1.  Load the KDD Cup data")
    print(kddcup_data.head(10))
    print()

    print("# 2. Find the nearest neighbors using Overlap metric ")
    sim_matrix_overlap = overlap(kddcup_data, batch_size = 100)
    nn_overlap = nearest_neighbors(sim_matrix = sim_matrix_overlap)
    print(nn_overlap)
    print()

    print("# 3. Find the nearest neighbors using Inverse Frequency")
    sim_matrix_if = inverse_frequence(kddcup_data, batch_size = 100)
    nn_if = nearest_neighbors(sim_matrix = sim_matrix_if)
    print(nn_if)
    print()

def main():
    unittest.main(argv = [''], verbosity = 2, exit = False)
    Part1()

    Part2()

if __name__ == "__main__":
    main()