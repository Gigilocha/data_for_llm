import numpy as np
import pyarrow.parquet as pq

idx = pq.read_table("output/dataset/train/index.parquet")
first_doc = idx.slice(0, 1).to_pylist()[0]
print("Первый документ:", first_doc)

# Читаем из bin по смещению
data = np.memmap("output/dataset/train/data.bin", dtype=np.int16, mode="r")
tokens = data[first_doc["start_offset"] : first_doc["start_offset"] + first_doc["length"]]
print("Токены в bin:", tokens[:10], "...")