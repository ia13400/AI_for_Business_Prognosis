import pandas as pd
from gold_forecasting.hashing import stable_hash,dataframe_hash
def test_stable_hash_ignores_mapping_order(): assert stable_hash({"a":1,"b":2})==stable_hash({"b":2,"a":1})
def test_dataframe_content_changes_hash():
    a=pd.DataFrame({"x":[1,2]}); b=pd.DataFrame({"x":[1,3]}); assert dataframe_hash(a)!=dataframe_hash(b)
