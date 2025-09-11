import requests
import json
import numpy as np
url_list = ["http://10.10.10.10:5437/embedding","http://10.10.10.10:5438/embedding","http://10.10.10.10:5439/embedding"]


for url in url_list:
    #encode query
    texts=""
    payload = {"method": "embed_query", "queries": texts}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        a=np.array(result["embeddings"])
        
    