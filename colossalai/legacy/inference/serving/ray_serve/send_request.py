import ray
import requests


@ray.remote
def send_query(text):
    resp = requests.get("http://localhost:8000/?text={}".format(text))
    return resp.text


test_sentence = "Introduce some landmarks in Beijing"

result = ray.get(send_query.remote(test_sentence))
print("Result returned:")
print(result)
