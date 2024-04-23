import ray
import requests


@ray.remote
def send_query(text):
    resp = requests.get("http://localhost:8000/?text={}".format(text))
    return resp.text


test_sentences = [
    "Introduce some landmarks in Beijing",
    "What is the weather today",
    "Coding requires practice and patience",
    "Rainy days inspire cozy reading",
    "Laughter is contagious and heartwarming",
    "Hiking mountains builds strength and resilience",
    "Family bonds grow stronger with time",
    "Science unlocks mysteries of the universe",
    "Music soothes the soul and ignites passion",
    "Artistic expression knows no boundaries",
]

results = ray.get([send_query.remote(text) for text in test_sentences])
print("Result returned:")
for res in results:
    print(res)
