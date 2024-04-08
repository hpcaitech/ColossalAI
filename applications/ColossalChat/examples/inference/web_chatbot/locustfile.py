from locust import HttpUser, task

samples = [
    [
        dict(
            instruction="Who is the best player in the history of NBA?",
            response="The best player in the history of the NBA is widely considered to be Michael Jordan. He is one of the most successful players in the league, having won 6 NBA championships with the Chicago Bulls and 5 more with the Washington Wizards. He is a 5-time MVP, 1",
        ),
        dict(instruction="continue this talk", response=""),
    ],
    [
        dict(instruction="Who is the best player in the history of NBA?", response=""),
    ],
]


class GenerationUser(HttpUser):
    @task
    def generate(self):
        for sample in samples:
            data = {"max_new_tokens": 64, "history": sample}
            with self.client.post("/generate", json=data, catch_response=True) as response:
                if response.status_code in (200, 406):
                    response.success()
                else:
                    response.failure("Response wrong")
