from locust import HttpUser, task


class GenerationUser(HttpUser):
    @task
    def generate(self):
        prompt = "Question: What is the longest river on the earth? Answer:"
        for i in range(4, 9):
            data = {"max_tokens": 2**i, "prompt": prompt}
            with self.client.post("/generation", json=data, catch_response=True) as response:
                if response.status_code in (200, 406):
                    response.success()
                else:
                    response.failure("Response wrong")
