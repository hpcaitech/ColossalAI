from locust import HttpUser, between, tag, task


class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    @tag("online-generation")
    @task(5)
    def completion(self):
        self.client.post("/v1/completion", json={"prompt": "hello, who are you? ", "stream": "False"})

    @tag("online-generation")
    @task(5)
    def completion_streaming(self):
        self.client.post("/v1/completion", json={"prompt": "hello, who are you? ", "stream": "True"})

    @tag("offline-generation")
    @task(5)
    def generate_stream(self):
        self.client.post("/generate", json={"prompt": "Can you help me? ", "stream": "True"})

    @tag("offline-generation")
    @task(5)
    def generate(self):
        self.client.post("/generate", json={"prompt": "Can you help me? ", "stream": "False"})

    @tag("online-generation", "offline-generation")
    @task
    def get_models(self):
        self.client.get("/v0/models")
