from locust import HttpUser, between, tag, task


class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    @tag("online-generation")
    @task(5)
    def completion(self):
        self.client.post("/completion", json={"prompt": "hello, who are you? ", "stream": "False"})

    @tag("online-generation")
    @task(5)
    def completion_streaming(self):
        self.client.post("/completion", json={"prompt": "hello, who are you? ", "stream": "True"})

    @tag("online-chat")
    @task(5)
    def chat(self):
        self.client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": "what is 1+1?"},
                ],
                "stream": "False",
            },
        )

    @tag("online-chat")
    @task(5)
    def chat_streaming(self):
        self.client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": "what is 1+1?"},
                ],
                "stream": "True",
            },
        )

    # offline-generation is only for showing the usage, it will never be used in actual serving.
    @tag("offline-generation")
    @task(5)
    def generate_streaming(self):
        self.client.post("/generate", json={"prompt": "Can you help me? ", "stream": "True"})

    @tag("offline-generation")
    @task(5)
    def generate(self):
        self.client.post("/generate", json={"prompt": "Can you help me? ", "stream": "False"})

    @tag("online-generation", "offline-generation")
    @task
    def health_check(self):
        self.client.get("/ping")
