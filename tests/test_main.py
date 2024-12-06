import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

env_path = Path(".") / ".env-test"
load_dotenv(dotenv_path=env_path, verbose=True)

OLLAMA_SERVER = os.getenv("OLLAMA_TEST_SERVER")
OLLMA_MODEL = os.getenv("OLLAMA_TEST_MODEL")
PROXY_SERVER = os.getenv("PROXY_SERVER")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
PROXY_PORT = os.getenv("OP_PORT")
AUTH_USER = ':'.join(os.getenv("OP_AUTHORIZED_USERS").split(';')[0:2])


def isdebugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


if isdebugging():
    TIMEOUT = 300
else:
    TIMEOUT = 30


class CheckOllamaRunning(unittest.TestCase):
    @pytest.mark.dependency()
    def test_generate(self):
        url = f"{OLLAMA_SERVER}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"model": OLLMA_MODEL, "prompt": "Why is the sky blue?", "stream": False}
        logger.info("Testing Ollama Server at %s", url)
        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn("response", response.json())
        self.assertIsInstance(response.json()["response"], str)
        self.assertGreater(len(response.json()["response"]), 10)


@pytest.mark.dependency(depends=["CheckOllamaRunning::test_generate"])
class TestRestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Make sure Ollama is running:
        url = f"{OLLAMA_SERVER}/"
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code != 200:
            raise Exception("Ollama Server is not running")
        logger.debug("Ollama Server is running")
        # Start the backend server
        logger.debug("Starting Proxy Server")
        # NEW COMMAND: python -m coverage run -m sample.program
        command = [sys.executable, '-m', 'coverage', 'run', '-m', 'ollama_proxy_server']

        cls.backend_process = subprocess.Popen(command,                 # python -m coverage run -m sample.program
                                               stdout=subprocess.PIPE,  # pipe to stdout
                                               env=os.environ.copy()    # pass parent's environment
                                               )
        logger.debug("Waiting for Proxy Server to start")
        # Wait for the server to start
        for i in range(20):
            time.sleep(2)
            url = f"http://{PROXY_SERVER}"
            try:
                response = requests.get(url, timeout=TIMEOUT)
            except requests.exceptions.ConnectionError:
                continue
            if response.status_code == 200:
                logger.debug("Proxy Server is running")
                return
        logger.debug("Proxy server failed to start")
        raise Exception("Unable to start Proxy Server after 40 seconds")

    @classmethod
    def tearDownClass(cls):
        # Stop the backend server
        logger.debug("Stopping Proxy Server")
        os.kill(cls.backend_process.pid, signal.SIGTERM)
        cls.backend_process.wait()
        logger.debug("Proxy Server Stopped")

    @pytest.mark.dependency(depends=["CheckOllamaRunning::test_generate"])
    def test_01_ollama_tags(self):
        url = f"http://{PROXY_SERVER}/api/tags"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        response = requests.get(url, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn(b"models", response.content)

        # Make sure tinyllama is installed
        models = json.loads(response.content)["models"]
        models = [model['name'] for model in models]
        # Needed for other tests.
        self.assertIn("tinyllama:latest", models, "Tinyllama not detected, Needed for other tests")

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_02_generate_ollama_generate(self):
        url = f"http://{PROXY_SERVER}/api/generate"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tinyllama:latest",
            "prompt": "Why is the sky blue?",
            "stream": False,
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('Transfer-Encoding', response.headers)
        self.assertIn("Content-Length", response.headers)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn("response", response.json())
        self.assertIsInstance(response.json()["response"], str)
        self.assertGreater(len(response.json()["response"]), 10)

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_02_generate_ollama_generate_default_steam(self):
        url = f"http://{PROXY_SERVER}/api/generate"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tinyllama:latest",
            "prompt": "Why is the sky blue?",
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Transfer-Encoding'], 'chunked')
        self.assertNotIn("Content-Length", response.headers)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn(b"response", response.content)
        self.assertIsInstance(response.content.split(b"\n")[-2], bytes)
        self.assertGreater(len(response.content.split(b"\n")), 10)

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_03_ollama_generate_stream(self):
        url = f"http://{PROXY_SERVER}/api/generate"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tinyllama:latest",
            "prompt": "Why is the sky blue?",
            "stream": True,
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Transfer-Encoding'], 'chunked')
        self.assertNotIn("Content-Length", response.headers)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn(b"response", response.content)
        self.assertIsInstance(response.content.split(b"\n")[-2], bytes)
        self.assertGreater(len(response.content.split(b"\n")), 10)

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_02_generate_ollama_chat(self):
        url = f"http://{PROXY_SERVER}/api/chat"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tinyllama:latest",
            "messages": [
                {
                    "role": "user",
                    "content": "why is the sky blue?"
                }
            ],
            "stream": False
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('Transfer-Encoding', response.headers)
        self.assertIn("Content-Length", response.headers)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn("message", response.json())
        self.assertIn('role', response.json()['message'])
        self.assertIn('content', response.json()['message'])
        self.assertEqual(response.json()['message']['role'], 'assistant')

        self.assertIsInstance(response.json()['message']["content"], str)
        self.assertGreater(len(response.json()['message']["content"]), 10)

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_03_ollama_chat_stream(self):
        url = f"http://{PROXY_SERVER}/api/chat"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tinyllama:latest",
            "messages": [
                {
                    "role": "user",
                    "content": "why is the sky blue?"
                }
            ],
            "stream": True
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Transfer-Encoding'], 'chunked')
        self.assertNotIn("Content-Length", response.headers)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn(b"message", response.content)
        message = json.loads(response.content.split(b"\n")[2])

        self.assertIn('role', message['message'])
        self.assertIn('content', message['message'])
        self.assertEqual(message['message']['role'], 'assistant')

        self.assertIsInstance(message['message']["content"], str)
        self.assertGreater(len(message['message']["content"]), 1)

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_04_openai_generate(self):
        url = f"http://{PROXY_SERVER}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tinyllama:latest",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Why is the sky blue?"},
            ],
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn("choices", response.json())
        self.assertIsInstance(response.json()["choices"][0]["message"]["content"], str)
        self.assertGreater(len(response.json()["choices"][0]["message"]["content"]), 10)

    @pytest.mark.dependency(depends=["TestRestAPI::test_04_openai_generate"])
    def test_05_openai_models(self):
        url = f"http://{PROXY_SERVER}/v1/models"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn("object", response.json())
        self.assertIn("data", response.json())

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_06_openai_multiplemessages(self):
        url = f"http://{PROXY_SERVER}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tinyllama:latest",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Why is the sky blue?"},
                {"role": "user", "content": "Why is the sky red at night?"},
            ],
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn("choices", response.json())
        self.assertIsInstance(response.json()["choices"][0]["message"]["content"], str)
        self.assertGreater(len(response.json()["choices"][0]["message"]["content"]), 10)

    @unittest.skip("Long Test, skip for now")
    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_07_ollama_image(self):
        url = f"http://{PROXY_SERVER}/api/generate"
        headers = {
            "Authorization": f"Bearer {AUTH_USER}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llava-llama3:latest",
            "prompt": "What is in this picture?",
            "stream": False,
            "images": ["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"]
        }

        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT * 4)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn(b"response", response.content)
        self.assertIn(b"cartoon", response.content)

    @pytest.mark.dependency(depends=["TestRestAPI::test_01_ollama_tags"])
    def test_08_local_health(self):
        # No authorization should be nessacary
        url = f"http://{PROXY_SERVER}/health"
        response = requests.get(url, timeout=TIMEOUT)

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(failfast=False)
