import logging
import os
import signal
import subprocess
import time
import unittest
from pathlib import Path
from dotenv import load_dotenv

import requests

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

env_path = Path(".") / ".env-test"
load_dotenv(dotenv_path=env_path, verbose=True)

OLLAMA_SERVER = os.getenv("OLLAMA_TEST_SERVER")
OLLMA_MODEL = os.getenv("OLLAMA_TEST_MODEL")
PROXY_SERVER = os.getenv("PROXY_SERVER")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
PROXY_PORT = os.getenv("OP_PORT")
AUTH_USER = os.getenv("OP_AUTHORIZED_USERS")


class CheckOllamaRunning(unittest.TestCase):
    def test_generate(self):
        url = f'{OLLAMA_SERVER}/api/generate'
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "model": OLLMA_MODEL,
            "prompt": "Why is the sky blue?",
            "stream": False
        }
        logger.info("Testing Ollama Server at %s", url)
        response = requests.post(url, json=data, headers=headers, timeout=30)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn('response', response.json())
        self.assertIsInstance(response.json()['response'], str)
        self.assertGreater(len(response.json()['response']), 10)

class TestRestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #Make sure Ollama is running:
        url = f'{OLLAMA_SERVER}/'
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception("Ollama Server is not running")
        logger.debug("Ollama Server is running")
        # Start the backend server
        logger.debug('Starting Proxy Server')
        cls.backend_process = subprocess.Popen([
            'python3', 'ollama_proxy_server/main.py'
        ])
        logger.debug('Waiting for Proxy Server to start')
        # Wait for the server to start
        for i in range(20):
            time.sleep(2)
            url = f'http://{PROXY_SERVER}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.debug("Proxy Server is running")
                return
        logger.debug("Proxy server failed to start")
        raise Exception("Unable to start Proxy Server after 40 seconds")


    @classmethod
    def tearDownClass(cls):
        # Stop the backend server
        logger.debug('Stopping Proxy Server')
        os.kill(cls.backend_process.pid, signal.SIGTERM)
        cls.backend_process.wait()
        logger.debug('Proxy Server Stopped')

    def test_generate_ollama_generate(self):
        url = f'http://{PROXY_SERVER}/api/generate'
        headers = {
            'Authorization': f'Bearer {AUTH_USER}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "tinyllama:latest",
            "prompt": "Why is the sky blue?",
            "stream": False
        }

        response = requests.post(url, json=data, headers=headers, timeout=30)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn('response', response.json())
        self.assertIsInstance(response.json()['response'], str)
        self.assertGreater(len(response.json()['response']), 10)

    def test_ollama_generate_stream(self):
        url = f'http://{PROXY_SERVER}/api/generate'
        headers = {
            'Authorization': f'Bearer {AUTH_USER}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "tinyllama:latest",
            "prompt": "Why is the sky blue?",
            "stream": True
        }

        response = requests.post(url, json=data, headers=headers, timeout=30)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn(b'response', response.content)
        self.assertIsInstance(response.content.split(b'\n')[-2], bytes)
        self.assertGreater(len(response.content.split(b'\n')), 10)
        
    def test_ollama_tags(self):
        url = f'http://{PROXY_SERVER}/api/tags'
        headers = {
            'Authorization': f'Bearer {AUTH_USER}',
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers, timeout=30)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn(b'models', response.content)

    def test_openai_generate(self):
        url = f'http://{PROXY_SERVER}/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {AUTH_USER}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "tinyllama:latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Why is the sky blue?"
                }
            ]
        }

        response = requests.post(url, json=data, headers=headers, timeout=30)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn('choices', response.json())
        self.assertIsInstance(response.json()['choices'][0]['message']['content'], str)
        self.assertGreater(len(response.json()['choices'][0]['message']['content']), 10)
        
    def test_openai_models(self):
        url = f'http://{PROXY_SERVER}/v1/models'
        headers = {
            'Authorization': f'Bearer {AUTH_USER}',
            'Content-Type': 'application/json'
        }

        response = requests.get(url, headers=headers, timeout=30)

        # Check if the request was successful
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected data structure
        # Adjust the expected structure based on your actual response
        self.assertIn('object', response.json())
        self.assertIn('data', response.json())


if __name__ == '__main__':
    unittest.main(failfast=True)
