import argparse
import configparser
import csv
import datetime
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Queue
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
import re

import logging
import requests

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_config(filename, default_timeout=300):
    config = configparser.ConfigParser()
    config.read(filename)
    servers = []
    for name in config.sections():  
        try:
            timeout = int(config[name].get('timeout', default_timeout))
            if timeout <= 0:
                print(f"Invalid timeout value for server '{name}'. Using default {default_timeout} seconds.")
                timeout = default_timeout
        except ValueError:
            print(f"Non-integer timeout value for server '{name}'. Using default {default_timeout} seconds.")
            timeout = default_timeout

        server_info = {
            'url': config[name]['url'],
            'queue': Queue(),
            'models': [model.strip() for model in config[name]['models'].split(',')],
            'timeout': timeout
        }
        servers.append((name, server_info))
    print(f"Loaded servers from {filename}: {servers}")
    return servers


def get_authorized_users(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    authorized_users = {}
    for line in lines:
        if line.strip() == "":
            continue
        try:
            user, key = line.strip().split(':')
            authorized_users[user] = key
        except:
            print(f"User entry broken: {line.strip()}")
    print(f"Loaded authorized users from {filename}: {list(authorized_users.keys())}")
    return authorized_users


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.ini", help='Path to the config file')
    parser.add_argument('--log_path', default="access_log.txt", help='Path to the access log file')
    parser.add_argument('--users_list', default="authorized_users.txt", help='Path to the authorized users list')
    parser.add_argument('--port', type=int, default=8000, help='Port number for the server')
    parser.add_argument('--retry_attempts', type=int, default=3, help='Number of retry attempts for failed calls')
    parser.add_argument('-d', '--deactivate_security', action='store_true', help='Deactivates security')
    args = parser.parse_args()

    print("Ollama Proxy server")
    print("Author: ParisNeo")

    class RequestHandler(BaseHTTPRequestHandler):
        # Class variables to access arguments and servers
        retry_attempts = args.retry_attempts
        servers = get_config(args.config)
        authorized_users = get_authorized_users(args.users_list)
        deactivate_security = args.deactivate_security
        log_path = args.log_path

        def add_access_log_entry(self, event, user, ip_address, access, server, nb_queued_requests_on_server, error=""):
            log_file_path = Path(self.log_path)

            if not log_file_path.exists():
                with open(log_file_path, mode='w', newline='') as csvfile:
                    fieldnames = ['time_stamp', 'event', 'user_name', 'ip_address', 'access', 'server', 'nb_queued_requests_on_server', 'error']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

            with open(log_file_path, mode='a', newline='') as csvfile:
                fieldnames = ['time_stamp', 'event', 'user_name', 'ip_address', 'access', 'server', 'nb_queued_requests_on_server', 'error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                row = {'time_stamp': str(datetime.datetime.now()), 'event': event, 'user_name': user, 'ip_address': ip_address, 'access': access, 'server': server, 'nb_queued_requests_on_server': nb_queued_requests_on_server, 'error': error}
                writer.writerow(row)

        def _send_response(self, response):
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in ['content-length', 'transfer-encoding', 'content-encoding']:
                    self.send_header(key, value)
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()

            try:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except BrokenPipeError:
                pass

        def do_GET(self):
            self.log_request()
            self.proxy()

        def do_POST(self):
            self.log_request()
            self.proxy()

        def _validate_user_and_key(self):
            try:
                # Extract the bearer token from the headers
                auth_header = self.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return False
                token = auth_header.split(' ')[1]
                user, key = token.split(':')
                print(user, key)

                # Check if the user and key are in the list of authorized users
                if self.authorized_users.get(user) == key:
                    self.user = user
                    return True
                else:
                    self.user = "unknown"
                return False
            except:
                return False

        def is_server_available(self, server_info):
            self.timeout=20
            try:
                # Attempt to send a HEAD request to the server's URL with a short timeout
                response = requests.head(server_info['url'], timeout=self.timeout)
                return response.status_code == 200
            except:
                return False

        def send_request_with_retries(self, server_info, path, get_params, post_data_dict, backend_headers):
            for attempt in range(self.retry_attempts):
                try:
                    # Send request to backend server with timeout
                    print(f"Attempt {attempt+1} forwarding request to {server_info['url'] + path}")
                    response = requests.request(
                        self.command,
                        server_info['url'] + path,
                        params=get_params,
                        json=post_data_dict if post_data_dict else None,
                        stream=post_data_dict.get("stream", False),
                        headers=backend_headers,
                        timeout=server_info['timeout']
                    )
                    print(f"Received response with status code {response.status_code}")
                    return response
                except requests.Timeout:
                    print(f"Timeout on attempt {attempt+1} forwarding request to {server_info['url']}")
                except Exception as ex:
                    print(f"Error on attempt {attempt+1} forwarding request: {ex}")
            return None  # If all attempts failed

        def proxy(self):
            self.user = "unknown"
            if not self.deactivate_security and not self._validate_user_and_key():
                print('User is not authorized')
                client_ip, client_port = self.client_address
                # Extract the bearer token from the headers
                auth_header = self.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    self.add_access_log_entry(event='rejected', user="unknown", ip_address=client_ip, access="Denied", server="None", nb_queued_requests_on_server=-1, error="Authentication failed")
                else:
                    token = auth_header.split(' ')[1]
                    self.add_access_log_entry(event='rejected', user=token, ip_address=client_ip, access="Denied", server="None", nb_queued_requests_on_server=-1, error="Authentication failed")
                self.send_response(403)
                self.end_headers()  
                return
            url = urlparse(self.path)
            path = url.path
            get_params = parse_qs(url.query) or {}

            client_ip, client_port = self.client_address

            # Prepare headers for the backend request
            backend_headers = dict(self.headers)
            # Remove 'Authorization' header
            backend_headers.pop('Authorization', None)

            # Log the incoming request
            print(f"Incoming request from {client_ip}:{client_port}")
            print(f"Request method: {self.command}")
            print(f"Request path: {path}")
            print(f"Query parameters: {get_params}")

            if self.command == "POST":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                post_data_dict = {}
                try:
                    post_data_str = post_data.decode('utf-8')
                    post_data_dict = json.loads(post_data_str)
                    print(f"POST data: {post_data_dict}")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Failed to decode POST data: {e}")
                    post_data_dict = {}
            else:
                post_data = None
                post_data_dict = {}

            # Extract model from post_data or get_params
            model = post_data_dict.get('model')
            if not model:
                model = get_params.get('model', [None])[0]

            print(f"Extracted model: {model}")

            # Endpoints that require model-based routing
            model_based_endpoints = ['/api/generate', '/api/chat', '/api/chat/completions', '/generate', '/chat']
            stripped_path = re.sub('/$', '', path)
            if stripped_path in model_based_endpoints:
                if not model:
                    # Model is required for these endpoints
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Missing 'model' in request")
                    print("Missing 'model' in request")
                    return
                
                # Filter servers that support the requested model
                available_servers = [server for server in self.servers if model in server[1]['models']]

                if not available_servers:
                    # No server supports the requested model
                    print(f"No servers support model '{model}'.")
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"No servers support the requested model.")
                    return
                else:
                    print(f"Available servers for model '{model}': {[s[0] for s in available_servers]}")

                # Try to find an available server
                response = None

                while available_servers:
                    # Find the server with the lowest queue size among available_servers
                    min_queued_server = min(available_servers, key=lambda s: s[1]['queue'].qsize())
                    
                    if not self.is_server_available(min_queued_server[1]):
                        print(f"Server {min_queued_server[0]} is not available.")
                        available_servers.remove(min_queued_server)
                        continue
                    que = min_queued_server[1]['queue']
                    try:
                        que.put_nowait(1)
                        self.add_access_log_entry(event="gen_request", user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize())
                    except:
                        self.add_access_log_entry(event="gen_error", user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize(), error="Queue is full")
                        self.send_response(503)
                        self.end_headers()
                        self.wfile.write(b"Server is busy. Please try again later.")
                        return

                    try:
                        # Send request with retries
                        response = self.send_request_with_retries(min_queued_server[1], path, get_params, post_data_dict, backend_headers)
                        if response:
                            self._send_response(response)
                            print(f"COST_USER:{self.user} Input:{json.loads(response.content.split()[-1])['prompt_eval_count']} OUTPUT:{json.loads(response.content.split()[-1])['eval_count']}")
                            break  # Success
                        else:
                            # All retries failed, try next server
                            print(f"All retries failed for server {min_queued_server[0]}")
                            available_servers.remove(min_queued_server)
                    finally:
                        try:
                            que.get_nowait()
                            self.add_access_log_entry(event="gen_done", user=self.user, ip_address=client_ip, access="Authorized", server=min_queued_server[0], nb_queued_requests_on_server=que.qsize())
                        except:
                            pass
                if not response:
                    # No server could handle the request
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"No available servers could handle the request.")
            elif stripped_path in ['/api/tags']:
                logger.debug(f"List supported models")
                models = [model for server in self.servers for model in server[1]['models']]
                model_info = []
                server_tags = {}
                for server in self.servers:
                    response = self.send_request_with_retries(server[1], path, get_params, post_data_dict, backend_headers)
                    
                    if response:
                        logger.debug(f'Received response from server {server[0]}')
                    else:
                        logger.debug(f'Received response from server {server[0]}')
                        self.wfile.write(b"Failed to forward request to {}.".format(server[0]))
                        
                    server_tags[server[0]] = {model['name']:model for model in json.loads(response.content)['models']}
                for model in models:
                    available_servers = [server for server in self.servers if model in server[1]['models']]
                    print(f"Available servers for model '{model}': {[s[0] for s in available_servers]}")
                    for server in available_servers:
                        try:
                            model_info.append(server_tags[server[0]][model])
                        except:
                            logger.warning(f"Model {model} not found in server {server[0]}")
                model_info = {"models": model_info}
                self.send_response(200) 
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(model_info).encode('utf-8'))
            elif stripped_path in ['/api/ps']:
                logger.debug(f"ps servers")
                server_ps = {}
                for server in self.servers:
                    response = self.send_request_with_retries(server[1], path, get_params, post_data_dict, backend_headers)
                    
                    if response:
                        logger.debug(f'Received response from server {server[0]}')
                    else:
                        logger.debug(f'Received response from server {server[0]}')
                        self.wfile.write(b"Failed to forward request to {}.".format(server[0]))
                    
                    server_ps[server[0]] = json.loads(response.content)
                    
                self.send_response(200) 
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(server_ps).encode('utf-8'))
            else:
                logger.warning(f"Not recognized path, running : {stripped_path}")
                ## TODO Approve endpoints
                # For other endpoints, mirror the request to the default server with retries
                default_server = self.servers[0]
                if not self.is_server_available(default_server[1]):
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"Default server is not available.")
                    return
                response = self.send_request_with_retries(default_server[1], path, get_params, post_data_dict, backend_headers)
                if response:
                    self._send_response(response)
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"Failed to forward request to default server.")

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True  # Gracefully handle shutdown

    print('Starting server')
    server = ThreadedHTTPServer(('', args.port), RequestHandler)
    print(f'Running server on port {args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the server.")
        server.server_close()


if __name__ == "__main__":
    main()
