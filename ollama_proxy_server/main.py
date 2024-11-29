import argparse
import configparser
import csv
import datetime
import json
import logging
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Queue
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import requests

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_config(filename, config_string=None, default_timeout=300):
    config = configparser.ConfigParser()
    if config_string is None or config_string == "":
        config.read(filename)
    else:
        config.read_string(config_string)
    servers = []
    for name in config.sections():
        try:
            timeout = int(config[name].get("timeout", default_timeout))
            if timeout <= 0:
                raise ValueError
        except (ValueError, TypeError):
            logger.info(
                "Invalid timeout value for server %s. Using default %d seconds.",
                name,
                default_timeout,
            )
            timeout = default_timeout

        server_info = {
            "url": config[name]["url"],
            "queue": Queue(),
            "queue_size": int(config[name].get("queue_size", 1)),
            "models": [model.strip() for model in config[name]["models"].split(",")],
            "timeout": timeout,
        }
        servers.append((name, server_info))
    if config_string is None or config_string == "":
        logger.debug("Loaded servers from %s: %s", filename, servers)
    else:
        logger.debug("Loaded servers from env config string")
    return servers


def read_users_from_lines(lines, location):
    authorized_users = {}
    for line in lines:
        if line.strip() == "":
            continue
        try:
            user, key, role, models = line.strip().split(";")
            authorized_users[user] = {"key": key, "role": role, "models": models.split(",")}
        except Exception as e:
            logger.debug("User entry broken, Exception: %s", e)
            logger.info("User entry broken form %s: %s", location, line.strip())
    return authorized_users


def get_authorized_users(filename, users_env=None):
    # If file is available load from file
    try:
        logger.debug("Loading authorized users from %s", filename)
        with open(filename, "r", encoding="utf8") as f:
            file_lines = f.readlines()
    except FileNotFoundError:
        logger.debug("No authorized users file found")
        file_lines = []
    authorized_users_file = read_users_from_lines(file_lines, filename)
    if users_env:
        lines = users_env.split("\n")
        authorized_users_env = read_users_from_lines(lines, 'Env')
    else:
        authorized_users_env = {}
    # Env has priority, merge two dictionaries with env in priority
    authorized_users = {**authorized_users_file, **authorized_users_env}
    logger.debug(
        "Loaded authorized users from File and Env(priority): %s",
        str(list(authorized_users.keys())),
    )
    logger.debug("RAW Authorized users: %s", authorized_users)
    return authorized_users


def check_sys_env(name, default=None):
    if name in os.environ:
        logger.debug("Using environment variable %s, %s", name, os.environ[name])
        return os.environ[name]
    else:
        if default is not None:
            return default
        return None


def get_version():
    try:
        with open("GIT_VERSION_TAG.txt", "r") as f:
            version = f.read().strip()
    except FileNotFoundError:
        version = "unknown"
    try:
        with open("GIT_VERSION_HASH.txt", "r") as f:
            hash = f.read().strip()
    except FileNotFoundError:
        try:
            hash = os.popen('git rev-parse --verify HEAD').read().strip()
        except Exception as e:
            logger.debug("Exception: %s", e)
            hash = "unknown"
    return f"Version:{version}, Git-Hash:{hash}"


def parse_args(home_folder=""):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=home_folder + "config.ini", help="Path to the config file"
    )
    parser.add_argument(
        "--log_path", default=home_folder + "access_log.txt", help="Path to the access log file"
    )
    parser.add_argument(
        "--users_list",
        default=home_folder + "authorized_users.txt",
        help="Path to the authorized users list",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number for the server"
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=3,
        help="Number of retry attempts for failed calls",
    )
    parser.add_argument(
        "-d", "--deactivate_security", action="store_true", help="Deactivates security"
    )
    args = parser.parse_args()

    return args


def get_running_models(class_object, path, get_params, post_data_dict, backend_headers):
    logger.debug("ps servers")
    server_ps = {}
    for server in class_object.servers:
        response = class_object.send_request_with_retries(
            server[1], path, get_params, post_data_dict, backend_headers
        )

        if response:
            logger.debug("Received response from server %s", server[0])
            try:
                server_ps[server[0]] = json.loads(response.content)['models']
                server_ps[server[0] + '_error'] = "None"
            except KeyError:
                logger.debug("No models found in response from server %s", server[0])
                server_ps[server[0]] = []
                server_ps[server[0] + '_error'] = "JSON decode error"
                continue
        else:
            server_ps[server[0]] = []
            server_ps[server[0] + '_error'] = "No response"
            logger.debug("Received No response from server %s", server[0])

    return server_ps


def get_available_models(class_object, path, get_params, post_data_dict, backend_headers):
    logger.debug("List supported models")
    models = [
        model for server in class_object.servers for model in server[1]["models"]
    ]
    model_info = []
    server_tags = {}
    if "v1/models" in path:
        # OpenAI
        model_entry_name = "id"
        data_entry_name = "data"
    else:
        # Ollama
        model_entry_name = "name"
        data_entry_name = "models"

    for server in class_object.servers:
        response = class_object.send_request_with_retries(
            server[1], path, get_params, post_data_dict, backend_headers
        )

        if response:
            logger.debug("Received response from server %s", server[0])
            server_tags[server[0]] = {
                model[model_entry_name]: model
                for model in json.loads(response.content)[data_entry_name]
            }
        else:
            logger.debug("Failed to receive response from server %s", server[0])

    for model in models:
        available_servers = [
            server
            for server in class_object.servers
            if model in server[1]["models"]
        ]
        logger.debug(
            "Available servers for model '%s': %s",
            model,
            str([s[0] for s in available_servers]),
        )
        for server in available_servers:
            try:
                model_info.append(server_tags[server[0]][model])
            except Exception as e:
                logger.debug("Model not found, Exception: %s", e)
                logger.warning(
                    "Model %s not found in server %s", model, server[0]
                )
    if "v1/models" in path:
        model_info = {"object": "list", "data": model_info}
    else:
        model_info = {"models": model_info}
    return model_info


def get_best_server(available_servers):
    chosen_server = None
    queue_size = 999
    while chosen_server is None:
        for server in available_servers:
            if (server[1]['queue_size'] - server[1]["queue"].qsize() > 0) & (server[1]["queue"].qsize() < queue_size):
                # Find a server with shorter queue than the previous and with capacity.
                chosen_server = server
                queue_size = server[1]["queue"].qsize()
        if chosen_server is None:
            time.sleep(0.01)
    return chosen_server


def ring_buffer(data, new_data):
    data.pop(0)
    data.append(new_data)
    return data


def add_access_log_entry(
    log_path,
    event,
    user,
    ip_address,
    access,
    server,
    nb_queued_requests_on_server,
    error="",
    input_tokens=0,
    output_tokens=0,
):
    log_file_path = Path(log_path)
    logger.debug("Adding log entry")
    if not log_file_path.exists():
        with open(
            log_file_path, mode="w", newline="", encoding="utf8"
        ) as csvfile:
            fieldnames = [
                "time_stamp",
                "event",
                "user_name",
                "ip_address",
                "access",
                "server",
                "nb_queued_requests_on_server",
                "input_tokens",
                "output_tokens",
                "error",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(log_file_path, mode="a", newline="", encoding="utf8") as csvfile:
        fieldnames = [
            "time_stamp",
            "event",
            "user_name",
            "ip_address",
            "access",
            "server",
            "nb_queued_requests_on_server",
            "input_tokens",
            "output_tokens",
            "error",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        row = {
            "time_stamp": str(datetime.datetime.now()),
            "event": event,
            "user_name": user,
            "ip_address": ip_address,
            "access": access,
            "server": server,
            "nb_queued_requests_on_server": nb_queued_requests_on_server,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error,
        }
        logger.debug("Log: %s", str(row))
        writer.writerow(row)


def main_loop():
    logger.info("Ollama Proxy server")
    logger.info("Author: ParisNeo, rcastberg")
    logger.info("Version: %s", get_version())
    home_folder = check_sys_env("OP_HOME", default="")
    logger.info("Home folder: %s", home_folder)
    args = parse_args()
    logger.debug("Default Arguments: %s", args)

    class RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # Force HTTP/1.1 responses

        # Class variables to access arguments and servers
        retry_attempts = int(check_sys_env(
            "OP_RETRY_ATTEMPTS", default=args.retry_attempts
        ))  # Sys env has priority
        servers = get_config(
            args.config,
            config_string=check_sys_env("OP_SERVERS", "").replace(";", "\n"),
        )
        authorized_users = get_authorized_users(
            args.users_list, check_sys_env("OP_AUTHORIZED_USERS")
        )
        deactivate_security = check_sys_env(
            "OP_DEACTIVATE_SECURITY", default=args.deactivate_security
        )
        log_path = check_sys_env("OP_LOG_PATH", default=args.log_path)
        logger.debug(
            f"Start up parameters: retry_attempts={retry_attempts}, servers={servers}, authorized_users={authorized_users}, deactivate_security={deactivate_security}, log_path={log_path}"
        )

        def send_header(self, keyword, value):
            # Remove duplicate data in headers
            if keyword.lower() == 'date' and b'Date' in b''.join(self._headers_buffer):
                return
            super().send_header(keyword, value)

        def _send_response(self, response, stream=True):
            if stream:
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.lower() not in [
                        "content-length",
                        "transfer-encoding",
                        "content-encoding",
                    ]:
                        self.send_header(key, value)
                        logger.debug('Sending Header: %s:%s', key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                chunks = [b"", b"", b""]
                count = 0
                eval_data = None
                try:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            count += 1
                            self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                            chunks = ring_buffer(chunks, chunk)
                            if b"eval_count" in chunks[1]:
                                eval_data = chunks
                    if not eval_data:
                        eval_data = chunks
                    self.wfile.write(b"0\r\n\r\n")
                except BrokenPipeError:
                    pass
                except Exception as e:
                    logging.error("An unexpected error occurred: %s", {e})
            elif not stream:
                content_length = len(response.content)
                # No streaming
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(content_length))
                self.end_headers()

                try:
                    self.wfile.write(response.content)
                    self.wfile.flush()

                    eval_data = response.content
                    count = len(eval_data)
                except BrokenPipeError:
                    eval_data = response.content
                    count = len(eval_data)
                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
            logger.debug('Eval_data content %s', str(eval_data))
            logger.debug("Curl string: %s", self.curl_string)
            if eval_data is not None:
                return b"".join(eval_data), count
            else:
                return eval_data, count

        def send_simple_response(self, content, code=200, response_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", response_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            self.log_request()
            self.proxy()

        def do_POST(self):
            self.log_request()
            self.proxy()

        def _validate_user_and_key(self):
            try:
                # Extract the bearer token from the headers
                auth_header = self.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    return False
                token = auth_header.split(" ")[1]
                user, key = token.split(":")
                logger.debug("%s %s", user, key)

                # Check if the user and key are in the list of authorized users
                if self.authorized_users.get(user)["key"] == key:
                    self.user = user
                    self.role = self.authorized_users.get(user)["role"]
                    self.models = self.authorized_users.get(user)["models"]
                    return True
                else:
                    self.user = "unknown"
                    self.role = "unknown"
                    self.models = []
                return False
            except Exception as e:
                logger.debug("User parse, Exception: %s", e)
                return False

        def is_server_available(self, server_info):
            self.timeout = 20
            try:
                # Attempt to send a HEAD request to the server's URL with a short timeout
                response = requests.head(server_info["url"], timeout=self.timeout)
                return response.status_code == 200
            except Exception as e:
                logger.debug("Server not available, Exception %s", e)
                return False

        def send_request_with_retries(
            self, server_info, path, get_params, post_data_dict, backend_headers
        ):
            for attempt in range(self.retry_attempts):
                try:
                    # Send request to backend server with timeout
                    logger.debug(
                        "Attempt %d forwarding request to %s",
                        attempt + 1,
                        server_info["url"] + path,
                    )
                    response = requests.request(
                        self.command,
                        server_info["url"] + path,
                        params=get_params,
                        json=post_data_dict if post_data_dict else None,
                        stream=post_data_dict.get("stream", True),  # Content is set to streaming unless user specifically asks.
                        headers=backend_headers,
                        timeout=server_info["timeout"],
                    )
                    logger.debug(
                        "Received response with status code %d", response.status_code
                    )
                    return response
                except requests.Timeout:
                    logger.debug(
                        "Timeout on attempt %d forwarding request to %s",
                        attempt + 1,
                        server_info["url"],
                    )
                except Exception as ex:
                    logger.debug(
                        "Error on attempt %d forwarding request: %s", attempt + 1, ex
                    )
            return None  # If all attempts failed

        def proxy(self):
            self.curl_string = "curl $OLLAMA_HOST"
            self.user = "unknown"
            url = urlparse(self.path)
            logger.debug("URL: %s", url)
            path = url.path
            self.curl_string += str(path)
            if path == "/":
                self.send_simple_response(b"Ollama is running")
                return
            if path == "/health":
                self.send_simple_response(b"OK")
                return
            if not self.deactivate_security and not self._validate_user_and_key():
                logger.warning("User is not authorized")
                client_ip, client_port = self.client_address
                # Extract the bearer token from the headers
                auth_header = self.headers.get("Authorization")
                logger.debug("Auth header: %s", auth_header)
                logger.debug("Client headers: %s", self.headers)
                if not auth_header or not auth_header.startswith("Bearer "):
                    add_access_log_entry(
                        log_path=self.log_path,
                        event="rejected",
                        user="unknown",
                        ip_address=client_ip,
                        access="Denied",
                        server="None",
                        nb_queued_requests_on_server=-1,
                        error="Authentication failed",
                    )
                    logger.debug("No Bearer authentication token provided")
                else:
                    token = auth_header.split(" ")[1]
                    add_access_log_entry(
                        log_path=self.log_path,
                        event="rejected",
                        user=token,
                        ip_address=client_ip,
                        access="Denied",
                        server="None",
                        nb_queued_requests_on_server=-1,
                        error="Authentication failed",
                    )
                    logger.debug("User authentication token not accepted")
                self.send_simple_response(b"No or Invalid authentication token provided", 403)
                return
            get_params = parse_qs(url.query) or {}

            for i in self.headers:
                self.curl_string += " -H \"" + str(i) + ": " + self.headers[i] + "\""
            client_ip, client_port = self.client_address

            # Prepare headers for the backend request
            backend_headers = dict(self.headers)
            # Remove 'Authorization' header
            backend_headers.pop("Authorization", None)
            backend_headers.pop("Host", None)

            # Log the incoming request
            logger.debug("Incoming request from %s:%s", client_ip, str(client_port))
            logger.debug("Request method: %s", self.command)
            logger.debug("Request path: %s", path)
            logger.debug("Query parameters: %s", get_params)

            if self.command == "POST":
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length)
                post_data_dict = {}
                try:
                    post_data_str = post_data.decode("utf-8")
                    post_data_dict = json.loads(post_data_str)
                    logger.debug("POST data: %s", post_data_dict)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning("Failed to decode POST data: %s", e)
                    post_data_dict = {}
                self.curl_string += "-d \"" + post_data_str.replace("\"", "\'") + "\""
            else:
                post_data = None
                post_data_dict = {}

            stripped_path = re.sub("/$", "", path)
            try:
                streamed_request = post_data_dict.get("stream", True)
            except AttributeError:
                # If the request is not a dictionary, it is not a streamed request
                streamed_request = False

            # Endpoints that require model-based routing
            model_based_endpoints = [
                "/api/generate",
                "/api/chat",
                "/api/chat/completions",
                "/generate",
                "/chat",
                "/v1/chat/completions",
                "/v1/completions",
            ]
            if stripped_path in model_based_endpoints:
                # Extract model from post_data or get_params
                model = post_data_dict.get("model")
                if not model:
                    model = get_params.get("model", [None])[0]

                logger.debug("Extracted model: %s", model)

                if not model:
                    # Model is required for these endpoints
                    self.send_simple_response(b"Missing 'model' in request", 400, "text/plain")
                    logger.info("Missing 'model' in request")
                    return
                if model not in self.models and '*' not in self.models:
                    # User is not authorized to use the requested model
                    self.send_simple_response(b"User is not authorized to use the requested model", 403, "text/plain")
                    logger.info("User is not authorized to use the requested model")
                    return
                # Filter servers that support the requested model
                available_servers = [
                    server for server in self.servers if model in server[1]["models"]
                ]

                # Wait for server queue to fall below threshold.
                if not available_servers:
                    # No server supports the requested model
                    self.send_simple_response(b"No servers support the requested model.", 503, "text/plain")
                    logger.info("No servers support model '%s'", model)
                    return
                else:
                    logger.debug(
                        "Available servers for model '%s': %s",
                        model,
                        str([s[0] for s in available_servers]),
                    )

                # Try to find an available server
                response = None

                while available_servers:
                    # Find the server with the lowest queue size among available_servers, divide by the queue size to prioritize more powerful server
                    min_queued_server = get_best_server(available_servers)

                    if not self.is_server_available(min_queued_server[1]):
                        logger.info("Server %s is not available.", min_queued_server[0])
                        available_servers.remove(min_queued_server)
                        continue
                    que = min_queued_server[1]["queue"]
                    try:
                        que.put_nowait(1)
                        add_access_log_entry(
                            log_path=self.log_path,
                            event="gen_request",
                            user=self.user,
                            ip_address=client_ip,
                            access="Authorized",
                            server=min_queued_server[0],
                            nb_queued_requests_on_server=que.qsize(),
                        )
                    except Exception as e:
                        logger.debug("Failed to put request in queue: %s", e)
                        add_access_log_entry(
                            log_path=self.log_path,
                            event="gen_error",
                            user=self.user,
                            ip_address=client_ip,
                            access="Authorized",
                            server=min_queued_server[0],
                            nb_queued_requests_on_server=que.qsize(),
                            error="Queue is full",
                        )
                        self.send_simple_response(b"Server is busy. Please try again later.", 503, "text/plain")
                        return
                    # Prepare to store input and output tokens
                    input_tokens = 0
                    output_tokens = 0
                    try:
                        # Send request with retries
                        response = self.send_request_with_retries(
                            min_queued_server[1],
                            path,
                            get_params,
                            post_data_dict,
                            backend_headers,
                        )
                        if response:
                            last_chunk, count = self._send_response(response, stream=streamed_request)
                            if "/v1/" in stripped_path:  # ChatGPT
                                # add stuff
                                try:
                                    info = json.loads(last_chunk.split(b"\n")[-2])
                                except Exception as e:
                                    logger.debug("Exception: %s", e)
                                    try:
                                        info = json.loads(response.content)
                                    except Exception as e2:
                                        logger.debug("Exception: %s", e2)
                                        # Fall back method, return number of words:
                                        # Should eventually be fixed by : https://github.com/ollama/ollama/pull/6784
                                        # See https://github.com/ollama/ollama/issues/4448
                                        info = {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
                                        info["usage"]["prompt_tokens"] = sum([len(i["content"].split()) for i in post_data_dict["messages"]])
                                        info["usage"]["completion_tokens"] = count
                                try:
                                    input_tokens = info["usage"]["prompt_tokens"]
                                    output_tokens = info["usage"]["completion_tokens"]
                                except json.decoder.JSONDecodeError:
                                    logger.info(
                                        "Failed to parse response: %s", response.content
                                    )
                                    logger.info("Response: %s", response)
                                break
                            else:
                                try:
                                    info = json.loads(last_chunk.split(b"\n")[-2])
                                except IndexError:
                                    info = json.loads(response.content)
                                except AttributeError:
                                    # Request was canceled? FIX
                                    info = ""
                                    logger.warning("Request was possibly canceled, not counting")
                                try:
                                    input_tokens = info["prompt_eval_count"]
                                    output_tokens = info["eval_count"]
                                except json.decoder.JSONDecodeError:
                                    logger.debug(
                                        "Failed to parse response: %s", response.content
                                    )
                                    logger.debug("Response: %s", response)
                                except Exception as e:
                                    logger.debug(
                                        "Failed to find tokens usage, response: %s",
                                        response.content,
                                    )
                                    logger.debug("Response: %s", response)
                                    logger.debug("Exception: %s", e)
                                break
                        else:
                            # All retries failed, try next server
                            logger.warning(
                                "All retries failed for server %s", min_queued_server[0]
                            )
                            available_servers.remove(min_queued_server)
                    finally:
                        try:
                            que.get_nowait()
                            add_access_log_entry(
                                log_path=self.log_path,
                                event="gen_done",
                                user=self.user,
                                ip_address=client_ip,
                                access="Authorized",
                                server=min_queued_server[0],
                                nb_queued_requests_on_server=que.qsize(),
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            )
                        except Exception as e:
                            logger.debug("Write to log, Exception: %s", e)
                if not response:
                    # No server could handle the request
                    self.send_simple_response(b"No available servers could handle the request.", 503, "text/plain")
            elif stripped_path in ["/api/tags", "/v1/models"]:
                model_info = get_available_models(self, path, get_params, post_data_dict, backend_headers)
                model_info = json.dumps(model_info).encode("utf-8")
                self.send_simple_response(model_info)
            elif stripped_path in ["/api/pull", "/api/delete", "/api/push",
                                   "/api/copy", "/api/create"]:
                self.send_simple_response("Unsupported in proxy", 503)
            elif stripped_path in ["/local/delete_user"]:
                if self.role == "admin":
                    try:
                        user = post_data_dict["user"]
                    except KeyError:
                        self.send_simple_response(b"Missing required fields in request", 400, "text/plain")
                        return
                    if user in self.authorized_users:
                        del self.authorized_users[user]

                        # Write the new user to the authorized users file
                        with open(args.users_list, "w", encoding="utf8") as f:
                            for user, data in self.authorized_users.items():
                                f.write(
                                    f"{user}:{data['key']}:{data['role']}:{','.join(data['models'])}\n"
                                )
                        self.send_simple_response(b"User deleted", 200, "text/plain")
                    else:
                        self.send_simple_response(b"User does not exist", 400, "text/plain")

            elif stripped_path in ["/local/add_user"]:
                # Check the role of the user:
                if self.role == "admin":
                    try:
                        user = post_data_dict["user"]
                        role = post_data_dict["role"]
                        key = post_data_dict["key"]
                        models = post_data_dict["models"]
                    except KeyError:
                        self.send_simple_response(b"Missing required fields in request", 400, "text/plain")
                        return
                    if user in self.authorized_users:
                        self.send_simple_response(b"User already exists, updating", 400, "text/plain")
                        return

                    self.authorized_users[user] = {"key": key, "role": role, "models": models}
                    self.send_simple_response(b"User added", 200, "text/plain")
                    # Write the new user to the authorized users file
                    with open(args.users_list, "w", encoding="utf8") as f:
                        for user, data in self.authorized_users.items():
                            f.write(
                                f"{user};{data['key']};{data['role']};{','.join(data['models'])}\n"
                            )
                else:
                    self.send_simple_response(b"Unauthorized, %s, user not admin" % self.user, 403)
            elif stripped_path in ["/local/add_bulk_users"]:
                # Check the role of the user:
                import_Status = {}
                if self.role == "admin":
                    for user_info in post_data_dict:
                        try:
                            user = user_info["user"]
                            role = user_info["role"]
                            key = user_info["key"]
                            models = user_info["models"].split(',')
                            import_Status[user] = True
                            self.authorized_users[user] = {"key": key, "role": role, "models": models}
                        except KeyError:
                            import_Status[user] = False
                            continue
                    if post_data_dict == {}:
                        self.send_simple_response(b"Invalid json data, Failed to add users", 400, "text/plain")
                    elif all(import_Status.values()):
                        self.send_simple_response(b"Users added, %s" % json.dumps(import_Status).encode('utf-8'), 200, "text/plain")
                    else:
                        self.send_simple_response(b"Failed to add users, %s" % json.dumps(import_Status).encode('utf-8'), 400, "text/plain")
                    # Write the new user to the authorized users file
                    with open(args.users_list, "w", encoding="utf8") as f:
                        for user, data in self.authorized_users.items():
                            f.write(
                                f"{user};{data['key']};{data['role']};{','.join(data['models'])}\n"
                            )
                else:
                    response_data = "Unauthorized, %s, user not admin" % self.user
                    self.send_simple_response(response_data.encode('utf-8'), 403)
            elif stripped_path in ["/local/reload_config"]:
                # Update the values in the main class.  This is a hack to allow the server to reload the config file.
                RequestHandler.servers = get_config(args.config, config_string=check_sys_env("OP_SERVERS", "").replace(";", "\n"))
                RequestHandler.authorized_users = get_authorized_users(args.users_list, check_sys_env("OP_AUTHORIZED_USERS"))
                current_config = {'servers': self.servers, 'users': [user for user in self.authorized_users]}

                # Remove queues from the config as they are not serializable
                def default(o):
                    return f"<<non-serializable: {type(o).__qualname__}>>"
                current_config = json.dumps(current_config, default=default).encode("utf-8")
                self.send_simple_response(current_config)
            elif stripped_path in ["/local/get_config"]:
                current_config = {'servers': self.servers, 'users': [user for user in self.authorized_users]}

                # Remove queues from the config as they are not serializable
                def default(o):
                    return f"<<non-serializable: {type(o).__qualname__}>>"
                current_config = json.dumps(current_config, default=default).encode("utf-8")
                self.send_simple_response(current_config)
            elif stripped_path in ["/api/ps"]:
                server_ps = get_running_models(self, path, get_params, post_data_dict, backend_headers)
                server_ps = json.dumps(server_ps).encode("utf-8")
                self.send_simple_response(server_ps)
            else:
                logger.warning("Not recognized path, running : %s", stripped_path)
                # For other endpoints, mirror the request to the default server with retries
                default_server = self.servers[0]
                if not self.is_server_available(default_server[1]):
                    self.send_simple_response(b"Default server is not available.", 503, "text/plain")
                    return
                response = self.send_request_with_retries(
                    default_server[1], path, get_params, post_data_dict, backend_headers
                )
                if response:
                    self._send_response(response, stream=streamed_request)
                else:
                    self.send_simple_response(b"Failed to forward request to default server.", 503, "text/plain")

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True  # Gracefully handle shutdown

    logger.info("Starting server")
    port = int(check_sys_env("OP_PORT", args.port))
    server = ThreadedHTTPServer(("", port), RequestHandler)
    logger.info("Running server on port %s", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down the server.")
        server.server_close()


if __name__ == "__main__":
    main_loop()
