"""
This script acts as a proxy to the Ollama backend.
"""

import argparse
import configparser
import csv
import datetime
import json
import logging
import os
import re
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from pathlib import Path
from queue import Queue
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_HEADER = [
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
    "model",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
    "total_duration"
]

MIME_TYPES = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css',
              'json': 'application/json; charset=utf-8', 'txt': 'text/plain',
              'ico': "image/x-icon", 'csv': 'text/csv'}

BEARER = "Bearer "


def get_config(filename, config_string=None, default_timeout=300):
    """Read the server configuration from a file or a string.
    Args:
        filename (_type_): _description_
        config_string (_type_, optional): _description_. Defaults to None.
        default_timeout (int, optional): _description_. Defaults to 300.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    config = configparser.ConfigParser()
    if config_string is None or config_string == "":
        config.read(filename)
    else:
        config.read_string(config_string)
    servers = []
    for name in config.sections():
        try:
            timeout = int(config[name].get("timeout", default_timeout))
            timeout if timeout > 0 else default_timeout
        except (ValueError, TypeError):
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


def read_access_data(filename, require_tokens=False):
    """Read the access data from a CSV file.
    Args:
        filename (_type_): _description_
        require_tokens (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    data = pd.read_csv(filename, encoding='utf-8', delimiter=',', names=CSV_HEADER)
    if data.iloc[0, 0] == 'time_stamp':
        data.drop(index=data.index[0], axis=0, inplace=True)
    data['input_tokens'] = pd.to_numeric(data['input_tokens'], errors='coerce')
    data['output_tokens'] = pd.to_numeric(data['output_tokens'], errors='coerce')
    data['total_duration'] = pd.to_numeric(data['total_duration'], errors='coerce')
    data['eval_duration'] = pd.to_numeric(data['eval_duration'], errors='coerce')
    data['prompt_eval_duration'] = pd.to_numeric(data['prompt_eval_duration'], errors='coerce')
    # Filter to hour level and remove data with no tokens or valid users.
    data.index = pd.to_datetime(data['time_stamp'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')

    if require_tokens:
        data = data[(data["input_tokens"] > 0) & (data["output_tokens"] > 0)]
    return data


def write_config(filename, servers):
    """Write the server configuration to a file.
    Args:
        filename (_type_): _description_
        servers (_type_): _description_
    """
    config = configparser.ConfigParser()
    for name, server in servers:
        config[name] = {
            "url": server["url"],
            "models": ", ".join(server["models"]),
            "timeout": str(server["timeout"]),
            "queue_size": str(server["queue_size"]),
        }
    with open(filename, "w", encoding="utf-8") as f:
        config.write(f)


def read_users_from_lines(lines, location):
    """Read the authorized users from a list of lines.
    Args:
        lines (_type_): _description_
        location (_type_): _description_

    Returns:
        _type_: _description_
    """
    authorized_users = {}
    for line in lines:
        if line.strip() == "":
            continue
        try:
            user, key, role, models = line.strip().split(";")
            authorized_users[user] = {"key": key, "role": role, "models": models.split(",")}
        except ValueError:
            logger.info("User entry broken, Unable to split info")
            logger.debug("User entry broken form %s: %s", location, line.strip())
        except Exception as e:
            logger.debug("User entry broken, Exception %s", e)
            logger.info("User entry broken form %s: %s", location, line.strip())
    return authorized_users


def get_authorized_users(filename, users_env=None):
    """Read the authorized users from a file or environment variable.
    Args:
        filename (_type_): _description_
        users_env (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
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
        lines = re.split('[|\n]', users_env)
        authorized_users_env = read_users_from_lines(lines, 'Env')
    else:
        authorized_users_env = {}
    # Env has priority, merge two dictionaries with env in priority
    authorized_users = {**authorized_users_file, **authorized_users_env}
    logger.debug(
        "Loaded authorized users from File and Env(priority): %s",
        str(list(authorized_users.keys())),
    )
    return authorized_users


def check_sys_env(name, default=None):
    """Check if an environment variable is set, otherwise return the default value.
    Args:
        name (_type_): _description_
        default (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if name in os.environ:
        logger.debug("Using environment variable %s: %s...", name, os.environ[name][0:5])
        return os.environ[name]
    else:
        if default is not None:
            return default
        return None


def get_version():
    """Get the version and git hash from the version files.
    Returns:
        _type_: _description_
    """
    try:
        with open("GIT_VERSION_TAG.txt", "r", encoding="utf-8") as f:
            version = f.read().strip()
    except FileNotFoundError:
        version = "unknown"
    try:
        with open("GIT_VERSION_HASH.txt", "r", encoding="utf-8") as f:
            git_hash = f.read().strip()
    except FileNotFoundError:
        git_hash = os.popen('git rev-parse --verify HEAD').read().strip()
        # if command not found, returns ''
    return f"Version:{version}, Git-Hash:{git_hash}"


def parse_args(home_folder=""):
    """Parse the command line arguments.
    Args:
        home_folder (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    logger.debug("Arg parser, home folder: %s", home_folder)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.path.join(home_folder, "config.ini"),
        help="Path to the config file"
    )
    parser.add_argument(
        "--log_path", default=os.path.join(home_folder, "access_log.txt"),
        help="Path to the access log file"
    )
    parser.add_argument(
        "--users_list",
        default=os.path.join(home_folder, "authorized_users.txt"),
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

    args = parser.parse_args()

    return args


def get_running_models(class_object, path, get_params, post_data_dict, backend_headers):
    """Get the running models from all servers.
    Args:
        class_object (_type_): _description_
        path (_type_): _description_
        get_params (_type_): _description_
        post_data_dict (_type_): _description_
        backend_headers (_type_): _description_

    Returns:
        _type_: _description_
    """
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


def get_available_models(class_object, path, get_params, post_data_dict, backend_headers,
                         filtered_list=True):
    """Get the available models from all servers.
    Args:
        class_object (_type_): _description_
        path (_type_): _description_
        get_params (_type_): _description_
        post_data_dict (_type_): _description_
        backend_headers (_type_): _description_
        filtered_list (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    logger.debug("List supported models")
    models = list({
        model for server in class_object.servers for model in server[1]["models"]
    })  # Get unique models, flatten list of models
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

    # Get complete list of supported models from all servers
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

    # If we want the raw data, for exmaple for the admin page, return the full list
    if not filtered_list:
        return server_tags

    # Filter only the models allowed by the server spesification
    for model in models:
        available_servers = [server for server in class_object.servers if model in server[1]["models"]]
        logger.debug(
            "Available servers for model '%s': %s",
            model,
            str([s[0] for s in available_servers]),
        )
        try:
            current_model = server_tags[available_servers[0][0]][model]
            current_model["servers"] = [server[0] for server in available_servers]
            model_info.append(current_model)
        except IndexError:
            logger.warning("Model %s not found in any server", model)

    if "v1/models" in path:
        model_info = {"object": "list", "data": model_info}
    else:
        model_info = {"models": model_info}
    return model_info


def get_best_server(available_servers):
    """Get the best server based on queue size.
    Args:
        available_servers (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """Ring buffer for data.
    Args:
        data (_type_): _description_
        new_data (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    eval_info=None,
    model="",
):
    """Add an entry to the access log.
    Args:
        log_path (_type_): path for the log file
        event (_type_): event types
        user (_type_): user running query_
        ip_address (_type_): calling IP
        access (_type_): user access level
        server (_type_): server request sent to
        nb_queued_requests_on_server (_type_): number of queries in queue
        error (str, optional): error code raised Defaults to "".
        eval_info (dict, optional): information about evaluation. Defaults to {}.
        model (str, optional): model used Defaults to "".
    """
    if eval_info is None:
        eval_info = {}
    log_file_path = Path(log_path)
    logger.debug("Adding log entry")
    if not log_file_path.exists():
        with open(
            log_file_path, mode="w", newline="", encoding="utf8"
        ) as csvfile:
            fieldnames = CSV_HEADER
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(log_file_path, mode="a", newline="", encoding="utf8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
        row = {
            "time_stamp": str(datetime.datetime.now()),
            "event": event,
            "user_name": user,
            "ip_address": ip_address,
            "access": access,
            "server": server,
            "nb_queued_requests_on_server": nb_queued_requests_on_server,
            "input_tokens": eval_info.get("input_tokens", 0),
            "output_tokens": eval_info.get("output_tokens", 0),
            "error": error,
            "model": model,
            "load_duration": eval_info.get("load_duration", 0),
            "prompt_eval_duration": eval_info.get("prompt_eval_duration", 0),
            "eval_duration": eval_info.get("eval_duration", 0),
            "total_duration": eval_info.get("total_duration", 0),
        }
        logger.debug("Log: %s", str(row))
        writer.writerow(row)


def get_streamed_token_count(chunks, chatgpt=False):
    """Get the token count from the streamed chunks.
    Args:
        chunks (_type_): _description_
        chatgpt (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    eval_info = {}
    # Check if chatgpt api
    if isinstance(chunks, list):
        chunks = b','.join(chunks)
    if chatgpt:
        # Use regex as eval_count is not always in a valid json message
        # ChatGPT doesn't provide the durations, but we need populate the dictionary
        # rb'(?!x)x' should be a fast match that doesn't match anything looking for x and not x
        patterns = {"eval_count": (rb'"completion_tokens":(\d+)', 1, int),
                    "prompt_count": (rb'"prompt_tokens":(\d+)', 1, int),
                    "load_duation": (rb'(?!x)x', 1e9, float),
                    "prompt_eval_duration": (rb'(?!x)x', 1e9, float),
                    "eval_duration": (rb'(?!x)x', 1e9, float),
                    "total_duration": (rb'(?!x)x', 1e9, float)
                    }
        # Search for the pattern in the data
    else:
        # Use regex as eval_count is not always in a valid json message
        patterns = {"eval_count": (rb'"eval_count":(\d+)', 1, int),
                    "prompt_count": (rb'"prompt_eval_count":(\d+)', 1, int),
                    "load_duation": (rb'"load_duration":(\d+)', 1e9, float),
                    "prompt_eval_duration": (rb'"prompt_eval_duration":(\d+)', 1e9, float),
                    "eval_duration": (rb'"eval_duration":(\d+)', 1e9, float),
                    "total_duration": (rb'"total_duration":(\d+)', 1e9, float)
                    }

    for key, (pattern, divisor, convert_type) in patterns.items():
        match = re.findall(pattern, chunks)
        if match:
            eval_info[key] = convert_type(int(match[0]) / divisor)
        else:
            eval_info[key] = 0
    return eval_info


def main_loop(test_mode=False):
    """Main loop for the server."""
    logger.info("Ollama Proxy server")
    logger.info("Author: ParisNeo, rcastberg")
    logger.info("Version: %s", get_version())
    home_folder = check_sys_env("OP_HOME", default="")
    logger.info("Home folder: %s", home_folder)

    args = parse_args(home_folder=home_folder)
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

        log_path = check_sys_env("OP_LOG_PATH", default=args.log_path)

        def send_header(self, keyword, value):
            # Remove duplicate data in headers
            if keyword.lower() == 'date' and b'Date' in b''.join(self._headers_buffer):
                return
            super().send_header(keyword, value)

        def _send_response(self, response, stream=True, chat_gpt=False):
            # Send the response to the LLM
            # Calculate the number of tokens, and if not returned by ollama
            # return the number of words as a rough estimate.
            eval_info = {}
            t0 = time.time()
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

                llm_response = None
                try:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            count += 1
                            self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                            chunks = ring_buffer(chunks, chunk)

                    eval_info = get_streamed_token_count(chunks, chat_gpt)

                    self.wfile.write(b"0\r\n\r\n")
                except BrokenPipeError:
                    pass
                except Exception as e:
                    logging.error("An unexpected error occurred: %s", {e})
            elif not stream:
                content_length = len(response.content)
                # No streaming
                self.send_response(200)
                self.send_header('Content-type', MIME_TYPES['json'])
                self.send_header('Content-Length', str(content_length))
                self.end_headers()

                try:
                    self.wfile.write(response.content)
                    self.wfile.flush()

                    llm_response = response.content
                except BrokenPipeError:
                    llm_response = response.content
                except Exception as e:
                    logging.error("An unexpected error occurred: %s", str(e))
                count = len(llm_response)
                eval_info = get_streamed_token_count(llm_response, chat_gpt)
            t1 = time.time()
            if ('total_duration' not in eval_info) or (eval_info['total_duration'] == 0):
                eval_info['total_duration'] = t1 - t0
            if ('total_count' not in eval_info) or (eval_info['total_count'] == 0):
                eval_info['total_count'] = count
            logger.debug('Eval_count %s', str(eval_info))
            return eval_info

        def send_simple_response(self, content, code=200, response_type=MIME_TYPES['json']):
            """Send a simple response with the given content and response code."""
            self.send_response(code)
            self.send_header("Content-Type", response_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            """Handle GET requests."""
            self.log_request()
            self.proxy()

        def do_POST(self):
            """Handle POST requests."""
            self.log_request()
            self.proxy()

        def _validate_user_and_key(self):
            try:
                # Extract the bearer token from the headers
                auth_header = self.headers.get("Authorization")
                if self.cookie_auth_token:
                    user, key = self.cookie_auth_token.value.split(":")
                elif auth_header and auth_header.startswith(BEARER):
                    token = auth_header.split(" ")[1]
                    user, key = token.split(":")
                else:
                    return False
                logger.debug("%s %s", user, key)

                # Check if the user and key are in the list of authorized users
                if (user in self.authorized_users) and (self.authorized_users.get(user, None)["key"] == key):
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
            """Check if the server is available by sending a HEAD request."""
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
            """Send a request to the backend server with retries.
            Args:
                server_info (_type_): server details
                path (_type_): request path
                get_params (_type_):
                post_data_dict (_type_):
                backend_headers (_type_):

            Returns:
                request.response: request response
            """
            logger.debug("Backend headers: %s", backend_headers)
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
                        # Content is set to streaming unless user specifically asks.
                        stream=post_data_dict.get("stream", True),
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
            """Main proxy function to handle requests."""
            self.user = "unknown"
            url = urlparse(self.path)
            logger.debug("URL: %s", url)
            path = url.path

            # Check for authentication cookie
            cookie_header = self.headers.get("Cookie")
            cookies = SimpleCookie(cookie_header)
            self.cookie_auth_token = cookies.get("auth_token")

            if path == "/":
                self.send_simple_response(b"Ollama is running")
                return
            if path == "/health":
                self.send_simple_response(b"OK")
                return
            if path == "/favicon.ico":
                favicon_path = "ollama_proxy_server/favicon.ico"
                if os.path.exists(favicon_path):
                    with open(favicon_path, "rb") as f:
                        favicon_data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", MIME_TYPES['ico'])
                    self.send_header("Content-Length", str(len(favicon_data)))
                    self.end_headers()
                    self.wfile.write(favicon_data)
                else:
                    self.send_simple_response(b"Favicon not found", 404, MIME_TYPES['txt'])
                return
            if path.startswith("/local"):
                # Check if user is authenticated, if not redirect to login page
                if self._validate_user_and_key():
                    if path == "/local/login":
                        content = "Redirecting to stats page\n".encode('utf8')
                        self.send_response(302)
                        self.send_header("Location", "/local/view_stats.html")
                        self.send_header("Content-Type", MIME_TYPES['txt'])
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                        return
                else:
                    with open("ollama_proxy_server/login.html", "r", encoding="utf-8") as f:
                        file_contents = f.read()
                    self.send_simple_response(file_contents.encode("utf-8"), 200, MIME_TYPES['html'])
                    return
            if not self._validate_user_and_key():
                logger.warning("User is not authorized")
                client_ip, client_port = self.client_address
                # Extract the bearer token from the headers
                auth_header = self.headers.get("Authorization")
                logger.debug("Auth header: %s", auth_header)
                logger.debug("Client headers: %s", self.headers)
                if not auth_header or not auth_header.startswith(BEARER):
                    add_access_log_entry(
                        log_path=self.log_path,
                        event="rejected",
                        user="unknown",
                        ip_address=client_ip,
                        access="Denied",
                        server="None",
                        nb_queued_requests_on_server=-1,
                        error="Authentication failed"
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

            client_ip, client_port = self.client_address

            # Prepare headers for the backend request
            backend_headers = dict(self.headers)
            # Remove 'Authorization' header
            backend_headers.pop("Authorization", None)
            backend_headers.pop("Host", None)
            backend_headers.pop("Origin", None)

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
                    self.send_simple_response(b"Missing 'model' in request", 400, MIME_TYPES["txt"])
                    logger.info("Missing 'model' in request")
                    return
                if model not in self.models and '*' not in self.models:
                    # User is not authorized to use the requested model
                    self.send_simple_response(b"User is not authorized to use the requested model",
                                              403, MIME_TYPES["txt"])
                    logger.info("User is not authorized to use the requested model")
                    return
                # Filter servers that support the requested model
                available_servers = [
                    server for server in self.servers if model in server[1]["models"]
                ]

                # Wait for server queue to fall below threshold.
                if not available_servers:
                    # No server supports the requested model
                    self.send_simple_response(b"No servers support the requested model.",
                                              503, MIME_TYPES["txt"])
                    logger.info("No servers support model '%s'", model)
                    return
                logger.debug(
                    "Available servers for model '%s': %s",
                    model,
                    str([s[0] for s in available_servers]),
                )

                # Try to find an available server
                response = None

                while available_servers:
                    # Find the server with the lowest queue size among available_servers,
                    # divide by the queue size to prioritize more powerful server

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
                        self.send_simple_response(b"Server is busy. Please try again later.", 503, MIME_TYPES["txt"])
                        return
                    # Prepare to store input and output tokens
                    eval_info = {}
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
                            chat_gpt = "/v1/" in stripped_path
                            eval_info = self._send_response(response, stream=streamed_request,
                                                            chat_gpt=chat_gpt)
                            break
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
                                eval_info=eval_info,
                                model=model,
                            )
                        except Exception as e:
                            logger.debug("Write to log, Exception: %s", e)
                if not response:
                    # No server could handle the request
                    self.send_simple_response(b"No available servers could handle the request.",
                                              503, MIME_TYPES["txt"])
            elif stripped_path in ["/api/tags", "/v1/models"]:
                model_info = get_available_models(self, path, get_params, post_data_dict, backend_headers,
                                                  filtered_list=True)
                model_info = json.dumps(model_info).encode("utf-8")
                self.send_simple_response(model_info)
            elif stripped_path == "/api/full_tags":
                model_info = get_available_models(self, "/api/tags", get_params, post_data_dict, backend_headers,
                                                  filtered_list=False)
                model_info = json.dumps(model_info).encode("utf-8")
                self.send_simple_response(model_info)
            elif stripped_path in ["/api/pull", "/api/delete", "/api/push",
                                   "/api/copy", "/api/create"]:
                self.send_simple_response(b"Unsupported in proxy", 503, MIME_TYPES["txt"])
            elif stripped_path == "/api/ps":
                server_ps = get_running_models(self, path, get_params, post_data_dict, backend_headers)
                server_ps = json.dumps(server_ps).encode("utf-8")
                self.send_simple_response(server_ps)
            elif stripped_path.startswith("/local") and (self.role in ["user", "admin"]):
                match = re.search(r'^\/local\/([A-Za-z_-]+\.(html|js))$', path)
                if match:
                    try:
                        with open("ollama_proxy_server/" + match.group(1), "r", encoding="utf-8") as f:
                            file_contents = f.read()
                        self.send_simple_response(file_contents.encode("utf-8"), 200, MIME_TYPES[path.split('.')[-1]])
                    except FileNotFoundError:
                        self.send_simple_response(b"No such file", 404, MIME_TYPES['txt'])
                    return
                elif stripped_path == "/local/server_info":
                    version_items = (item.split(':') for item in get_version().split(','))
                    version_info = {key.strip(): value.strip() for key, value in version_items}
                    self.send_simple_response(json.dumps(version_info).encode("utf-8"), 200)
                    return
                elif stripped_path in ["/local/user_info"]:
                    user_info = self.authorized_users[self.user].copy()
                    del user_info['key']
                    models = get_available_models(self, "/api/tags", get_params, post_data_dict,
                                                  backend_headers, filtered_list=True)
                    models = [m['name'] for m in models['models']]
                    user_info['models'] = models
                    user_info['username'] = self.user
                    user_info['all_models'] = '*' in self.models
                    self.send_simple_response(json.dumps(user_info).encode("utf-8"), 200)
                    return
                elif stripped_path in ["/local", "/local/"]:
                    self.send_response(302)
                    self.send_header("Location", "/local/view_stats.html")
                    self.end_headers()
                    self.wfile.write("Redirecting to stats page".encode("utf-8"))
                elif stripped_path == "/local/get_settings":
                    if self.role == "admin":
                        # Remove objects that cannot be serialized
                        def default(_):
                            return ""
                        self.send_simple_response(json.dumps(self.servers, default=default).encode('utf-8'), 200)
                        return
                elif stripped_path == "/local/push_settings":
                    if self.role == "admin":
                        self.servers = post_data_dict
                        for server in self.servers:
                            server[1]["queue"] = Queue()
                        RequestHandler.servers = self.servers
                        return_data = {"status": "success", "message": "Server data updated successfully."}
                        self.send_simple_response(json.dumps(return_data).encode('utf-8'), 200)
                        write_config(args.config, self.servers)
                        return
                elif stripped_path == "/local/download_stats":
                    if self.role == "admin":
                        with open(self.log_path, 'r', encoding='utf-8') as f:
                            file_contents = f.readlines()
                        self.send_simple_response('\n'.join(file_contents).encode('utf-8'), 200, MIME_TYPES['csv'])
                        return
                elif stripped_path == "/local/json_stats":
                    data = read_access_data(self.log_path, require_tokens=True)

                    data = data.groupby('user_name').resample('1h').sum()[['input_tokens', 'output_tokens']]
                    data = data.reset_index().rename(columns={'date': 'time_stamp'})

                    # Remove data with no tokens
                    data = data[(data['input_tokens'] != 0) & (data['output_tokens'] != 0)]
                    # For non admin users anonmize the data for other users.
                    if self.role != "admin":
                        data.loc[data["user_name"] != self.user, "user_name"] = "Others"
                    self.send_simple_response(str(data.to_json(date_format="iso")).encode("utf-8"), 200)
                    return
                elif stripped_path == "/local/model_stats":
                    data = read_access_data(self.log_path, require_tokens=True)
                    # Filter the data
                    data = data[((data["input_tokens"] > 0) | (data["input_tokens"] > 0))].copy()

                    # Calculate the speed of the models
                    data['total_speed'] = (data['input_tokens'] + data['output_tokens']) / data['total_duration']
                    data['prompt_eval_speed'] = data['input_tokens'] / data['prompt_eval_duration']
                    data['eval_speed'] = data['output_tokens'] / data['eval_duration']

                    # Replace inf with NaN, drop unnessacary columns and convert model to category
                    data.replace([np.inf, -np.inf], np.nan, inplace=True)
                    data['model'] = data['model'].astype('category')
                    data.drop(['error', 'access', 'user_name', 'event', 'ip_address', 'time_stamp'],
                              inplace=True, axis=1)
                    aggregated_data = data.groupby(['model', 'server'], observed=True).resample('1h')
                    aggregated_data = aggregated_data.agg({
                                                          'input_tokens': 'sum',
                                                          'output_tokens': 'sum',
                                                          'total_speed': 'mean',
                                                          'prompt_eval_speed': 'mean',
                                                          'eval_speed': 'mean'
                                                          })
                    aggregated_data = aggregated_data.reset_index().rename(columns={'date': 'time_stamp'})
                    # Remove data with no tokens
                    condition = (aggregated_data['input_tokens'] != 0) & (aggregated_data['output_tokens'] != 0)
                    aggregated_data = aggregated_data[condition]
                    self.send_simple_response(str(aggregated_data.to_json(date_format="iso")).encode("utf-8"), 200)
                    return
                elif stripped_path == "/local/user_dump":
                    if self.role == "admin":
                        data = self.authorized_users
                        self.send_simple_response(json.dumps(data).encode("utf-8"), 200)
                        return
                elif stripped_path == "/local/user_update":
                    if self.role == "admin":
                        self.authorized_users = post_data_dict
                        RequestHandler.authorized_users = self.authorized_users
                        data = pd.read_json(StringIO(post_data.decode('utf-8'))).transpose()
                        data['models'] = data['models'].apply(lambda x: ','.join(map(str, x)))
                        data.to_csv(args.users_list, sep=';', header=False, columns=["key", "role", "models"])
                        return_data = {"status": "success", "message": "User data updated successfully."}
                        self.send_simple_response(json.dumps(return_data).encode('utf-8'), 200)
                        return
                else:
                    self.send_simple_response(b"Page not found", 404, MIME_TYPES['txt'])
                    return
                self.send_simple_response(b"Unauthorized", 403)
                return
            else:
                logger.warning("Not recognized path, running : %s", stripped_path)
                # For other endpoints, mirror the request to the default server with retries
                default_server = self.servers[0]
                if not self.is_server_available(default_server[1]):
                    self.send_simple_response(b"Default server is not available.", 503,
                                              MIME_TYPES["txt"])
                    return
                response = self.send_request_with_retries(
                    default_server[1], path, get_params, post_data_dict, backend_headers
                )
                if response:
                    self._send_response(response, stream=streamed_request)
                else:
                    self.send_simple_response(b"Failed to forward request to default server.",
                                              503, MIME_TYPES["txt"])

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Threaded HTTP server"""
        daemon_threads = True  # Gracefully handle shutdown

    logger.info("Starting server")
    port = int(check_sys_env("OP_PORT", args.port))
    server = ThreadedHTTPServer(("", port), RequestHandler)
    logger.info("Running server on port %s", port)
    if test_mode:
        return server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down the server.")
        server.server_close()


if __name__ == "__main__":
    main_loop()
