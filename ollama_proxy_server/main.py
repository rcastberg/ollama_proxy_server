import argparse
import configparser
import csv
import datetime
import json
import logging
import os
import re
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
                logger.info(
                    "Invalid timeout value for server '%s'. Using default %d seconds.",
                    name,
                    default_timeout,
                )
                timeout = default_timeout
        except ValueError:
            logger.info(
                "Non-integer timeout value for server %s. Using default %d seconds.",
                name,
                default_timeout,
            )
            timeout = default_timeout

        server_info = {
            "url": config[name]["url"],
            "queue": Queue(),
            "models": [model.strip() for model in config[name]["models"].split(",")],
            "timeout": timeout,
        }
        servers.append((name, server_info))
    if config_string is None:
        logger.debug("Loaded servers from %s: %s", filename, servers)
    else:
        logger.debug("Loaded servers from env config string")
    return servers


def get_authorized_users(filename, users_env=None):
    if users_env:
        lines = users_env.replace(";", "\n").split("\n")
    else:
        with open(filename, "r", encoding="utf8") as f:
            lines = f.readlines()
    authorized_users = {}
    for line in lines:
        if line.strip() == "":
            continue
        try:
            user, key = line.strip().split(":")
            authorized_users[user] = key
        except Exception as e:
            logger.degug("User entry broken, Exception: %s", e)
            logger.info("User entry broken: %s", line.strip())
    logger.debug(
        "Loaded authorized users from %s: %s",
        filename,
        str(list(authorized_users.keys())),
    )
    return authorized_users


def check_sys_env(name, default=None):
    if name in os.environ:
        logger.debug("Using environment variable %s", name)
        return os.environ[name]
    else:
        if default is not None:
            return default
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.ini", help="Path to the config file"
    )
    parser.add_argument(
        "--log_path", default="access_log.txt", help="Path to the access log file"
    )
    parser.add_argument(
        "--users_list",
        default="authorized_users.txt",
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

    logger.info("Ollama Proxy server")
    logger.info("Author: ParisNeo, rcastberg")

    class RequestHandler(BaseHTTPRequestHandler):
        # Class variables to access arguments and servers
        retry_attempts = check_sys_env(
            "OP_RETRY_ATTEMPTS", default=args.retry_attempts
        )  # Sys env has priority
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

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.user = None

        def add_access_log_entry(
            self,
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
            log_file_path = Path(self.log_path)
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

        def ring_buffer(self, data, new_data):
            data.pop(0)
            data.append(new_data)
            return data

        def _send_response(self, response):
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in [
                    "content-length",
                    "transfer-encoding",
                    "content-encoding",
                ]:
                    self.send_header(key, value)
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
                        chunks = self.ring_buffer(chunks, chunk)
                        if b"eval_count" in chunks[1]:
                            eval_data = chunks
                if not eval_data:
                    eval_data = chunks
                self.wfile.write(b"0\r\n\r\n")
            except BrokenPipeError:
                pass
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
            return b"".join(eval_data), count

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
                if self.authorized_users.get(user) == key:
                    self.user = user
                    return True
                else:
                    self.user = "unknown"
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
                        stream=post_data_dict.get("stream", False),
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
            self.user = "unknown"
            url = urlparse(self.path)
            path = url.path
            if path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"Ollama is running")
                return
            if not self.deactivate_security and not self._validate_user_and_key():
                logger.warning("User is not authorized")
                client_ip, client_port = self.client_address
                # Extract the bearer token from the headers
                auth_header = self.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    self.add_access_log_entry(
                        event="rejected",
                        user="unknown",
                        ip_address=client_ip,
                        access="Denied",
                        server="None",
                        nb_queued_requests_on_server=-1,
                        error="Authentication failed",
                    )
                else:
                    token = auth_header.split(" ")[1]
                    self.add_access_log_entry(
                        event="rejected",
                        user=token,
                        ip_address=client_ip,
                        access="Denied",
                        server="None",
                        nb_queued_requests_on_server=-1,
                        error="Authentication failed",
                    )
                self.send_response(403)
                self.end_headers()
                return
            get_params = parse_qs(url.query) or {}

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
            else:
                post_data = None
                post_data_dict = {}

            # Extract model from post_data or get_params
            model = post_data_dict.get("model")
            if not model:
                model = get_params.get("model", [None])[0]

            logger.debug("Extracted model: %s", model)

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
            stripped_path = re.sub("/$", "", path)
            if stripped_path in model_based_endpoints:
                if not model:
                    # Model is required for these endpoints
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Missing 'model' in request")
                    logger.info("Missing 'model' in request")
                    return

                # Filter servers that support the requested model
                available_servers = [
                    server for server in self.servers if model in server[1]["models"]
                ]

                if not available_servers:
                    # No server supports the requested model
                    logger.info("No servers support model '%s'", model)
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"No servers support the requested model.")
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
                    # Find the server with the lowest queue size among available_servers
                    min_queued_server = min(
                        available_servers, key=lambda s: s[1]["queue"].qsize()
                    )

                    if not self.is_server_available(min_queued_server[1]):
                        logger.info("Server %s is not available.", min_queued_server[0])
                        available_servers.remove(min_queued_server)
                        continue
                    que = min_queued_server[1]["queue"]
                    try:
                        que.put_nowait(1)
                        self.add_access_log_entry(
                            event="gen_request",
                            user=self.user,
                            ip_address=client_ip,
                            access="Authorized",
                            server=min_queued_server[0],
                            nb_queued_requests_on_server=que.qsize(),
                        )
                    except Exception as e:
                        logger.debug("Failed to put request in queue: %s", e)
                        self.add_access_log_entry(
                            event="gen_error",
                            user=self.user,
                            ip_address=client_ip,
                            access="Authorized",
                            server=min_queued_server[0],
                            nb_queued_requests_on_server=que.qsize(),
                            error="Queue is full",
                        )
                        self.send_response(503)
                        self.end_headers()
                        self.wfile.write(b"Server is busy. Please try again later.")
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
                            last_chunk, count = self._send_response(response)
                            if "/v1/" in stripped_path:  # ChatGPT
                                # add stuff
                                try:
                                    info = json.loads(last_chunk.split(b"\n")[-2])
                                except Exception as e:
                                    logger.debug("Exception: %s", e)
                                    try:
                                        info = json.loads(response.content)
                                    except Exception as e:
                                        logger.debug("Exception: %s", e)
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
                            self.add_access_log_entry(
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
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"No available servers could handle the request.")
            elif stripped_path in ["/api/tags"]:
                logger.debug("List supported models")
                models = [
                    model for server in self.servers for model in server[1]["models"]
                ]
                model_info = []
                server_tags = {}
                for server in self.servers:
                    response = self.send_request_with_retries(
                        server[1], path, get_params, post_data_dict, backend_headers
                    )

                    if response:
                        logger.debug("Received response from server %s", server[0])
                    else:
                        logger.debug("Received response from server %s", server[0])
                        self.wfile.write(
                            bytes("Failed to forward request to {server[0]}.")
                        )

                    server_tags[server[0]] = {
                        model["name"]: model
                        for model in json.loads(response.content)["models"]
                    }
                for model in models:
                    available_servers = [
                        server
                        for server in self.servers
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
                model_info = {"models": model_info}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(model_info).encode("utf-8"))
            elif stripped_path in ["/v1/models"]:
                logger.debug("List supported models")
                models = [
                    model for server in self.servers for model in server[1]["models"]
                ]
                model_info = []
                server_tags = {}
                for server in self.servers:
                    response = self.send_request_with_retries(
                        server[1], path, get_params, post_data_dict, backend_headers
                    )

                    if response:
                        logger.debug("Received response from server %s", server[0])
                    else:
                        logger.debug("Received response from server %s", server[0])
                        self.wfile.write(
                            bytes(f"Failed to forward request to {server[0]}.")
                        )

                    server_tags[server[0]] = {
                        model["id"]: model
                        for model in json.loads(response.content)["data"]
                    }
                for model in models:
                    available_servers = [
                        server
                        for server in self.servers
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
                model_info = {"object": "list", "data": model_info}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(model_info).encode("utf-8"))
            elif stripped_path in [
                "/api/pull",
                "/api/delete",
                "/api/push",
                "/api/copy",
                "/api/create",
            ]:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Unsupported in proxy.")
            elif stripped_path in ["/api/ps"]:
                logger.debug("ps servers")
                server_ps = {}
                for server in self.servers:
                    response = self.send_request_with_retries(
                        server[1], path, get_params, post_data_dict, backend_headers
                    )

                    if response:
                        logger.debug("Received response from server %s", server[0])
                    else:
                        logger.debug("Received response from server %s", server[0])
                        self.wfile.write(
                            bytes(f"Failed to forward request to {server[0]}.")
                        )

                    server_ps[server[0]] = json.loads(response.content)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(server_ps).encode("utf-8"))
            else:
                logger.warning("Not recognized path, running : %s", stripped_path)
                # For other endpoints, mirror the request to the default server with retries
                default_server = self.servers[0]
                if not self.is_server_available(default_server[1]):
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"Default server is not available.")
                    return
                response = self.send_request_with_retries(
                    default_server[1], path, get_params, post_data_dict, backend_headers
                )
                if response:
                    self._send_response(response)
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"Failed to forward request to default server.")

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
    main()
