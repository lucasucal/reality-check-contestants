"""Optional TLS for the class servers used by Whistle."""

import ssl
import sys
from http.server import ThreadingHTTPServer


def tls_server_context(env):
    cert = env.get("RC_POLICY_TLS_CERT")
    key = env.get("RC_POLICY_TLS_KEY")
    if not (cert or key):
        return None
    if not (cert and key):
        raise ValueError("set both RC_POLICY_TLS_CERT and RC_POLICY_TLS_KEY to serve HTTPS")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        context.load_cert_chain(cert, key, password="")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load policy TLS certificate and key: {exc}") from exc
    return context


class PolicyHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler, *, tls_context=None):
        super().__init__(address, handler)
        if tls_context is not None:
            try:
                # Defer each handshake to a worker so it cannot block accepts.
                self.socket = tls_context.wrap_socket(
                    self.socket, server_side=True, do_handshake_on_connect=False)
            except (OSError, ValueError):
                self.server_close()
                raise

    def process_request_thread(self, request, client_address):
        if isinstance(request, ssl.SSLSocket):
            timeout = request.gettimeout()
            request.settimeout(5)
            try:
                request.do_handshake()
            except OSError as exc:
                print(f"[rc_policy] TLS handshake failed: {exc}", file=sys.stderr, flush=True)
                self.shutdown_request(request)
                return
            request.settimeout(timeout)
        super().process_request_thread(request, client_address)
