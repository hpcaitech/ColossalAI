from typing import List
from xml.dom import NotFoundErr
import socket


class HostInfo:

    def __init__(
        self,
        hostname: str,
        port: str = None,
    ):
        self.hostname = hostname
        self.port = port
        self.is_local_host = HostInfo.is_host_localhost(hostname, port)

    @staticmethod
    def is_host_localhost(hostname, port=None):
        if port is None:
            port = 22    # no port specified, lets just use the ssh port
        hostname = socket.getfqdn(hostname)
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            return True
        localhost = socket.gethostname()
        localaddrs = socket.getaddrinfo(localhost, port)
        targetaddrs = socket.getaddrinfo(hostname, port)
        for (family, socktype, proto, canonname, sockaddr) in localaddrs:
            for (rfamily, rsocktype, rproto, rcanonname, rsockaddr) in targetaddrs:
                if rsockaddr[0] == sockaddr[0]:
                    return True
        return False

    def __str__(self):
        return f'hostname: {self.hostname}, port: {self.port}, slots: {self.slots}'

    def __repr__(self):
        return self.__str__()


class HostInfoList:

    def __init__(self):
        self.hostinfo_list = []

    def append(self, hostinfo: HostInfo):
        self.hostinfo_list.append(hostinfo)

    def remove(self, hostname: str):
        hostinfo = self.get_hostinfo(hostname)
        self.hostinfo_list.remove(hostinfo)

    def get_hostinfo(self, hostname: str):
        for hostinfo in self.hostinfo_list:
            if hostinfo.hostname == hostname:
                return hostinfo

        raise NotFoundErr(f"Hostname {hostname} is not found")

    def has(self, hostname: str):
        for hostinfo in self.hostinfo_list:
            if hostinfo.hostname == hostname:
                return True
        return False

    def __iter__(self):
        return iter(self.hostinfo_list)

    def __len__(self):
        return len(self.hostinfo_list)
