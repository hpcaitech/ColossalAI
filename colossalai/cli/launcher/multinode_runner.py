import fabric
from .hostinfo import HostInfo, HostInfoList
from typing import List
from multiprocessing import Pipe, Process
import click


def run_on_host(hostinfo, workdir, recv_conn, send_conn, env):
    fab_conn = fabric.Connection(hostinfo.hostname, port=hostinfo.port)
    finish = False
    while not finish:
        cmds = recv_conn.recv()
        if cmds == 'exit':
            finish = True
            break
        else:
            try:
                with fab_conn.cd(workdir):
                    if hostinfo.is_local_host:
                        fab_conn.local(cmds, hide=False)
                    else:
                        env_msg = ' '.join([f'{k}=\"{v}\"' for k, v in env.items()])
                        with fab_conn.prefix(f"export {env_msg}"):
                            fab_conn.run(cmds, hide=False, env=env)
                    send_conn.send('success')
            except:
                click.echo(f"Error: failed to run {cmds} on {hostinfo.hostname}")
                send_conn.send('failure')
    send_conn.send("finish")
    fab_conn.close()


class MultiNodeRunner:

    def __init__(self,):
        self.processes = {}
        self.master_send_conns = {}
        self.master_recv_conns = {}

    def add_export(self, key, var):
        self.exports[key.strip()] = var.strip()

    def connect(self, host_info_list: HostInfoList, workdir: str, env: dict):
        for hostinfo in host_info_list:
            master_send_conn, worker_recv_conn = Pipe()
            master_recv_conn, worker_send_conn = Pipe()
            p = Process(target=run_on_host, args=(hostinfo, workdir, worker_recv_conn, worker_send_conn, env))
            p.start()
            self.processes[hostinfo.hostname] = p
            self.master_recv_conns[hostinfo.hostname] = master_recv_conn
            self.master_send_conns[hostinfo.hostname] = master_send_conn

    def send_to_remote(self, hostinfo, cmd):
        assert hostinfo.hostname in self.master_send_conns, \
            f'{hostinfo} is not found in the current connections'
        conn = self.master_send_conns[hostinfo.hostname]
        conn.send(cmd)

    def stop_all(self):
        for hostname, conn in self.master_send_conns.items():
            conn.send('exit')

    def recv_from_all(self):
        msg_from_node = dict()
        for hostname, conn in self.master_recv_conns.items():
            msg_from_node[hostname] = conn.recv()
        return msg_from_node
