from multiprocessing import Pipe, Process
from multiprocessing import connection as mp_connection

import click
import fabric

from .hostinfo import HostInfo, HostInfoList


def run_on_host(hostinfo: HostInfo, workdir: str, recv_conn: mp_connection.Connection,
                send_conn: mp_connection.Connection, env: dict) -> None:
    """
    Use fabric connection to execute command on local or remote hosts.

    Args:
        hostinfo (HostInfo): host information
        workdir (str): the directory to execute the command
        recv_conn (multiprocessing.connection.Connection): receive messages from the master sender
        send_conn (multiprocessing.connection.Connection): send messages to the master receiver
        env (dict): a dictionary for environment variables
    """

    fab_conn = fabric.Connection(hostinfo.hostname, port=hostinfo.port)
    finish = False
    env_msg = ' '.join([f'{k}=\"{v}\"' for k, v in env.items()])

    # keep listening until exit
    while not finish:
        # receive cmd
        cmds = recv_conn.recv()

        if cmds == 'exit':
            # exit from the loop
            finish = True
            break
        else:
            # execute the commands
            try:
                # cd to execute directory
                with fab_conn.cd(workdir):
                    # propagate the runtime environment
                    with fab_conn.prefix(f"export {env_msg}"):
                        if hostinfo.is_local_host:
                            # execute on the local machine
                            fab_conn.local(cmds, hide=False)
                        else:
                            # execute on the remote machine
                            fab_conn.run(cmds, hide=False)
                    send_conn.send('success')
            except Exception as e:
                click.echo(
                    f"Error: failed to run {cmds} on {hostinfo.hostname}, is localhost: {hostinfo.is_local_host}, exception: {e}"
                )
                send_conn.send('failure')

    # shutdown
    send_conn.send("finish")
    fab_conn.close()


class MultiNodeRunner:
    """
    A runner to execute commands on an array of machines. This runner
    is inspired by Nezha (https://github.com/zhuzilin/NeZha).
    """

    def __init__(self):
        self.processes = {}
        self.master_send_conns = {}
        self.master_recv_conns = {}

    def connect(self, host_info_list: HostInfoList, workdir: str, env: dict) -> None:
        """
        Establish connections to a list of hosts

        Args:
            host_info_list (HostInfoList): a list of HostInfo objects
            workdir (str): the directory where command is executed
            env (dict): environment variables to propagate to hosts
        """
        for hostinfo in host_info_list:
            master_send_conn, worker_recv_conn = Pipe()
            master_recv_conn, worker_send_conn = Pipe()
            p = Process(target=run_on_host, args=(hostinfo, workdir, worker_recv_conn, worker_send_conn, env))
            p.start()
            self.processes[hostinfo.hostname] = p
            self.master_recv_conns[hostinfo.hostname] = master_recv_conn
            self.master_send_conns[hostinfo.hostname] = master_send_conn

    def send(self, hostinfo: HostInfo, cmd: str) -> None:
        """
        Send a command to a local/remote host.

        Args:
            hostinfo (HostInfo): host information
            cmd (str): the command to execute
        """

        assert hostinfo.hostname in self.master_send_conns, \
            f'{hostinfo} is not found in the current connections'
        conn = self.master_send_conns[hostinfo.hostname]
        conn.send(cmd)

    def stop_all(self) -> None:
        """
        Stop connections to all hosts.
        """

        for hostname, conn in self.master_send_conns.items():
            conn.send('exit')

    def recv_from_all(self) -> dict:
        """
        Receive messages from all hosts

        Returns:
            msg_from_node (dict): a dictionry which contains messages from each node
        """

        msg_from_node = dict()
        for hostname, conn in self.master_recv_conns.items():
            msg_from_node[hostname] = conn.recv()
        return msg_from_node
