import argparse
import paramiko
import os


class SSHTool():
    """Credit belongs to: http://stackoverflow.com/a/10432669 for this tool."""
    def __init__(self, host, user, auth,
                 via=None, via_user=None, via_auth=None):
        if via:
            t0 = paramiko.Transport(via)
            t0.start_client()
            t0.auth_password(via_user, via_auth)
            # setup forwarding from 127.0.0.1:<free_random_port> to |host|
            channel = t0.open_channel('direct-tcpip', host, ('127.0.0.1', 0))
            self.transport = paramiko.Transport(channel)
        else:
            self.transport = paramiko.Transport(host)
        self.transport.start_client()
        self.transport.auth_password(user, auth)

    def run(self, cmd):
        ch = self.transport.open_session()
        ch.set_combine_stderr(True)
        ch.exec_command(cmd)
        retcode = ch.recv_exit_status()
        buf = ''
        while ch.recv_ready():
            buf += ch.recv(1024)
        return buf, retcode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synfire GPU', fromfile_prefix_chars='@')

    # Flags & Optional arguments
    parser.add_argument('-v', '--verbose', action='store_true', help="detailed output.")
    parser.add_argument('-q', '--quiet', action='store_true', help="completely quiet.")
    parser.add_argument('-l', '--local', type=str, default='.', nargs='?', help="local source path")
    parser.add_argument('-d', '--dest', type=str, default='.', nargs='?', help="remote destination")

    # Positional arguments
    parser.add_argument('username', type=str, help="SSH username")
    parser.add_argument('password', type=str, help="SSH password")
    parser.add_argument('host', type=str, default='gpu.cs.fsu.edu', help="Destination server")
    parser.add_argument('via', type=str, nargs='?', help="Pass through server to host.")

    args = parser.parse_args()

    if args.via is None:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(args.host, username=args.username, password=args.password)
        # stdin, stdout, stderr = ssh.exec_command('uname -a')
        # type(stdout)
        # print(stdout.readlines())
    else:
        ssht = SSHTool((args.host, 22), args.username, args.password,
                       via=(args.via, 22), via_user=args.username, via_auth=args.password)

        # print ssht.run('uname -a')
        sftp = paramiko.SFTPClient.from_transport(ssht.transport)