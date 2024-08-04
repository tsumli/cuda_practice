import os
import pwd
import grp
import click

@click.command()
@click.option('--root_dir', required=True, help='Root directory to set permissions and ownership')
@click.option('--user', required=True, help='User name to set permissions and ownership')
@click.option('--group', required=True, help='Group name to set permissions and ownership')
def set_permissions_and_ownership(root_dir, user, group):
    uid = pwd.getpwnam(user).pw_uid
    gid = grp.getgrnam(group).gr_gid

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # .gitディレクトリを除外
        if '.git' in dirnames:
            dirnames.remove('.git')

        for dirname in dirnames:
            dir_full_path = os.path.join(dirpath, dirname)
            print(f'Setting directory permissions 777 and ownership to {user}:{group}: {dir_full_path}')
            os.chmod(dir_full_path, 0o777)
            os.chown(dir_full_path, uid, gid)

        for filename in filenames:
            file_full_path = os.path.join(dirpath, filename)
            print(f'Setting file permissions 666 and ownership to {user}:{group}: {file_full_path}')
            os.chmod(file_full_path, 0o666)
            os.chown(file_full_path, uid, gid)

if __name__ == "__main__":
    set_permissions_and_ownership()
