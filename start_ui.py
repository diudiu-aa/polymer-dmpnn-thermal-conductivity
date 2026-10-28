"""Install once in a dedicated local environment, then open the browser UI."""
import hashlib
from pathlib import Path
import subprocess
import sys
import venv


def main():
    root = Path(__file__).resolve().parent
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        print('请先安装Python 3.10–3.13（建议3.12），再用该版本运行start_ui.py。')
        return 1
    environment = root / '.ui-venv'
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    marker = environment / 'installed-source.sha256'
    paths = [root / 'pyproject.toml'] + sorted((root / 'polymer_dmpnn').rglob('*.py')) + sorted((root / 'polymer_dmpnn/web_assets').glob('*.html'))
    digest = hashlib.sha256()
    for p in paths:
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    fingerprint = digest.hexdigest()
    if not python.exists():
        print('首次启动：正在创建专用Python环境。', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    if not marker.exists() or marker.read_text().strip() != fingerprint:
        print('正在安装模型和依赖；首次安装需要联网，请等待终端完成。', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', str(root)], cwd=root, check=True)
        marker.write_text(fingerprint + '\n')
    print('安装已就绪，正在打开浏览器。终端需要保持运行。', flush=True)
    return subprocess.call([str(python), '-m', 'polymer_dmpnn.web', '--repository', str(root)], cwd=root)

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('本地界面已关闭。')
        raise SystemExit(0)
    except (OSError, subprocess.CalledProcessError) as exc:
        print('启动未完成。请保留终端报错，并按README手动安装或反馈。\n' + str(exc))
        raise SystemExit(1)
