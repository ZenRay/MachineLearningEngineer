## 1. 问题描述


![img.png](default.png)

在该项目中，你将使用强化学习算法，实现一个自动走迷宫机器人。

1. 如上图所示，智能机器人显示在右上角。在我们的迷宫中，有陷阱（红色炸弹）及终点（蓝色的目标点）两种情景。机器人要尽量避开陷阱、尽快到达目的地。
2. 小车可执行的动作包括：向上走 `u`、向右走 `r`、向下走 `d`、向左走 `l`。
3. 执行不同的动作后，根据不同的情况会获得不同的奖励，具体而言，有以下几种情况。
    - 撞到墙壁：-10
    - 走到终点：50
    - 走到陷阱：-30
    - 其余情况：-0.1
4. 我们需要通过修改 `robot.py` 中的代码，来实现一个 Q Learning 机器人，实现上述的目标。

## 2. 完成项目流程

1. 配置环境，使用 `envirnment.yml` 文件配置名为 `robot-env` 的 conda 环境，具体而言，你只需转到当前的目录，在命令行/终端中运行如下代码，稍作等待即可。
```
conda env create -f environment.yml
```
安装完毕后，在命令行/终端中运行 `source activate robot-env`（Mac/Linux 系统）或 `activate robot-env`（Windows 系统）激活该环境。

2. 阅读 `robot_maze.ipynb` 中的指导完成项目，并根据指导修改对应的代码，生成、观察结果。
3. 导出代码与报告，上传文件，提交审阅并优化。

## 3. 经历问题
第一次出现无法安装环境,即使用 `conda env create -f envirnment.yml`出现报错信息,相关内容如下:

```
Solving environment: failed

# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

`$ /Users/username/anaconda3/bin/conda-env create -f environment.yml`

  environment variables:
                 CIO_TEST=<not set>
  CONDA_AUTO_UPDATE_CONDA=false
               CONDA_ROOT=/Users/username/anaconda3
                     PATH=/Users/username/.cargo/bin:/Users/username/anaconda3/bin:/usr/local/bin:/u
                          sr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin
       REQUESTS_CA_BUNDLE=<not set>
            SSL_CERT_FILE=<not set>

     active environment : None
       user config file : /Users/username/.condarc
 populated config files : /Users/username/.condarc
          conda version : 4.5.0
    conda-build version : 3.7.2
         python version : 3.6.5.final.0
       base environment : /Users/username/anaconda3  (writable)
           channel URLs : https://repo.anaconda.com/pkgs/main/osx-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/free/osx-64
                          https://repo.anaconda.com/pkgs/free/noarch
                          https://repo.anaconda.com/pkgs/r/osx-64
                          https://repo.anaconda.com/pkgs/r/noarch
                          https://repo.anaconda.com/pkgs/pro/osx-64
                          https://repo.anaconda.com/pkgs/pro/noarch
          package cache : /Users/username/anaconda3/pkgs
                          /Users/username/.conda/pkgs
       envs directories : /Users/username/anaconda3/envs
                          /Users/username/.conda/envs
               platform : osx-64
             user-agent : conda/4.5.0 requests/2.18.4 CPython/3.6.5 Darwin/17.5.0 OSX/10.13.4
                UID:GID : 501:20
             netrc file : None
           offline mode : False


V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V

CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/osx-64/repodata.json>
Elapsed: -

An HTTP error occurred when trying to retrieve this URL.
HTTP errors are often intermittent, and a simple retry will get you on your way.
ConnectionError(ReadTimeoutError("HTTPSConnectionPool(host='mirrors.tuna.tsinghua.edu.cn', port=443): Read timed out.",),)

A reportable application error has occurred. Conda has prepared the above report.
If submitted, this report will be used by core maintainers to improve
future releases of conda.
Would you like conda to send this report to the core maintainers?
[y/N]: y
Upload successful.

Thank you for helping to improve conda.
Opt-in to always sending reports (and not see this message again)
by running

    $ conda config --set report_errors true
```

**解决方式**: 经过查看发现是运行了一个 `jupyter notebook`,也就是说启用了一个环境.之后关闭该任务后,安装能够正常进行.
