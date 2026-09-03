#!/usr/bin/env bash
# 按账号跑 kaggle CLI。用法: kg.sh <账号别名> <kaggle 子命令...>
#
# 🚨 两个写死的路径(读源码得来, 别想当然):
#   kagglesdk/kaggle_env.py:106   token 读 ~/.kaggle/access_token      —— 写死
#   kagglesdk/kaggle_creds.py:15  凭据存 ~/.kaggle/credentials.json    —— 写死
#   => KAGGLE_CONFIG_DIR 只管老式 kaggle.json, 管不了 OAuth token
#   => 而 authenticate() 里 access_token 优先, 所以自己的 token 一直在就永远切不走
# 真正的开关是 KAGGLE_API_TOKEN(值可以是 token 本身, 也可以是存放 token 的文件路径)。
#
# 凭据放法: 别名 me → 用默认(~/.kaggle/access_token)
#           其他别名 → ~/.kaggle-<别名>/access_token  里放那个账号的 token 字符串
set -uo pipefail
ALIAS="${1:-}"
[ -z "$ALIAS" ] && { echo "用法: kg.sh <账号别名> <kaggle 参数...>" >&2; exit 2; }
shift
K="$HOME/.local/bin/kaggle"
PROXY_ON="http_proxy=http://127.0.0.1:6789 https_proxy=http://127.0.0.1:6789"

if [ "$ALIAS" = "me" ]; then
  RUN=(env -u KAGGLE_API_TOKEN -u KAGGLE_CONFIG_DIR)
else
  TOK="$HOME/.kaggle-$ALIAS/access_token"
  if [ ! -s "$TOK" ]; then
    echo "找不到 $TOK(或为空)。" >&2
    echo "让队友在他自己机器上跑: kaggle auth login && kaggle auth print-access-token" >&2
    echo "把打印出来的 token 存进这个文件即可(只存 token 字符串, 无需其他内容)。" >&2
    exit 1
  fi
  chmod 600 "$TOK" 2>/dev/null
  RUN=(env -u KAGGLE_CONFIG_DIR KAGGLE_API_TOKEN="$TOK")
fi

# 直连优先, 失败换 Clash(本机 DNS 一整天不通, 基本都会走代理)
"${RUN[@]}" -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY "$K" "$@" && exit 0
echo "[kg] 直连失败, 换代理重试" >&2
"${RUN[@]}" http_proxy=http://127.0.0.1:6789 https_proxy=http://127.0.0.1:6789 "$K" "$@"
