# Duck 底座相关 (2026-08-25)

- `field_guide_en.txt`: 种类手册英文版, 作为 FIELD_GUIDE_ADDENDUM 追加进 Duck 系统提示词。
- `handbook-source.patch`: 对 Tufa 源码包(jeroencottaar/taaf-kaggle-source-share)的改动 = 上传为数据集 jinbowang1/taaf-source-handbook。
- `hub-thinking-disabled.patch`: 本地 reference/duck-harness 的改动(HUB_THINKING_DISABLED=1 时请求带 thinking:disabled; DUCK_FIELD_GUIDE=<文件> 时追加手册)。
- `inference.hub-ds*.json`: 本地跑 Duck 对接 hub DeepSeek 的配置(无图/关思考/离线环境目录)。
- `kernels/k_duck_official`: Tufa 官方 notebook 原样 → jinbowang1/arc3-duck-official; `kernels/k_duck_handbook`: 只换源码包为手册版 → jinbowang1/arc3-duck-handbook。
- `run_duck_*.sh`: 本地对照脚本(🚨 必须 unset http_proxy, Python requests 走 Clash 每轮读超时); `submit_handbook.sh`: 等 kernel 跑完→五项核验→提交。
- 08-25 结论: vc33 三种子 Duck 2关 58/172/129 步 vs dsh 2关 317/400/400 步; 手册 A/B(DeepSeek, 3局×2遍) 加手册 2.5 关 vs 1.5 关(ft09/ls20 赢, r11l 回退) → 今晚提交手册版。
