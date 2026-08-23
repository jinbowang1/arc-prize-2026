# dsh(deepseek-harness) 攻关配方 (2026-08-23, 三局全破)

战果: r11l L1@27步 · ft09 L1-L4连破(44/70/87/110步) · ls20 L1@59步
(各局第二关均预算尽未破; 全部 deepseek-v4-flash 官方API, 关思考)

跑法:
1. 游戏侧: uv run python -m kaggle_agent.game_server --game <gid> --port 18999 --max-actions 200
2. agent侧: cd ~/Desktop/project/arc3-dsh-workspace &&
   DSH_HOME=$PWD/.dsh-home DEEPSEEK_API_KEY=<官方key> \
   node ~/Desktop/project/deepseek-harness/apps/cli/lib/bin.js \
     --profile headless --patch <本目录>/arc3-official.patch.yml "$(cat TASK_FULL.md)"

三件套缺一不可(消融实证):
- game_server 观察层(对象级diff/颜色账本/计数器去噪): 裸格子dsh也难打
- arc3-official.patch.yml 关思考(thinking:disabled+reasoningEffort:'off'):
  默认high一轮1分钟, 18分钟只走3步
- TASK_FULL.md 种类手册: ft09无手册128步0关, 有手册4关连杀;
  ls20无强化B类200步漫游全灭, 强化后59步过
⚠️独立 DSH_HOME 必须: 共享 ~/.dsh 的 settings.yaml 会用 sd-pro 盖掉 patch
