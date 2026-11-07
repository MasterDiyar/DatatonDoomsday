#!/bin/zsh
cd "$(dirname "$0")"
if ! command -v node >/dev/null 2>&1; then
  echo 'Установите Node.js 24 или новее с https://nodejs.org и откройте этот файл снова.'
  read -r 'answer?Нажмите Enter для выхода.'
  exit 1
fi
node --env-file-if-exists=.env scripts/dev.mjs
